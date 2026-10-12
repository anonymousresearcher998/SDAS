from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.pipeline.sdas._common import banner, parse_csv
from src.sdas.layout import atomic_write_text, now_ts, read_jsonl
from src.sdas.prompts import PAIRS, pair_cfg
from src.sdas.stats import abs_gap_test, bh_fdr, cluster_bootstrap_mean, mean, paired_cluster_test

Key = Tuple[str, str]


def last_run(prompt_dir: Path, lang: str) -> Optional[Path]:
    runs = sorted(p for p in prompt_dir.iterdir() if p.is_dir() and p.name.startswith(f"{lang}_run_") and (p / f"{lang}_results.jsonl").is_file()) if prompt_dir.is_dir() else []
    return runs[-1] if runs else None


def load_scores(model_dir: Path, lang: str) -> Tuple[Dict[Key, float], Dict[str, str]]:
    scores: Dict[Key, float] = {}
    runs_used: Dict[str, str] = {}
    key = f"score_{lang}_mt_vs_{lang}_human"
    for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        run = last_run(prompt_dir, lang)
        if run is None:
            continue
        runs_used[prompt_dir.name] = run.name
        for r in read_jsonl(run / f"{lang}_results.jsonl"):
            if r.get(key) is not None:
                scores[(prompt_dir.name, r["id"])] = float(r[key])
    return scores, runs_used


def _sd(vals: List[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def compare(
    mit: Dict[str, Dict[Key, float]], ref: Dict[str, Dict[Key, float]], *, n_boot: int, n_perm: int, seed: int,
    sides: Tuple[str, str] = ("ru", "ua"),
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    a, b = sides
    cc = [k for k in mit[a] if k in mit[b] and k in ref[a] and k in ref[b]]
    if not cc:
        return {"n_complete": 0}, []
    g_m = {k: mit[b][k] - mit[a][k] for k in cc}
    g_r = {k: ref[b][k] - ref[a][k] for k in cc}
    delta = {k: g_m[k] - g_r[k] for k in cc}
    dabs = {k: abs(g_m[k]) - abs(g_r[k]) for k in cc}
    vals = list(delta.values())
    n = len(vals)
    m, sd = mean(vals), _sd(vals)
    _, p_ex = paired_cluster_test({f"{k[0]}|{k[1]}": [v] for k, v in delta.items()}, n_perm=n_perm, seed=seed)
    by_item: Dict[str, List[float]] = defaultdict(list)
    for (_, item_id), v in delta.items():
        by_item[item_id].append(v)
    _, p_cl = paired_cluster_test(by_item, n_perm=n_perm, seed=seed)
    _, lo, hi = cluster_bootstrap_mean(by_item, n_boot=n_boot, seed=seed)
    m_abs, p_abs = paired_cluster_test({f"{k[0]}|{k[1]}": [v] for k, v in dabs.items()}, n_perm=n_perm, seed=seed)
    shrank = sum(1 for v in dabs.values() if v < -1e-12)
    grew = sum(1 for v in dabs.values() if v > 1e-12)
    dd_a = mean([mit[a][k] - ref[a][k] for k in cc])
    dd_b = mean([mit[b][k] - ref[b][k] for k in cc])
    rec = {
        "n_complete": n, f"dd_{a}": round(dd_a, 4), f"dd_{b}": round(dd_b, 4), "d_gap": round(m, 4), "sd": round(sd, 4), "se": round(sd / math.sqrt(n), 4),
        "ci_lo_gap": round(lo, 4), "ci_hi_gap": round(hi, 4), "p_ex": round(p_ex, 4), "p_cluster": round(p_cl, 4),
        "d_absgap_ex": round(m_abs, 4), "p_absgap_ex": round(p_abs, 4), "share_shrank": round(shrank / n, 4), "share_grew": round(grew / n, 4),
    }
    rec.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in abs_gap_test(mit, ref, n_boot=n_boot, n_perm=n_perm, seed=seed, sides=sides).items()})
    examples = [{"prompt": k[0], "id": k[1], f"{a}_baseline": ref[a][k], f"{b}_baseline": ref[b][k], f"{a}_mit": mit[a][k], f"{b}_mit": mit[b][k],
                 "gap_baseline": round(g_r[k], 4), "gap_mit": round(g_m[k], 4), "delta_gap": round(delta[k], 4), "delta_absgap": round(dabs[k], 4)} for k in sorted(cc)]
    return rec, examples


def columns(a: str = "ru", b: str = "ua") -> List[str]:
    return ["model", "mitigation", "in_family", f"n_{a}", f"n_{b}", f"d_{a}", f"d_{b}", "gap", "n_events", "absgap_ref", "absgap", "d_absgap", "ci_lo", "ci_hi", "p", "q",
            "n_complete", f"dd_{a}", f"dd_{b}", "d_gap", "sd", "se", "ci_lo_gap", "ci_hi_gap", "p_ex", "q_ex_all", "p_cluster", "d_absgap_ex", "p_absgap_ex", "share_shrank", "share_grew", f"runs_{a}", f"runs_{b}"]


COLUMNS = columns()


def main() -> None:
    ap = argparse.ArgumentParser(description="Test-set statistics over baseline vs injection vs steering cells.")
    ap.add_argument("--pair", default="ru_ua", choices=sorted(PAIRS), help="Dataset: ru_ua (results/) or il_ps (results_il_ps/) [ru_ua]")
    ap.add_argument("--results_root", default=None, help="Results root [the pair's: results | results_il_ps]")
    ap.add_argument("--judge", default="llama3-8b")
    ap.add_argument("--judge_prompt", default="BASE_JUDGE")
    ap.add_argument("--models", default="all", help="Translator models, comma-separated [all with cells under sdas_test]")
    ap.add_argument("--reference", default="baseline", help="Mitigation dir used as the regular reference [baseline]")
    ap.add_argument("--injection", default="self_distance", help="Mitigation dir of the prompt injection [self_distance]; '' to skip")
    ap.add_argument("--sdas", default="sdas_test", help="Mitigation dir holding the exported steer cells [sdas_test]")
    ap.add_argument("--include_injection", action=argparse.BooleanOptionalAction, default=True, help="Put the injection into the BH family too [yes]")
    ap.add_argument("--direction", default="self_distance_v1",
                    help="Steering direction of the cells to read [self_distance_v1: <model>_L..]; another direction's cells are <model>_<direction>_L..")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Output dir [<results_root>/<steering>/_stats/<ts>]")
    args = ap.parse_args()
    pair = pair_cfg(args.pair)
    a, b = pair.langs
    A, B = a.upper(), b.upper()
    if args.results_root is None:
        args.results_root = pair.results_root

    root = Path(args.results_root)
    base = lambda mit: root / mit / f"{args.judge}_judge" / args.judge_prompt
    steer_dir = base(args.sdas)
    pseudo = sorted(p.name for p in steer_dir.iterdir() if p.is_dir()) if steer_dir.is_dir() else []
    cell_infix = "_L" if args.direction == "self_distance_v1" else f"_{args.direction}_L"
    if args.models.strip().lower() == "all":
        models = sorted({p.rsplit(cell_infix, 1)[0] for p in pseudo if cell_infix in p})
    else:
        models = parse_csv(args.models)
    if not models:
        raise SystemExit(f"no exported steer cells under {steer_dir}")
    banner("Test-set statistics", [("results root", str(root)), ("judge", f"{args.judge} / {args.judge_prompt}"), ("reference", args.reference), ("injection", args.injection or "-"), ("sdas", f"{args.sdas} ({args.direction})"), ("models", ", ".join(models))])

    rows: List[Dict[str, object]] = []
    examples: List[Dict[str, object]] = []
    for model in models:
        ref = {lang: load_scores(base(args.reference) / model, lang) for lang in pair.langs}
        if not ref[a][0] or not ref[b][0]:
            print(f"[{model}] no {args.reference} rows in both languages - skipped")
            continue
        mits: List[Tuple[str, Path, bool]] = [(args.reference, base(args.reference) / model, False)]
        if args.injection and (base(args.injection) / model).is_dir():
            mits.append((args.injection, base(args.injection) / model, args.include_injection))
        mits += [(f"{args.sdas}/{p}", steer_dir / p, "_a0" not in p) for p in pseudo if p.startswith(model + cell_infix)]
        for name, mdir, in_family in mits:
            sc = {lang: load_scores(mdir, lang) for lang in pair.langs}
            rec: Dict[str, object] = {
                "model": model, "mitigation": name, "in_family": in_family,
                f"n_{a}": len(sc[a][0]), f"n_{b}": len(sc[b][0]),
                f"d_{a}": round(mean(list(sc[a][0].values())), 4) if sc[a][0] else None,
                f"d_{b}": round(mean(list(sc[b][0].values())), 4) if sc[b][0] else None,
                f"runs_{a}": f"{len(sc[a][1])} prompts", f"runs_{b}": f"{len(sc[b][1])} prompts",
            }
            if rec[f"d_{a}"] is not None and rec[f"d_{b}"] is not None:
                rec["gap"] = round(rec[f"d_{b}"] - rec[f"d_{a}"], 4)
            if name != args.reference and sc[a][0] and sc[b][0]:
                cmp, ex = compare({a: sc[a][0], b: sc[b][0]}, {a: ref[a][0], b: ref[b][0]}, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed, sides=pair.langs)
                rec.update(cmp)
                for e in ex:
                    e.update(model=model, mitigation=name)
                examples.extend(ex)
            rows.append(rec)
            print(f"  {model:26s} {name:40s} n={rec[f'n_{a}']}/{rec[f'n_{b}']} d_{a}={rec.get(f'd_{a}')} d_{b}={rec.get(f'd_{b}')} gap={rec.get('gap')} | n_cc={rec.get('n_complete', '')} d_absgap={rec.get('d_absgap', '')} ci=[{rec.get('ci_lo', '')}, {rec.get('ci_hi', '')}] p={rec.get('p', '')}", flush=True)
    family = [r for r in rows if r["in_family"] and r.get("p_ex") is not None]
    for r, q in zip(family, bh_fdr([r["p_ex"] for r in family])):
        r["q_ex_all"] = round(q, 4)
    for r, q in zip(family, bh_fdr([r["p"] for r in family])):
        r["q"] = round(q, 4)

    out = Path(args.out) if args.out else root / args.sdas / "_stats" / now_ts()
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out / "test_stats.csv", "w", newline="", encoding="utf-8") as f:
        cols = columns(a, b)
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    ex_cols = ["model", "mitigation", "prompt", "id", f"{a}_baseline", f"{b}_baseline", f"{a}_mit", f"{b}_mit", "gap_baseline", "gap_mit", "delta_gap", "delta_absgap"]
    with open(out / "test_examples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ex_cols, extrasaction="ignore"); w.writeheader()
        for e in examples:
            w.writerow(e)
    md_cols = ["model", "mitigation", f"n_{a}", f"n_{b}", f"d_{a}", f"d_{b}", "gap", "n_events", "absgap_ref", "absgap", "d_absgap", "ci_lo", "ci_hi", "p", "q"]
    md_supp = ["model", "mitigation", "n_complete", f"dd_{a}", f"dd_{b}", "d_gap", "se", "ci_lo_gap", "ci_hi_gap", "p_ex", "q_ex_all", "d_absgap_ex", "share_shrank", "share_grew"]
    table = lambda cols: ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |" for r in rows]
    lines = [f"# Test-set statistics - last run per prompt, judge {args.judge}/{args.judge_prompt}", "",
             f"Paper statistic (Eq. 5, App. H): d_absgap = |d_{B} - d_{A}|_mitigation - |d_{B} - d_{A}|_{args.reference}, gaps of the aggregate means; "
             f"95% CI = event-clustered bootstrap ({args.n_boot} resamples); `p` = two-sided event-level paired permutation ({args.n_perm} block swaps); "
             f"`q` = BH over the one family of {len(family)} hypotheses (all steering cells of all models{', plus the injection' if args.include_injection else ''}).", "",
             *table(md_cols), "",
             "## Supplementary diagnostics", "",
             f"delta = (d_{B} - d_{A})_mitigation - (d_{B} - d_{A})_{args.reference} per complete-case example (prompt x event); `d_gap` = its mean, CI over events; "
             "`p_ex` = sign-flip permutation over the examples, `q_ex_all` = BH over the same family; `p_cluster` in the CSV = the same test with events as clusters; "
             "`d_absgap_ex` = mean per-example change in |gap|.", "",
             *table(md_supp)]
    atomic_write_text(out / "test_stats.md", "\n".join(lines) + "\n")
    print(f"\nwrote {out / 'test_stats.csv'}, test_stats.md, test_examples.csv")


if __name__ == "__main__":
    main()
