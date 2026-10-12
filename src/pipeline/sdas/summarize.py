from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.env import load_env
from src.pipeline.sdas._common import add_experiment_dir_arg, banner, parse_csv
from src.sdas.data import SPLITS, in_split
from src.sdas.layout import (
    ExperimentLayout,
    atomic_write_text,
    load_experiment,
    now_ts,
    parse_condition,
    read_json,
    read_jsonl,
)
from src.sdas.prompts import experiment_pair, prompt_group, resolve_langs
from src.sdas.stats import bh_fdr, cluster_bootstrap_mean, mean, paired_cluster_test, paired_flips, paired_gap_test

Key = Tuple[str, str]

def load_generations(layout: ExperimentLayout, condition: str, lang: str, split: Optional[str] = None) -> Dict[Key, Dict]:
    out: Dict[Key, Dict] = {}
    for lg, prompt_name, gen_dir in layout.list_leaves(condition):
        if lg != lang:
            continue
        for row in read_jsonl(gen_dir / "gen.jsonl"):
            if in_split(row["id"], split):
                out[(prompt_name, row["id"])] = row
    return out


def run_dirs(prompt_dir: Path, lang: str) -> List[Path]:
    if not prompt_dir.is_dir():
        return []
    return sorted(p for p in prompt_dir.iterdir() if p.is_dir() and p.name.startswith(f"{lang}_run_") and (p / f"{lang}_results.jsonl").is_file())


def load_judged(
    layout: ExperimentLayout, condition: str, bucket: str, judge_prompt: str, translator: str, lang: str, split: Optional[str] = None
) -> Dict[Key, Dict]:
    out: Dict[Key, Dict] = {}
    base = layout.judged_dir / condition.strip("/") / f"{bucket}_judge" / judge_prompt / translator
    if not base.is_dir():
        return out
    for prompt_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for run in run_dirs(prompt_dir, lang):
            for row in read_jsonl(run / f"{lang}_results.jsonl"):
                if in_split(row["id"], split):
                    out[(prompt_dir.name, row["id"])] = row
    return out


def detect_judges(layout: ExperimentLayout) -> List[Tuple[str, str]]:
    found = set()
    if not layout.judged_dir.is_dir():
        return []
    for bucket_dir in layout.judged_dir.rglob("*_judge"):
        if "_anchors" in bucket_dir.parts:
            continue
        for jp in bucket_dir.iterdir():
            if jp.is_dir():
                found.add((bucket_dir.name[: -len("_judge")], jp.name))
    return sorted(found)

def row_metrics(row: Dict) -> Dict[str, Any]:
    return {
        "chars": len(row["mt"]),
        "finish_length": row.get("finish") == "length",
    }


def paired_contrast(cond: Dict[Key, float], reg: Dict[Key, float], n_boot: int, n_perm: int, seed: int) -> Dict[str, Any]:
    keys = [k for k in cond if k in reg and cond[k] is not None and reg[k] is not None]
    by_item: Dict[str, List[float]] = defaultdict(list)
    for k in keys:
        by_item[k[1]].append(cond[k] - reg[k])
    if not keys:
        return {"n_pairs": 0, "n_items": 0, "mean": None, "lo": None, "hi": None, "p": None}
    m, lo, hi = cluster_bootstrap_mean(by_item, n_boot=n_boot, seed=seed)
    _, p = paired_cluster_test(by_item, n_perm=n_perm, seed=seed)
    return {"n_pairs": len(keys), "n_items": len(by_item), "mean": m, "lo": lo, "hi": hi, "p": p}


def _r(x: Any, nd: int = 3) -> Any:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(x, nd) if isinstance(x, float) else x


def summarize_condition(
    *,
    condition: str,
    lang: str,
    gens: Dict[Key, Dict],
    judged: Dict[Key, Dict],
    regular_gens: Dict[Key, Dict],
    regular_judged: Dict[Key, Dict],
    tau: float,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    same_key = f"score_{lang}_mt_vs_{lang}_human"
    is_regular = condition == "regular"
    parsed = parse_condition(condition)

    metrics = {k: row_metrics(r) for k, r in gens.items()}

    out: Dict[str, Any] = {
        "condition": condition,
        "lang": lang,
        "kind": parsed["kind"],
        "layer": parsed.get("layer"),
        "alpha": parsed.get("alpha"),
        "seed_override": parsed.get("seed"),
        "rep": parsed.get("rep"),
        "n_rows": len(gens),
        "finish_length": _r(mean([m["finish_length"] for m in metrics.values()])),
    }
    if metrics:
        g = list(metrics.values())
        out["mean_chars"] = _r(mean([m["chars"] for m in g]), 1)
    if parsed["kind"] == "sdas":
        out["exact_match"] = _r(mean([bool(r.get("exact_match_regular")) for r in gens.values()]))
        proj_keys = sorted({pk for r in gens.values() for pk in (r.get("proj") or {})})
        for pk in proj_keys:
            vals = [r["proj"][pk] for r in gens.values() if r.get("proj") and r["proj"].get(pk) is not None]
            out[f"proj_{pk}"] = _r(mean(vals)) if vals else None

    same = {k: j.get(same_key) for k, j in judged.items() if k in gens and j.get(same_key) is not None}
    out["n_judged"] = len(same)
    out["n_judge_fail"] = sum(1 for j in judged.values() if j.get(same_key) is None)
    if same:
        out["same_mean"] = _r(mean(list(same.values())))
        out["share_ge_0.5"] = _r(mean([1.0 if v >= 0.5 else 0.0 for v in same.values()]))

    if not is_regular and regular_judged:
        reg_same = {k: j.get(same_key) for k, j in regular_judged.items() if k in regular_gens and j.get(same_key) is not None}
        c = paired_contrast(same, reg_same, n_boot, n_perm, seed)
        out["d_same"] = _r(c["mean"])
        out["d_same_lo"] = _r(c["lo"])
        out["d_same_hi"] = _r(c["hi"])
        out["p_same"] = _r(c["p"], 4)
        out["n_paired_same"] = c["n_pairs"]
        flips = paired_flips({k: v for k, v in reg_same.items()}, same, tau=tau)
        out["flips_up"], out["flips_down"], out["flips_same"] = flips["n_up"], flips["n_down"], flips["n_same"]

    # per prompt
    by_prompt: Dict[str, Dict[str, Any]] = {}
    for prompt_name in sorted({k[0] for k in gens}):
        keys = [k for k in gens if k[0] == prompt_name]
        rec: Dict[str, Any] = {
            "condition": condition, "lang": lang, "prompt": prompt_name, "group": prompt_group(prompt_name),
            "n": len(keys),
            "same_mean": _r(mean([same[k] for k in keys if k in same])) if any(k in same for k in keys) else None,
        }
        if not is_regular and regular_judged:
            rs = {k: regular_judged[k].get(same_key) for k in keys if k in regular_judged and k in regular_gens}
            ds = [same[k] - rs[k] for k in keys if k in same and rs.get(k) is not None]
            rec["d_same"] = _r(mean(ds)) if ds else None
        by_prompt[prompt_name] = rec
    return out, by_prompt, same


def gap_table(
    ordered: List[str], same_by: Dict[Tuple[str, str], Dict[Key, float]], *, n_boot: int, n_perm: int, seed: int,
    examples_out: Optional[List[Dict[str, Any]]] = None, reference: str = "regular", sides: Tuple[str, str] = ("ru", "ua"),
) -> List[Dict[str, Any]]:
    a, b = sides
    reg = {lang: same_by.get((lang, reference), {}) for lang in sides}
    if not reg[a] and not reg[b]:
        print(f"[gap] reference condition {reference!r} has no judged rows - every Δgap column stays empty (pass --reference <slug>)", flush=True)
    rows: List[Dict[str, Any]] = []
    for condition in ordered:
        s = {lang: same_by.get((lang, condition), {}) for lang in sides}
        if not s[a] and not s[b]:
            continue
        parsed = parse_condition(condition)
        d_a, d_b = mean(list(s[a].values())), mean(list(s[b].values()))
        rec: Dict[str, Any] = {
            "condition": condition, "kind": parsed["kind"], "layer": parsed.get("layer"), "alpha": parsed.get("alpha"), "rep": parsed.get("rep"),
            f"n_{a}": len(s[a]), f"n_{b}": len(s[b]), f"d_{a}": _r(d_a), f"d_{b}": _r(d_b),
            "gap": _r(d_b - d_a) if s[a] and s[b] else None,
        }
        if condition != reference:
            by_item: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {a: [], b: []})
            dd = {}
            for lang in sides:
                diffs = {k: s[lang][k] - reg[lang][k] for k in s[lang] if k in reg[lang]}
                dd[lang] = mean(list(diffs.values())) if diffs else None
                for (_, item_id), v in diffs.items():
                    by_item[item_id][lang].append(v)
            t = paired_gap_test(by_item, n_boot=n_boot, n_perm=n_perm, seed=seed, sides=sides)
            rec.update({f"dd_{a}": _r(dd[a]), f"dd_{b}": _r(dd[b])})
            rec.update(d_gap=_r(t["mean"]), d_gap_lo=_r(t["lo"]), d_gap_hi=_r(t["hi"]), p_gap=_r(t["p"], 4), n_items=t["n_items"])
            # complete cases: the per-example paired gap difference (the headline test)
            cc = [k for k in s[a] if k in s[b] and k in reg[a] and k in reg[b]]
            if cc:
                g_c = {k: s[b][k] - s[a][k] for k in cc}
                g_r = {k: reg[b][k] - reg[a][k] for k in cc}
                delta = {k: g_c[k] - g_r[k] for k in cc}
                dabs = {k: abs(g_c[k]) - abs(g_r[k]) for k in cc}
                vals = list(delta.values())
                n = len(vals)
                m = mean(vals)
                sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else float("nan")
                _, p_ex = paired_cluster_test({f"{k[0]}|{k[1]}": [v] for k, v in delta.items()}, n_perm=n_perm, seed=seed)
                cc_by_item: Dict[str, List[float]] = defaultdict(list)
                for (_, item_id), v in delta.items():
                    cc_by_item[item_id].append(v)
                _, p_cl = paired_cluster_test(cc_by_item, n_perm=n_perm, seed=seed)
                _, lo_cc, hi_cc = cluster_bootstrap_mean(cc_by_item, n_boot=n_boot, seed=seed)
                m_abs, p_abs = paired_cluster_test({f"{k[0]}|{k[1]}": [v] for k, v in dabs.items()}, n_perm=n_perm, seed=seed)
                shrank = sum(1 for v in dabs.values() if v < -1e-12)
                grew = sum(1 for v in dabs.values() if v > 1e-12)
                rec.update(
                    n_complete=n, d_gap_cc=_r(m), sd_gap_cc=_r(sd), se_gap_cc=_r(sd / math.sqrt(n), 4),
                    d_gap_cc_lo=_r(lo_cc), d_gap_cc_hi=_r(hi_cc), p_gap_ex=_r(p_ex, 4), p_gap_cluster=_r(p_cl, 4),
                    d_absgap=_r(m_abs), p_absgap_ex=_r(p_abs, 4), share_shrank=_r(shrank / n), share_grew=_r(grew / n),
                )
                if examples_out is not None:
                    for k in sorted(cc):
                        examples_out.append({
                            "condition": condition, "prompt": k[0], "id": k[1],
                            f"{a}_regular": reg[a][k], f"{b}_regular": reg[b][k], f"{a}_cond": s[a][k], f"{b}_cond": s[b][k],
                            "gap_regular": _r(g_r[k], 4), "gap_cond": _r(g_c[k], 4), "delta_gap": _r(delta[k], 4), "delta_absgap": _r(dabs[k], 4),
                        })
        rows.append(rec)
    steer = [r for r in rows if r["kind"] == "sdas" and r.get("p_gap") is not None]
    for r, q in zip(steer, bh_fdr([r["p_gap"] for r in steer])):
        r["q_gap"] = _r(q, 4)
    steer_ex = [r for r in rows if r["kind"] == "sdas" and r.get("p_gap_ex") is not None]
    for r, q in zip(steer_ex, bh_fdr([r["p_gap_ex"] for r in steer_ex])):
        r["q_gap_ex"] = _r(q, 4)
    noise = max((abs(r["d_gap"]) for r in rows if r["kind"] == "sdas" and r.get("alpha") == 0 and (r.get("rep") is not None) and r.get("d_gap") is not None), default=0.0)
    gap_reg = next((r["gap"] for r in rows if r["condition"] == reference), None)
    for r in rows:
        if r["condition"] != reference and r.get("gap") is not None and gap_reg is not None:
            r["gap_closed"] = abs(r["gap"]) < abs(gap_reg) - noise
    if rows:
        rows[0]["_gap_noise"] = noise
    return rows


def proj_bands(layout: ExperimentLayout, condition: str) -> Dict[str, Dict[str, float]]:
    parsed = parse_condition(condition)
    if parsed["kind"] != "sdas":
        return {}
    diag_path = layout.direction_dir(parsed["direction"]) / "diagnostics.json"
    if not diag_path.is_file():
        return {}
    per_layer = read_json(diag_path).get("per_layer", {})
    bands: Dict[str, Dict[str, float]] = {}
    for layer_str, rec in per_layer.items():
        bands[f"L{int(layer_str):02d}"] = {"regular": rec.get("proj_regular_mean"), "int": rec.get("proj_int_mean")}
    return bands


def selection_rule(out: Dict[str, Any], noise_same: float, bands: Dict) -> None:
    if out["kind"] != "sdas":
        return
    d_same = out.get("d_same")
    moved = None
    for pk, val in ((k, v) for k, v in out.items() if k.startswith("proj_post_L")):
        band = bands.get(pk[len("proj_post_"):])
        if band and val is not None and band["regular"] is not None and band["int"] is not None:
            moved = (val - band["regular"]) * (band["int"] - band["regular"]) > 0
            break
    out["proj_moved"] = moved
    out["same_improved"] = d_same is not None and d_same > abs(noise_same)
    out["selected"] = bool(out["same_improved"] and (moved is not False))

CSV_COLUMNS = [
    "condition", "lang", "kind", "layer", "alpha", "seed_override", "rep", "n_rows", "finish_length", "mean_chars", "exact_match",
    "n_judged", "n_judge_fail", "same_mean", "share_ge_0.5",
    "d_same", "d_same_lo", "d_same_hi", "p_same", "q_same",
    "flips_up", "flips_down", "flips_same",
    "same_improved", "proj_moved", "selected",
]


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def md_table(rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in columns) + " |")
    return lines


def gap_columns(a: str = "ru", b: str = "ua") -> List[str]:
    return [
        "condition", "kind", "layer", "alpha", "rep", f"n_{a}", f"n_{b}", f"d_{a}", f"d_{b}", "gap",
        "n_complete", "d_gap_cc", "sd_gap_cc", "se_gap_cc", "d_gap_cc_lo", "d_gap_cc_hi", "p_gap_ex", "q_gap_ex", "p_gap_cluster",
        "d_absgap", "p_absgap_ex", "share_shrank", "share_grew",
        f"dd_{a}", f"dd_{b}", "d_gap", "d_gap_lo", "d_gap_hi", "p_gap", "q_gap", "n_items", "gap_closed",
    ]


def gap_example_columns(a: str = "ru", b: str = "ua") -> List[str]:
    return ["condition", "prompt", "id", f"{a}_regular", f"{b}_regular", f"{a}_cond", f"{b}_cond", "gap_regular", "gap_cond", "delta_gap", "delta_absgap"]


GAP_COLUMNS = gap_columns()
GAP_EXAMPLE_COLUMNS = gap_example_columns()


def write_outputs(
    out_dir: Path, rows: List[Dict[str, Any]], by_prompt: List[Dict[str, Any]], notes: List[str],
    extra_cols: List[str], gap_rows: Optional[List[Dict[str, Any]]] = None, gap_examples: Optional[List[Dict[str, Any]]] = None,
    sides: Tuple[str, str] = ("ru", "ua"),
) -> None:
    a, b = sides
    gap_cols, gap_ex_cols = gap_columns(a, b), gap_example_columns(a, b)
    out_dir.mkdir(parents=True, exist_ok=True)
    if gap_rows:
        with open(out_dir / "summary_gap.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=gap_cols, extrasaction="ignore")
            w.writeheader()
            for r in gap_rows:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in gap_cols})
    if gap_examples:
        with open(out_dir / "summary_gap_examples.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=gap_ex_cols, extrasaction="ignore")
            w.writeheader()
            for r in gap_examples:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in gap_ex_cols})
    columns = CSV_COLUMNS + [c for c in extra_cols if c not in CSV_COLUMNS]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    bp_cols = ["condition", "lang", "prompt", "group", "n", "same_mean", "d_same"]
    with open(out_dir / "summary_by_prompt.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bp_cols, extrasaction="ignore")
        w.writeheader()
        for r in by_prompt:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in bp_cols})

    lines = ["# SDAS summary", "", *notes, ""]
    if gap_rows:
        label = f"{a.upper()}/{b.upper()}"
        lines += [f"## {label} gap - per-example paired test (complete cases; δ = gap_cond - gap_regular per event × prompt)", ""]
        lines += md_table(gap_rows, ["condition", "n_complete", f"d_{a}", f"d_{b}", "gap", "d_gap_cc", "se_gap_cc", "d_gap_cc_lo", "d_gap_cc_hi", "p_gap_ex", "q_gap_ex", "p_gap_cluster", "d_absgap", "p_absgap_ex", "share_shrank", "share_grew"])
        lines += ["", f"## {label} gap - row means (all judged rows; Δ vs regular, event-clustered)", ""]
        lines += md_table(gap_rows, ["condition", f"n_{a}", f"n_{b}", "gap", f"dd_{a}", f"dd_{b}", "d_gap", "d_gap_lo", "d_gap_hi", "p_gap", "q_gap", "gap_closed"])
        lines.append("")
    for lang in sorted({r["lang"] for r in rows}):
        lr = [r for r in rows if r["lang"] == lang]
        lines += [f"## {lang}", "", "### Generations", ""]
        lines += md_table(lr, ["condition", "n_rows", "finish_length", "mean_chars", "exact_match"])
        lines += ["", "### Judge scores (same-side; contrasts are item-paired vs regular)", ""]
        lines += md_table(lr, ["condition", "n_judged", "n_judge_fail", "same_mean", "share_ge_0.5", "d_same", "d_same_lo", "d_same_hi", "q_same", "flips_up", "flips_down"])
        steer = [r for r in lr if r["kind"] == "sdas"]
        if steer:
            proj_cols = sorted({k for r in steer for k in r if k.startswith(("proj_pre_", "proj_post_"))})
            lines += ["", "### Steer cells - projections and selection rule", ""]
            lines += md_table(steer, ["condition"] + proj_cols + ["same_improved", "proj_moved", "selected"])
        groups = sorted({r["group"] for r in by_prompt if r["lang"] == lang})
        lines += ["", "### By prompt group (same-side mean / Δ vs regular)", ""]
        grp_rows = []
        for cond in [r["condition"] for r in lr]:
            for g in groups:
                sub = [r for r in by_prompt if r["lang"] == lang and r["condition"] == cond and r["group"] == g]
                if not sub:
                    continue
                grp_rows.append({
                    "condition": cond, "group": g, "n": sum(r["n"] for r in sub),
                    "same_mean": _r(mean([r["same_mean"] for r in sub if r["same_mean"] is not None])),
                    "d_same": _r(mean([r["d_same"] for r in sub if r.get("d_same") is not None])),
                })
        lines += md_table(grp_rows, ["condition", "group", "n", "same_mean", "d_same"])
        lines.append("")
    atomic_write_text(out_dir / "summary.md", "\n".join(lines))


def full_summary(layout: ExperimentLayout, args: argparse.Namespace) -> None:
    experiment = load_experiment(layout)
    pair = experiment_pair(experiment)
    a, b = pair.langs
    A, B = a.upper(), b.upper()
    translator = experiment["translator_model_name"]
    judges = detect_judges(layout)
    bucket, judge_prompt = args.judge_model, args.judge_prompt_name
    if bucket is None or judge_prompt is None:
        if len(judges) == 1:
            bucket, judge_prompt = judges[0]
        elif not judges:
            print("[summary] no judged runs found - summary will hold generation-side metrics only")
        else:
            raise SystemExit(f"several judges found {judges}; pass --judge_model and --judge_prompt_name")
    conditions = layout.list_conditions()
    if args.conditions.strip().lower() != "all":
        wanted = set(parse_csv(args.conditions))
        conditions = [c for c in conditions if c in wanted]
    if "regular" not in conditions:
        raise SystemExit("the regular condition is missing - run Stage A first")
    ordered = ["regular"] + sorted(c for c in conditions if c.startswith("prompt/")) + sorted(
        (c for c in conditions if c.startswith("sdas/")),
        key=lambda c: (parse_condition(c)["direction"], parse_condition(c)["layer"], parse_condition(c)["alpha"], parse_condition(c)["rep"] or 0, parse_condition(c)["seed"] or -1),
    )
    split = None if args.ids == "all" else args.ids
    banner(
        "Stage F - summary",
        [("experiment", str(layout.root)), ("ids", args.ids), ("langs", args.langs), ("conditions", len(ordered)), ("judge", f"{bucket} / {judge_prompt}" if bucket else "none"), ("tau", args.tau)],
    )

    rows: List[Dict[str, Any]] = []
    by_prompt: List[Dict[str, Any]] = []
    same_by: Dict[Tuple[str, str], Dict[Key, float]] = {}
    notes = [
        f"Experiment `{layout.experiment}`; items of the `{args.ids}` split; judge `{bucket}` / `{judge_prompt}`.",
        "`d_*` = item-paired contrast vs regular with a cluster-bootstrap "
        "95 % CI and a sign-flip permutation p (items as clusters); `q_*` = BH-FDR across steer cells within a language. "
        f"Flips use τ = {args.tau}.",
    ]
    for lang in parse_csv(args.langs):
        regular_gens = load_generations(layout, "regular", lang, split)
        if not regular_gens:
            n_any = len(load_generations(layout, "regular", lang))
            hint = (
                f" ({n_any} rows exist but none carry the `{split}_` id prefix - a single-file experiment? pass --ids all)"
                if n_any and split else ""
            )
            print(f"[summary] no regular generations for {lang}{hint}; skipping")
            continue
        regular_judged = load_judged(layout, "regular", bucket, judge_prompt, translator, lang, split) if bucket else {}
        lang_rows: List[Dict[str, Any]] = []
        for condition in ordered:
            gens = regular_gens if condition == "regular" else load_generations(layout, condition, lang, split)
            if not gens:
                continue
            judged = regular_judged if condition == "regular" else (load_judged(layout, condition, bucket, judge_prompt, translator, lang, split) if bucket else {})
            out, bp, same_scores = summarize_condition(
                condition=condition, lang=lang, gens=gens, judged=judged, regular_gens=regular_gens, regular_judged=regular_judged,
                tau=args.tau, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed,
            )
            same_by[(lang, condition)] = same_scores
            lang_rows.append(out)
            by_prompt.extend(bp.values())
            print(f"[{lang}] {condition}: rows={out['n_rows']} same={fmt(out.get('same_mean'))} d_same={fmt(out.get('d_same'))}", flush=True)
        noise = [r for r in lang_rows if r["kind"] == "sdas" and r.get("alpha") == 0 and (r.get("seed_override") is not None or r.get("rep") is not None)]
        noise_same = max((abs(r["d_same"]) for r in noise if r.get("d_same") is not None), default=0.0)
        notes.append(f"[{lang}] noise floor from {len(noise)} alpha=0 re-sampled cell(s): |Δsame| <= {noise_same:.3f}" + ("" if noise else " (none present - run a Stage-C cell with --alphas 0 and --rep <r> or --seed <other>)"))
        steer_rows = [r for r in lang_rows if r["kind"] == "sdas"]
        qs = bh_fdr([r.get("p_same") if r.get("p_same") is not None else float("nan") for r in steer_rows])
        for r, q in zip(steer_rows, qs):
            r["q_same"] = _r(q, 4)
        for r in steer_rows:
            selection_rule(r, noise_same, proj_bands(layout, r["condition"]))
        rows.extend(lang_rows)

    gap_rows: List[Dict[str, Any]] = []
    gap_examples: List[Dict[str, Any]] = []
    langs_done = {lang for lang, _ in same_by}
    if args.ids == "test":
        print(f"[summary] test split: gap statistics are computed by src/pipeline/sdas/test_stats.py over {pair.results_root}/{{baseline,self_distance,sdas_test}} (last run per prompt); skipping the in-flow gap table")
    elif set(pair.langs) <= langs_done and bucket:
        gap_rows = gap_table(ordered, same_by, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed, examples_out=gap_examples, reference=args.reference, sides=pair.langs)
        noise = gap_rows[0].pop("_gap_noise", 0.0) if gap_rows else 0.0
        notes.append(
            f"{A}/{B} gap = d_{B} - d_{A}; reference condition = `{args.reference}`. Per-example test (`*_cc`): for every event × prompt with all four scores, "
            f"δ = (d_{B} - d_{A})_cond - (d_{B} - d_{A})_reference; mean, SD, SE and a sign-flip permutation p over the examples "
            f"(`p_gap_ex`; each example flips independently), the event-clustered p and bootstrap CI beside it, BH q across the run's "
            f"steer cells (`q_gap_ex`); `d_absgap` = mean(|gap_cond| - |gap_regular|) with the share of examples whose |gap| shrank / grew. "
            f"Row-mean Δgap (`d_gap`, all judged rows, event-clustered) kept for reference; noise floor |Δgap| <= {noise:.3f} from the "
            f"alpha=0 replicate; `gap_closed` = |gap| moved toward 0 beyond it. Per-example rows: summary_gap_examples.csv."
        )
        for r in gap_rows:
            print(f"[gap] {r['condition']}: d_{a}={fmt(r.get(f'd_{a}'))} d_{b}={fmt(r.get(f'd_{b}'))} gap={fmt(r.get('gap'))} | n_cc={r.get('n_complete', '')} d_gap_cc={fmt(r.get('d_gap_cc'))} se={fmt(r.get('se_gap_cc'))} p_ex={fmt(r.get('p_gap_ex'))} q_ex={fmt(r.get('q_gap_ex'))} p_cluster={fmt(r.get('p_gap_cluster'))} | d_absgap={fmt(r.get('d_absgap'))} shrank/grew={fmt(r.get('share_shrank'))}/{fmt(r.get('share_grew'))}", flush=True)
    extra = sorted({k for r in rows for k in r if k.startswith(("proj_pre_", "proj_post_"))})
    out_dir = layout.summary_dir(f"{args.ids}_{now_ts()}") if args.out is None else Path(args.out)
    write_outputs(out_dir, rows, by_prompt, notes, extra, gap_rows, gap_examples, sides=pair.langs)
    print(f"\nwrote {out_dir / 'summary.csv'}, summary_by_prompt.csv, summary.md" + (", summary_gap.csv, summary_gap_examples.csv" if gap_rows else ""))


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="Stage F of the SDAS flow: cross-condition summary.")
    add_experiment_dir_arg(ap)
    ap.add_argument("--langs", default=None, help="Comma-separated languages [the experiment's pair: ru,ua | ps,il]")
    ap.add_argument("--ids", default="val", choices=[*SPLITS, "all"], help="Split to summarize: val | test | train | all [val]")
    ap.add_argument("--conditions", default="all", help="Comma-separated condition slugs to include [all]")
    ap.add_argument("--reference", default="regular", help="Condition the gap table pairs every cell against [regular]")
    ap.add_argument("--judge_model", default=None, help="Judge bucket name")
    ap.add_argument("--judge_prompt_name", default=None, help="Judge prompt dir")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Output directory [summary/<ts>]")
    args = ap.parse_args()

    layout = ExperimentLayout.from_dir(args.experiment_dir)
    args.langs = ",".join(resolve_langs(args.langs, experiment_pair(load_experiment(layout))))
    full_summary(layout, args)


if __name__ == "__main__":
    main()
