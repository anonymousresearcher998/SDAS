from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.sdas.layout import (
    ExperimentLayout,
    atomic_torch_save,
    atomic_write_json,
    prompt_condition,
    read_json,
    read_jsonl,
)
from src.sdas.prompts import prompt_group

POOLINGS = ("resp_mean", "resp_mean_k16", "last_prompt")

@dataclass
class Pairs:
    ids: List[str]
    langs: List[str]
    prompts: List[str]
    layers: List[int]
    H_reg: torch.Tensor
    H_int: torch.Tensor
    n_reg: torch.Tensor
    n_int: torch.Tensor
    pooling: str = "resp_mean"

    @property
    def n(self) -> int:
        return len(self.ids)

    def delta(self) -> torch.Tensor:
        return self.H_int - self.H_reg

    def subset(self, mask: torch.Tensor) -> "Pairs":
        idx = mask.nonzero(as_tuple=True)[0].tolist()
        return Pairs(
            ids=[self.ids[i] for i in idx],
            langs=[self.langs[i] for i in idx],
            prompts=[self.prompts[i] for i in idx],
            layers=list(self.layers),
            H_reg=self.H_reg[idx],
            H_int=self.H_int[idx],
            n_reg=self.n_reg[idx],
            n_int=self.n_int[idx],
            pooling=self.pooling,
        )


def _layer_indices(stored: Sequence[int], wanted: Optional[Sequence[int]]) -> Tuple[List[int], List[int]]:
    stored = [int(x) for x in stored]
    if wanted is None:
        return list(range(len(stored))), stored
    missing = [l for l in wanted if l not in stored]
    if missing:
        raise ValueError(f"layers {missing} were not captured (stored: {stored})")
    return [stored.index(l) for l in wanted], [int(l) for l in wanted]


def load_pairs(
    layout: ExperimentLayout,
    intervention: str,
    *,
    langs: Sequence[str],
    prompts: Sequence[str],
    pooling: str = "resp_mean",
    ids: Optional[Sequence[str]] = None,
    layers: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> Pairs:
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}")
    wanted_ids = set(ids) if ids is not None else None

    reg_list: List[torch.Tensor] = []
    int_list: List[torch.Tensor] = []
    out_ids: List[str] = []
    out_langs: List[str] = []
    out_prompts: List[str] = []
    n_reg: List[int] = []
    n_int: List[int] = []
    layer_numbers: Optional[List[int]] = None
    dropped = {"nan": 0, "missing_pair": 0}

    for lang in langs:
        for prompt_name in prompts:
            reg_dir = layout.generation_dir("regular", lang, prompt_name)
            int_dir = layout.generation_dir(prompt_condition(intervention), lang, prompt_name)
            if not (reg_dir / "acts.pt").is_file() or not (int_dir / "acts.pt").is_file():
                if verbose:
                    print(f"[pairs] {lang}/{prompt_name}: missing acts.pt in regular or prompt/{intervention}; skipped")
                continue
            reg_acts = torch.load(reg_dir / "acts.pt")
            int_acts = torch.load(int_dir / "acts.pt")
            reg_rows = {r["id"]: r for r in read_jsonl(reg_dir / "gen.jsonl")}
            int_rows = {r["id"]: r for r in read_jsonl(int_dir / "gen.jsonl")}
            sel, nums = _layer_indices(reg_acts["layers"], layers)
            sel_int, nums_int = _layer_indices(int_acts["layers"], nums)
            if layer_numbers is None:
                layer_numbers = nums
            elif layer_numbers != nums:
                raise ValueError(f"{lang}/{prompt_name}: layer set {nums} differs from {layer_numbers}")

            reg_pos = {i: k for k, i in enumerate(reg_acts["ids"])}
            int_pos = {i: k for k, i in enumerate(int_acts["ids"])}
            for item_id in reg_acts["ids"]:
                if wanted_ids is not None and item_id not in wanted_ids:
                    continue
                if item_id not in int_pos or item_id not in reg_rows or item_id not in int_rows:
                    dropped["missing_pair"] += 1
                    continue
                h_reg = reg_acts[pooling][reg_pos[item_id]][sel]
                h_int = int_acts[pooling][int_pos[item_id]][sel_int]
                if torch.isnan(h_reg).any() or torch.isnan(h_int).any():
                    dropped["nan"] += 1
                    continue
                reg_list.append(h_reg.float())
                int_list.append(h_int.float())
                out_ids.append(item_id)
                out_langs.append(lang)
                out_prompts.append(prompt_name)
                n_reg.append(int(reg_rows[item_id].get("n_tokens", 0)))
                n_int.append(int(int_rows[item_id].get("n_tokens", 0)))

    if not out_ids:
        raise ValueError(
            f"no usable pairs for prompt/{intervention} (langs={list(langs)}, prompts={list(prompts)}, "
            f"dropped={dropped}) - run Stage A first"
        )
    if verbose:
        print(f"[pairs] {len(out_ids)} pairs kept, dropped {dropped}, pooling={pooling}, layers={layer_numbers}")
    return Pairs(
        ids=out_ids,
        langs=out_langs,
        prompts=out_prompts,
        layers=list(layer_numbers or []),
        H_reg=torch.stack(reg_list),
        H_int=torch.stack(int_list),
        n_reg=torch.tensor(n_reg, dtype=torch.int32),
        n_int=torch.tensor(n_int, dtype=torch.int32),
        pooling=pooling,
    )

def _unit(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def compute_direction(pairs: Pairs, balance_by: Optional[str] = None) -> torch.Tensor:
    delta = pairs.delta()
    if not balance_by or balance_by == "none":
        return delta.mean(dim=0)
    keys = pairs.prompts if balance_by == "prompt" else pairs.langs if balance_by == "lang" else None
    if keys is None:
        raise ValueError("balance_by must be 'none', 'prompt' or 'lang'")
    means = []
    for key in sorted(set(keys)):
        mask = torch.tensor([k == key for k in keys])
        means.append(delta[mask].mean(dim=0))
    return torch.stack(means).mean(dim=0)


def _group_direction(delta: torch.Tensor, keys: Sequence[str], key: str) -> Optional[torch.Tensor]:
    mask = torch.tensor([k == key for k in keys])
    return delta[mask].mean(dim=0) if int(mask.sum()) > 0 else None


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 3:
        return float("nan")
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = (x.norm() * y.norm()).clamp_min(1e-12)
    return float((x @ y) / denom)


def diagnostics(
    pairs: Pairs,
    d: torch.Tensor,
    *,
    split_half_k: int = 10,
    seed: int = 0,
    compare_to: Optional[torch.Tensor] = None,
    min_group: int = 5,
) -> Dict[str, object]:
    delta = pairs.delta()
    n, L, _ = delta.shape
    gen = torch.Generator().manual_seed(seed)
    halves = [torch.randperm(n, generator=gen) for _ in range(split_half_k)] if n >= 4 else []
    langs = sorted(set(pairs.langs))
    groups = sorted(set(prompt_group(p) for p in pairs.prompts))
    prompt_names = sorted(set(pairs.prompts))
    group_keys = [prompt_group(p) for p in pairs.prompts]
    len_diff = (pairs.n_int - pairs.n_reg).float()

    per_layer: Dict[str, Dict[str, object]] = {}
    for j, layer in enumerate(pairs.layers):
        D = delta[:, j, :]
        dl = d[j]
        d_hat = _unit(dl)
        norms = D.norm(dim=1)
        rec: Dict[str, object] = {
            "n": n,
            "norm_d": float(dl.norm()),
            "mean_norm_delta": float(norms.mean()),
            "norm_ratio": float(dl.norm() / norms.mean().clamp_min(1e-12)),
            "mean_resid_norm_regular": float(pairs.H_reg[:, j, :].norm(dim=1).mean()),
        }
        if n > 1:
            Dm = (n * dl.unsqueeze(0) - D) / (n - 1)
            cos = (D * Dm).sum(1) / (norms * Dm.norm(dim=1)).clamp_min(1e-12)
            rec["loo_frac_pos_cos"] = float((cos > 0).float().mean())
            rec["loo_mean_cos"] = float(cos.mean())
        proj_reg = pairs.H_reg[:, j, :] @ d_hat
        proj_int = pairs.H_int[:, j, :] @ d_hat
        rec["proj_regular_mean"], rec["proj_regular_std"] = float(proj_reg.mean()), float(proj_reg.std(unbiased=False))
        rec["proj_int_mean"], rec["proj_int_std"] = float(proj_int.mean()), float(proj_int.std(unbiased=False))
        rec["proj_by_prompt_regular"] = {}
        rec["proj_by_prompt_int"] = {}
        for p in prompt_names:
            mask = torch.tensor([q == p for q in pairs.prompts])
            rec["proj_by_prompt_regular"][p] = float(proj_reg[mask].mean())
            rec["proj_by_prompt_int"][p] = float(proj_int[mask].mean())
        if halves:
            cs = []
            for perm in halves:
                a, b = perm[: n // 2], perm[n // 2 :]
                cs.append(cosine(D[a].mean(0), D[b].mean(0)))
            rec["split_half_cos_mean"] = float(sum(cs) / len(cs))
            rec["split_half_cos_std"] = float(torch.tensor(cs).std(unbiased=False)) if len(cs) > 1 else 0.0
        rec["subgroup_cos"] = {}
        rec["subgroup_norm"] = {}
        rec["subgroup_n"] = {}
        for label, keys, members in (("lang", pairs.langs, langs), ("group", group_keys, groups), ("prompt", pairs.prompts, prompt_names)):
            for key in members:
                mask = torch.tensor([k == key for k in keys])
                cnt = int(mask.sum())
                if cnt < min_group:
                    continue
                dg = D[mask].mean(0)
                rec["subgroup_cos"][f"{label}:{key}"] = cosine(dl, dg)
                rec["subgroup_norm"][f"{label}:{key}"] = float(dg.norm())
                rec["subgroup_n"][f"{label}:{key}"] = cnt
        if len(langs) == 2:
            da = _group_direction(D, pairs.langs, langs[0])
            db = _group_direction(D, pairs.langs, langs[1])
            if da is not None and db is not None:
                rec[f"cos_{langs[0]}_{langs[1]}"] = cosine(da, db)
        rec["mean_n_tokens_regular"] = float(pairs.n_reg.float().mean())
        rec["mean_n_tokens_int"] = float(pairs.n_int.float().mean())
        rec["corr_norm_delta_vs_length_diff"] = _pearson(norms, len_diff)
        if compare_to is not None:
            rec["cos_compare_to"] = cosine(dl, compare_to[j])
        per_layer[str(layer)] = rec

    return {"layers": list(pairs.layers), "n_pairs": n, "pooling": pairs.pooling, "per_layer": per_layer}


def random_like(d: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    r = torch.randn(d.shape, generator=gen)
    return r * (d.norm(dim=1, keepdim=True) / r.norm(dim=1, keepdim=True).clamp_min(1e-12))


def orthogonalize(d: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    o_hat = _unit(other)
    return d - (d * o_hat).sum(dim=1, keepdim=True) * o_hat


def save_direction(direction_dir: Path, d: torch.Tensor, meta: Dict[str, object]) -> None:
    payload = {
        "layers": [int(l) for l in meta["layers"]],
        "d": d.float().contiguous(),
        "pooling": meta.get("pooling"),
        "intervention": meta.get("intervention"),
        "train_ids": list(meta.get("train_ids", [])),
        "val_ids": list(meta.get("val_ids", [])),
        "test_ids": list(meta.get("test_ids", [])),
        "n_pairs": int(meta.get("n_pairs", 0)),
        "name": meta.get("name"),
    }
    atomic_torch_save(direction_dir / "direction.pt", payload)
    atomic_write_json(direction_dir / "meta.json", meta)


def load_direction(direction_dir: Path) -> Tuple[torch.Tensor, Dict[str, object]]:
    direction_dir = Path(direction_dir)
    if not (direction_dir / "direction.pt").is_file():
        raise FileNotFoundError(f"{direction_dir / 'direction.pt'} not found - run learn_direction.py first")
    payload = torch.load(direction_dir / "direction.pt")
    meta = read_json(direction_dir / "meta.json") if (direction_dir / "meta.json").is_file() else {}
    meta.setdefault("layers", payload["layers"])
    for key in ("train_ids", "val_ids", "test_ids", "pooling", "intervention", "n_pairs", "name"):
        meta.setdefault(key, payload.get(key))
    return payload["d"].float(), meta


def direction_vector(d: torch.Tensor, layers: Sequence[int], layer: int) -> torch.Tensor:
    layers = [int(l) for l in layers]
    if layer not in layers:
        raise ValueError(f"layer {layer} not in the direction's layers {layers}")
    return d[layers.index(layer)]


RELEASED_VECTOR_RE = re.compile(r"^layer_(\d+)\.pt$")


def load_released_vector(path: Path) -> Tuple[int, torch.Tensor]:
    path = Path(path)
    match = RELEASED_VECTOR_RE.match(path.name)
    if not match:
        raise ValueError(f"{path}: a released vector is named layer_<LL>.pt")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    v = torch.load(path, weights_only=True)
    if not isinstance(v, torch.Tensor) or v.dim() != 1:
        raise ValueError(f"{path}: expected a 1-D tensor, got {type(v).__name__} {getattr(v, 'shape', '')}")
    if not torch.isfinite(v).all():
        raise ValueError(f"{path}: the vector has non-finite values")
    return int(match.group(1)), v.float()
