from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


def mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def cluster_bootstrap_mean(
    values_by_item: Dict[str, Sequence[float]], n_boot: int = 2000, seed: int = 0
) -> Tuple[float, float, float]:
    items = [k for k, v in values_by_item.items() if len(v) > 0]
    if not items:
        return float("nan"), float("nan"), float("nan")
    observed = mean([x for k in items for x in values_by_item[k]])
    rng = random.Random(seed)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [items[rng.randrange(len(items))] for _ in items]
        boots.append(mean([x for k in sample for x in values_by_item[k]]))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(math.ceil(0.975 * n_boot)) - 1)]
    return observed, lo, hi


def paired_cluster_test(diff_by_item: Dict[str, Sequence[float]], n_perm: int = 5000, seed: int = 0) -> Tuple[float, float]:
    items = [k for k, v in diff_by_item.items() if len(v) > 0]
    if not items:
        return float("nan"), float("nan")
    sums = [sum(diff_by_item[k]) for k in items]
    counts = [len(diff_by_item[k]) for k in items]
    total = sum(counts)
    observed = sum(sums) / total
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        stat = sum(s if rng.random() < 0.5 else -s for s in sums) / total
        if abs(stat) >= abs(observed) - 1e-12:
            hits += 1
    return observed, (hits + 1) / (n_perm + 1)


def paired_gap_test(
    by_item: Dict[str, Dict[str, Sequence[float]]], n_boot: int = 2000, n_perm: int = 5000, seed: int = 0,
    sides: Tuple[str, str] = ("ru", "ua"),
) -> Dict[str, float]:
    a, b = sides
    items = [k for k, v in by_item.items() if v.get(a) or v.get(b)]
    if not items:
        return {"n_items": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}

    def stat(sample: Sequence[str], signs: Sequence[float] = ()) -> float:
        xa: List[float] = []
        xb: List[float] = []
        for j, k in enumerate(sample):
            s = signs[j] if signs else 1.0
            xa.extend(s * x for x in by_item[k].get(a, ()))
            xb.extend(s * x for x in by_item[k].get(b, ()))
        return (sum(xb) / len(xb) if xb else 0.0) - (sum(xa) / len(xa) if xa else 0.0)

    observed = stat(items)
    rng = random.Random(seed)
    boots = sorted(stat([items[rng.randrange(len(items))] for _ in items]) for _ in range(n_boot))
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(math.ceil(0.975 * n_boot)) - 1)]
    hits = 0
    for _ in range(n_perm):
        signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in items]
        if abs(stat(items, signs)) >= abs(observed) - 1e-12:
            hits += 1
    return {"n_items": len(items), "mean": observed, "lo": lo, "hi": hi, "p": (hits + 1) / (n_perm + 1)}


def abs_gap_test(
    mit: Dict[str, Dict[Tuple[str, str], float]], ref: Dict[str, Dict[Tuple[str, str], float]], n_boot: int = 2000,
    n_perm: int = 5000, seed: int = 0, sides: Tuple[str, str] = ("ru", "ua"),
) -> Dict[str, float]:
    a, b = sides
    events = sorted({k[1] for rec in (mit, ref) for side in (a, b) for k in rec[side]})
    col = {e: j for j, e in enumerate(events)}

    def sums(scores: Dict[Tuple[str, str], float]) -> Tuple[np.ndarray, np.ndarray]:
        s, c = np.zeros(len(events)), np.zeros(len(events))
        for (_, e), v in scores.items():
            s[col[e]] += v
            c[col[e]] += 1
        return s, c

    def abs_gap(b_sum, b_cnt, a_sum, a_cnt):
        return np.abs(b_sum / b_cnt - a_sum / a_cnt)

    m_b, m_a, r_b, r_a = sums(mit[b]), sums(mit[a]), sums(ref[b]), sums(ref[a])
    g_mit = abs_gap(m_b[0].sum(), m_b[1].sum(), m_a[0].sum(), m_a[1].sum())
    g_ref = abs_gap(r_b[0].sum(), r_b[1].sum(), r_a[0].sum(), r_a[1].sum())
    observed = float(g_mit - g_ref)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(events), size=(n_boot, len(events)))
    tot = [arr[idx].sum(axis=1) for pair in (m_b, m_a, r_b, r_a) for arr in pair]
    lo, hi = np.percentile(abs_gap(*tot[0:4]) - abs_gap(*tot[4:8]), [2.5, 97.5])
    swap = rng.integers(0, 2, size=(n_perm, len(events))).astype(float)
    perm_mit, perm_ref = [], []
    for mv, rv in ((m_b[0], r_b[0]), (m_b[1], r_b[1]), (m_a[0], r_a[0]), (m_a[1], r_a[1])):
        moved = swap @ (rv - mv)
        perm_mit.append(mv.sum() + moved)
        perm_ref.append(rv.sum() - moved)
    perms = abs_gap(*perm_mit) - abs_gap(*perm_ref)
    p = float((np.sum(np.abs(perms) >= abs(observed) - 1e-12) + 1) / (n_perm + 1))
    return {"n_events": len(events), "absgap_ref": float(g_ref), "absgap": float(g_mit), "d_absgap": observed,
            "ci_lo": float(lo), "ci_hi": float(hi), "p": p}


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    idx = [i for i, p in enumerate(pvals) if p is not None and not math.isnan(p)]
    m = len(idx)
    q = [float("nan")] * len(pvals)
    if m == 0:
        return q
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * m / rank)
        q[i] = val
        prev = val
    return q


def paired_flips(a: Dict[str, float], b: Dict[str, float], tau: float = 0.25) -> Dict[str, int]:
    ids = [i for i in a if i in b and a[i] is not None and b[i] is not None]
    up = sum(1 for i in ids if b[i] - a[i] >= tau)
    down = sum(1 for i in ids if a[i] - b[i] >= tau)
    return {"n_paired": len(ids), "n_up": up, "n_down": down, "n_same": len(ids) - up - down}
