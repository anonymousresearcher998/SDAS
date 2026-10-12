from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch

from src.sdas.directions import direction_vector, load_direction
from src.sdas.layout import atomic_write_text, now_ts
from src.sdas.prompts import PAIRS

COLUMNS = ["model", "selected_for", "layer", "cosine", "norm_ratio", "norm_ru_ua", "norm_il_ps"]


def parse_model(spec: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    name, _, layers = spec.partition(":")
    if not name:
        raise SystemExit(f"--model {spec!r}: missing the model name")
    if not layers:
        return name, None
    parts = layers.split(",")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise SystemExit(f"--model {spec!r}: layers must be NAME:L_RU_UA,L_IL_PS")
    return name, (int(parts[0]), int(parts[1]))


def compare(d_ru: torch.Tensor, d_il: torch.Tensor) -> Dict[str, float]:
    a, b = d_ru.double(), d_il.double()
    n_ru, n_il = float(a.norm()), float(b.norm())
    return {
        "cosine": float(a @ b) / max(n_ru * n_il, 1e-12),
        "norm_ratio": n_ru / max(n_il, 1e-12),
        "norm_ru_ua": n_ru,
        "norm_il_ps": n_il,
    }


def model_rows(name: str, layers: Optional[Tuple[int, int]], ru_dir: Path, il_dir: Path) -> List[Dict[str, object]]:
    d_ru, meta_ru = load_direction(ru_dir)
    d_il, meta_il = load_direction(il_dir)
    lay_ru, lay_il = [int(l) for l in meta_ru["layers"]], [int(l) for l in meta_il["layers"]]
    if d_ru.shape[-1] != d_il.shape[-1]:
        raise SystemExit(f"{name}: hidden sizes differ ({d_ru.shape[-1]} vs {d_il.shape[-1]}) - not the same model")
    if layers is None:
        wanted = [("all", l) for l in lay_ru if l in lay_il]
    else:
        wanted = [("RU-UA", layers[0]), ("IL-PS", layers[1])]
    rows = []
    for selected_for, layer in wanted:
        stats = compare(direction_vector(d_ru, lay_ru, layer), direction_vector(d_il, lay_il, layer))
        rows.append({"model": name, "selected_for": selected_for, "layer": layer, **stats})
    return rows


def fmt(row: Dict[str, object], col: str) -> str:
    v = row[col]
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cosine and norm ratio between the RU-UA and IL-PS SDAS directions (paper Table 2).")
    ap.add_argument("--model", action="append", required=True, metavar="NAME[:L_RU_UA,L_IL_PS]",
                    help="Model and its selected RU-UA / IL-PS layers (repeatable); without layers, every shared layer")
    ap.add_argument("--direction", default="self_distance_v1", help="Direction name in both experiments")
    ap.add_argument("--ru_ua_root", default=PAIRS["ru_ua"].out_root, help=f"RU-UA steering experiments [{PAIRS['ru_ua'].out_root}]")
    ap.add_argument("--il_ps_root", default=PAIRS["il_ps"].out_root, help=f"IL-PS steering experiments [{PAIRS['il_ps'].out_root}]")
    ap.add_argument("--out", default=None, help="Output directory [results/cross_conflict/alignment/<ts>]")
    args = ap.parse_args()

    rows: List[Dict[str, object]] = []
    for spec in args.model:
        name, layers = parse_model(spec)
        ru_dir = Path(args.ru_ua_root) / name / "directions" / args.direction
        il_dir = Path(args.il_ps_root) / name / "directions" / args.direction
        rows.extend(model_rows(name, layers, ru_dir, il_dir))

    out_dir = Path(args.out) if args.out else Path(PAIRS["ru_ua"].results_root) / "cross_conflict" / "alignment" / now_ts()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in COLUMNS})
    atomic_write_text(out_dir / "alignment.csv", buf.getvalue())
    md = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    md += ["| " + " | ".join(fmt(r, c) for c in COLUMNS) + " |" for r in rows]
    atomic_write_text(out_dir / "alignment.md", "\n".join(md) + "\n")

    print(f"direction {args.direction}: RU-UA from {args.ru_ua_root}, IL-PS from {args.il_ps_root}")
    print("norm ratio = ||d_RU-UA|| / ||d_IL-PS|| (before scaling by alpha); layers are zero-based\n")
    print("\n".join(md))
    print(f"\nwrote {out_dir / 'alignment.csv'}, alignment.md")


if __name__ == "__main__":
    main()
