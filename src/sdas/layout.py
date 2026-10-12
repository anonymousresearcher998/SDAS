from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch

RUN_TS_FORMAT = "%Y-%m-%d_%H-%M-%S"


def now_ts() -> str:
    return datetime.now().strftime(RUN_TS_FORMAT)


def format_alpha(alpha: float) -> str:
    """2.0 -> '2', 0.5 -> '0.5', -2.0 -> '-2'."""
    return f"{float(alpha):g}"


def prompt_condition(intervention: str) -> str:
    return f"prompt/{intervention}"


def sdas_condition(
    direction: str, layer: int, alpha: float, seed: Optional[int] = None, rep: Optional[int] = None
) -> str:
    if seed is not None and rep is not None:
        raise ValueError("a steer cell is either re-seeded (_seed<s>) or a replicate (_r<rep>), not both")
    slug = f"sdas/{direction}/L{int(layer):02d}_a{format_alpha(alpha)}"
    if seed is not None:
        slug += f"_seed{int(seed)}"
    if rep is not None:
        slug += f"_r{int(rep)}"
    return slug


_SDAS_RE = re.compile(
    r"^sdas/(?P<direction>[^/]+)/L(?P<layer>\d+)_a(?P<alpha>-?\d+(?:\.\d+)?(?:e-?\d+)?)"
    r"(?:_seed(?P<seed>\d+)|_r(?P<rep>\d+))?$"
)


def parse_condition(slug: str) -> Dict[str, Any]:
    """`{"kind": "regular"}` | `{"kind": "prompt", "intervention": ...}` |
    `{"kind": "sdas", "direction", "layer", "alpha", "seed", "rep"}`; raises on anything else."""
    slug = slug.strip("/")
    if slug == "regular":
        return {"kind": "regular"}
    if slug.startswith("prompt/") and slug.count("/") == 1:
        return {"kind": "prompt", "intervention": slug.split("/", 1)[1]}
    m = _SDAS_RE.match(slug)
    if m:
        return {
            "kind": "sdas",
            "direction": m.group("direction"),
            "layer": int(m.group("layer")),
            "alpha": float(m.group("alpha")),
            "seed": int(m.group("seed")) if m.group("seed") is not None else None,
            "rep": int(m.group("rep")) if m.group("rep") is not None else None,
        }
    raise ValueError(
        f"Unrecognized condition slug {slug!r}."
    )


def replicate_base_seed(base_seed: int, rep: int) -> int:
    if rep < 1:
        raise ValueError("replicate numbers start at 1 (the original run is the un-suffixed cell)")
    return zlib.crc32(f"{int(base_seed)}|rep{int(rep)}".encode("utf-8"))


class ExperimentLayout:
    def __init__(self, out_root: str, experiment: str):
        self.experiment = experiment
        self.root = Path(out_root) / experiment

    @classmethod
    def from_dir(cls, experiment_dir: str) -> "ExperimentLayout":
        root = Path(experiment_dir)
        return cls(str(root.parent), root.name)

    @property
    def experiment_json(self) -> Path:
        return self.root / "experiment.json"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def generations_dir(self) -> Path:
        return self.root / "generations"

    @property
    def directions_dir(self) -> Path:
        return self.root / "directions"

    @property
    def judged_dir(self) -> Path:
        return self.root / "judged"

    # leaves
    def generation_dir(self, condition: str, lang: str, prompt_name: str) -> Path:
        return self.generations_dir / condition.strip("/") / lang / prompt_name

    def direction_dir(self, name: str) -> Path:
        return self.directions_dir / name

    def screens_dir(self, direction: str, ts: str) -> Path:
        return self.root / "screens" / direction / ts

    def judged_prompt_dir(
        self, condition: str, judge_bucket: str, judge_prompt: str, translator: str, prompt_name: str
    ) -> Path:
        return self.judged_dir / condition.strip("/") / f"{judge_bucket}_judge" / judge_prompt / translator / prompt_name

    def judged_run_dir(
        self, condition: str, judge_bucket: str, judge_prompt: str, translator: str, prompt_name: str, lang: str, ts: str
    ) -> Path:
        return self.judged_prompt_dir(condition, judge_bucket, judge_prompt, translator, prompt_name) / f"{lang}_run_{ts}"

    def summary_dir(self, ts: str) -> Path:
        return self.root / "summary" / ts

    def list_conditions(self) -> List[str]:
        """Every condition slug that has at least one `gen.jsonl` leaf."""
        base = self.generations_dir
        if not base.is_dir():
            return []
        found: Set[str] = set()
        for gen in base.rglob("gen.jsonl"):
            rel = gen.parent.relative_to(base).parts
            if len(rel) >= 3:
                found.add("/".join(rel[:-2]))
        return sorted(found)

    def list_leaves(self, condition: str) -> List[Tuple[str, str, Path]]:
        """(lang, prompt_name, leaf_dir) for every leaf of a condition, sorted."""
        base = self.generations_dir / condition.strip("/")
        out: List[Tuple[str, str, Path]] = []
        if not base.is_dir():
            return out
        for lang_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for prompt_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
                if (prompt_dir / "gen.jsonl").is_file():
                    out.append((lang_dir.name, prompt_dir.name, prompt_dir))
        return out


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp{os.getpid()}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, canonical_json(obj) + "\n")


def atomic_torch_save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def unique_run_dir(parent: Path, lang: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    base = parent / f"{lang}_run_{now_ts()}"
    candidate = base
    for k in range(1, 1000):
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = parent / f"{base.name}_{k}"
    raise RuntimeError(f"could not create a unique run dir under {parent}")


def existing_ids(gen_dir: Path) -> Set[str]:
    return {row["id"] for row in read_jsonl(Path(gen_dir) / "gen.jsonl") if "id" in row}


def leaf_complete(gen_dir: Path, expected_ids: Iterable[str]) -> bool:
    meta_path = Path(gen_dir) / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = read_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return False
    return set(meta.get("ids", [])) >= set(expected_ids)


def _diff(a: Any, b: Any, prefix: str = "") -> List[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for key in sorted(set(a) | set(b)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in a:
                out.append(f"{path}: missing on disk, requested {b[key]!r}")
            elif key not in b:
                out.append(f"{path}: on disk {a[key]!r}, not requested")
            else:
                out.extend(_diff(a[key], b[key], path))
        return out
    return [] if a == b else [f"{prefix}: on disk {a!r}, requested {b!r}"]


def create_or_assert_experiment(layout: ExperimentLayout, invariants: Dict[str, Any]) -> Dict[str, Any]:
    path = layout.experiment_json
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as f:
            f.write(canonical_json(invariants) + "\n")
        return dict(invariants)
    except FileExistsError:
        pass
    existing = read_json(path)
    diffs = _diff(existing, invariants)
    if diffs:
        raise ValueError(
            f"{path} was created by an earlier run with different invariants"
        )
    return existing


def load_experiment(layout: ExperimentLayout) -> Dict[str, Any]:
    path = layout.experiment_json
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found."
        )
    return read_json(path)


def assert_generation_env(experiment: Dict[str, Any]) -> None:
    import transformers

    installed_version = transformers.__version__
    installed = int(installed_version.split(".")[0])
    recorded = experiment.get("transformers_major")
    where = f"this interpreter ({sys.executable}) has transformers {installed_version}"
    if recorded is None:
        if installed >= 5:
            raise SystemExit(
                f"experiment {experiment.get('translator_model_name')!r} was created under transformers 4.x"
            )
        return
    if int(recorded) != installed:
        raise SystemExit(
            f"experiment {experiment.get('translator_model_name')!r} was created under transformers {recorded}.x"
        )


def git_commit_short(cwd: Optional[str] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def write_run_record(layout: ExperimentLayout, stage: str, payload: Dict[str, Any]) -> Path:
    job = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    ts = now_ts()
    record: Dict[str, Any] = {
        "stage": stage,
        "started": ts,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "commit": git_commit_short(),
    }
    try:
        import transformers
        
        record["transformers"] = transformers.__version__
    except Exception:
        record["transformers"] = None
    record.update(payload)
    path = layout.runs_dir / f"{stage}_{job}_{task}_{ts}.json"
    atomic_write_json(path, record)
    return path
