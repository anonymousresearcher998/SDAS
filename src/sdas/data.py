from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
SPLIT_PLACEHOLDER = "{split}"
DEFAULT_DATA_PATTERN = "data/ru_ua/neutral_data_{split}.json"
DEFAULT_LANGS: Tuple[str, str] = ("ru", "ua")


@dataclass(frozen=True)
class Item:
    split: str
    idx: int
    id: str
    en: str
    refs: Tuple[Tuple[str, str], ...]

    def ref(self, lang: str) -> str:
        for code, text in self.refs:
            if code == lang:
                return text
        raise ValueError(f"Unsupported lang={lang!r} for this item (has {', '.join(c for c, _ in self.refs)}).")

    @property
    def langs(self) -> Tuple[str, ...]:
        return tuple(code for code, _ in self.refs)

    def ref_fields(self) -> Dict[str, str]:
        return {f"ref_{code}": text for code, text in self.refs}

def item_id(split: str, idx: int) -> str:
    return f"{split}_{idx:05d}"


def split_of(item_id_: str) -> str:
    prefix = item_id_.split("_", 1)[0]
    if prefix not in SPLITS:
        raise ValueError(
            f"id {item_id_!r} carries no split prefix (expected one of {SPLITS})"
        )
    return prefix


def in_split(item_id_: str, split: Optional[str]) -> bool:
    if split is None or split == "all":
        return True
    return item_id_.split("_", 1)[0] == split


def id_sort_key(item_id_: str) -> Tuple[int, int]:
    split, _, idx = item_id_.partition("_")
    return SPLITS.index(split_of(item_id_)), int(idx)


def parse_sets(spec: Optional[str]) -> List[str]:
    if spec is None or not spec.strip() or spec.strip().lower() == "all":
        return list(SPLITS)
    wanted = {tok.strip() for tok in spec.split(",") if tok.strip()}
    unknown = sorted(wanted - set(SPLITS))
    if unknown:
        raise ValueError(f"Unknown set(s) {unknown}; choose from {', '.join(SPLITS)} (or 'all')")
    return [s for s in SPLITS if s in wanted]


def split_path(pattern: str, split: str) -> str:
    if SPLIT_PLACEHOLDER not in pattern:
        raise ValueError(
            f"--data must be a pattern with a {SPLIT_PLACEHOLDER} placeholder, e.g. {DEFAULT_DATA_PATTERN} (got {pattern!r})"
        )
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {SPLITS}")
    return pattern.replace(SPLIT_PLACEHOLDER, split)


# ---------------------------
# Loading
# ---------------------------
def load_split(pattern: str, split: str, limit: Optional[int] = None, langs: Sequence[str] = DEFAULT_LANGS) -> List[Item]:
    path = split_path(pattern, split)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    messages = payload.get("messages", [])
    items = messages[:limit] if limit is not None else messages
    out: List[Item] = []
    for idx, ex in enumerate(items):
        if not isinstance(ex, dict) or "neutral" not in ex or any(lang not in ex for lang in langs):
            continue
        out.append(
            Item(
                split=split,
                idx=idx,
                id=item_id(split, idx),
                en=ex["neutral"].strip(),
                refs=tuple((lang, ex[lang].strip()) for lang in langs),
            )
        )
    return out


def load_items(
    pattern: str, limit: Optional[int] = None, sets: Optional[Sequence[str]] = None, langs: Sequence[str] = DEFAULT_LANGS
) -> List[Item]:
    chosen = list(SPLITS) if sets is None else [s for s in SPLITS if s in set(sets)]
    out: List[Item] = []
    for split in chosen:
        out.extend(load_split(pattern, split, limit, langs))
    return out


def items_by_id(items: Sequence[Item]) -> Dict[str, Item]:
    return {it.id: it for it in items}


def split_ids(items: Sequence[Item]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {s: [] for s in SPLITS}
    for it in items:
        out[it.split].append(it.id)
    return out


def data_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_manifest(pattern: str) -> Dict[str, object]:
    files: Dict[str, Dict[str, object]] = {}
    for split in SPLITS:
        path = split_path(pattern, split)
        with open(path, "r", encoding="utf-8") as f:
            n = len(json.load(f).get("messages", []))
        files[split] = {"basename": os.path.basename(path), "sha256": data_sha256(path), "n": n}
    return {"pattern": os.path.basename(pattern), "files": files}


def manifest_diffs(recorded: Optional[Dict[str, object]], pattern: str) -> List[str]:
    if recorded is None:
        return ["experiment.json has no `data` manifest"]
    current = data_manifest(pattern)
    diffs: List[str] = []
    if recorded.get("pattern") != current["pattern"]:
        diffs.append(f"pattern: recorded {recorded.get('pattern')!r}, now {current['pattern']!r}")
    rec_files = recorded.get("files") or {}
    for split in SPLITS:
        a, b = rec_files.get(split), current["files"][split]
        if a is None:
            diffs.append(f"{split}: not recorded")
            continue
        for key in ("basename", "sha256", "n"):
            if a.get(key) != b[key]:
                diffs.append(f"{split}.{key}: recorded {a.get(key)!r}, now {b[key]!r}")
    return diffs
