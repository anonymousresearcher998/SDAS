from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.mitigation.psychological_interventions.self_distance import (
    SELF_DISTANCE_PROMPT,
    SELF_DISTANCE_PROMPT_V2,
    SELF_DISTANCE_PROMPT_V3,
)
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB

INTERVENTION_PROMPTS: Dict[str, str] = {
    "self_distance_v1": SELF_DISTANCE_PROMPT,
    "self_distance_v2": SELF_DISTANCE_PROMPT_V2,
    "self_distance_v3": SELF_DISTANCE_PROMPT_V3,
}

@dataclass(frozen=True)
class LangCfg:
    code: str
    language: str
    country: str
    other: str


LANG_CFG: Dict[str, LangCfg] = {
    "ru": LangCfg(code="ru", language="Russian", country="Russia", other="ua"),
    "ua": LangCfg(code="ua", language="Ukrainian", country="Ukraine", other="ru"),
    "ps": LangCfg(code="ps", language="Arabic", country="Palestine", other="il"),
    "il": LangCfg(code="il", language="Hebrew", country="Israel", other="ps"),
}
LANG_CODES = tuple(LANG_CFG)


def lang_cfg(code: str) -> LangCfg:
    try:
        return LANG_CFG[code.lower().strip()]
    except KeyError:
        raise ValueError(f"Unsupported lang={code!r}. Use one of {', '.join(LANG_CODES)}.") from None


@dataclass(frozen=True)
class PairCfg:
    name: str
    a: str
    b: str
    data_pattern: str
    out_root: str
    results_root: str

    @property
    def langs(self) -> Tuple[str, str]:
        return (self.a, self.b)

    @property
    def label(self) -> str:
        return f"{self.a.upper()}/{self.b.upper()}"


DEFAULT_PAIR = "ru_ua"
PAIRS: Dict[str, PairCfg] = {
    "ru_ua": PairCfg("ru_ua", "ru", "ua", "data/ru_ua/neutral_data_{split}.json", "results/sdas", "results"),
    "il_ps": PairCfg("il_ps", "ps", "il", "data/il_ps/neutral_data_{split}.json", "results_il_ps/sdas", "results_il_ps"),
}


def pair_cfg(name: Optional[str]) -> PairCfg:
    key = DEFAULT_PAIR if name is None else name.strip().lower()
    try:
        return PAIRS[key]
    except KeyError:
        raise ValueError(f"Unknown pair {name!r}. Known: {', '.join(PAIRS)}.") from None


def pair_of_lang(code: str) -> PairCfg:
    code = lang_cfg(code).code
    return next(p for p in PAIRS.values() if code in p.langs)


def experiment_pair(experiment: Dict) -> PairCfg:
    return pair_cfg(experiment.get("pair"))


def assert_lang_in_pair(lang: str, pair: PairCfg) -> None:
    if lang_cfg(lang).code not in pair.langs:
        raise SystemExit(
            f"--lang {lang!r} does not belong to this experiment's pair {pair.name} ({', '.join(pair.langs)})"
        )


def resolve_langs(spec: Optional[str], pair: PairCfg) -> List[str]:
    if spec is None or not spec.strip():
        return list(pair.langs)
    langs = [tok.strip().lower() for tok in spec.split(",") if tok.strip()]
    for lang in langs:
        assert_lang_in_pair(lang, pair)
    return langs


MEAN_KEY: Dict[str, str] = {
    "ru": "mean_score_ru_MT_vs_ru_POV",
    "ua": "mean_score_UA_MT_vs_UA_POV",
    "ps": "mean_score_ps_MT_vs_ps_POV",
    "il": "mean_score_IL_MT_vs_IL_POV",
}
CONVENTION: Dict[str, str] = {
    "ru": "Lower score = closer to ru POV = more biased toward ru.",
    "ua": "Lower score = closer to UA POV = more biased toward UA.",
    "ps": "Lower score = closer to ps POV = more biased toward ps.",
    "il": "Lower score = closer to IL POV = more biased toward IL.",
}


def compose_system_prompt(template: str, intervention_text: str, *, language: str, country: str) -> str:
    if intervention_text:
        template = f"{intervention_text}\n\n{template}"
    return template.format(language=language, country=country)


def prompt_group(prompt_name: str) -> str:
    return prompt_name.rsplit("_", 1)[0]


def all_prompt_groups() -> List[str]:
    seen: List[str] = []
    for name in SYSTEM_PROMPTS_LIB:
        g = prompt_group(name)
        if g not in seen:
            seen.append(g)
    return seen


def resolve_prompts(spec: Optional[str]) -> List[str]:
    if spec is None or not spec.strip() or spec.strip().lower() == "all":
        return list(SYSTEM_PROMPTS_LIB)

    groups = all_prompt_groups()
    wanted = set()
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok in SYSTEM_PROMPTS_LIB:
            wanted.add(tok)
        elif tok in groups:
            wanted.update(n for n in SYSTEM_PROMPTS_LIB if prompt_group(n) == tok)
        else:
            raise ValueError(
                f"Unknown prompt or group {tok!r}. Prompts: {', '.join(SYSTEM_PROMPTS_LIB)}. "
                f"Groups: {', '.join(groups)}."
            )
    return [n for n in SYSTEM_PROMPTS_LIB if n in wanted]


def intervention_text(name: str) -> str:
    try:
        return INTERVENTION_PROMPTS[name]
    except KeyError:
        raise ValueError(
            f"Unknown intervention {name!r}. Known: {', '.join(sorted(INTERVENTION_PROMPTS))}."
        ) from None
