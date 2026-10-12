from src.mitigation.control import NEUTRAL_CONTROL_PROMPT
from src.mitigation.psychological_interventions.self_distance import (
    SELF_DISTANCE_PROMPT,
    SELF_DISTANCE_PROMPT_V2,
    SELF_DISTANCE_PROMPT_V3,
)

MITIGATION_PROMPTS_LIB = {
    "baseline": "",
    "self_distance": SELF_DISTANCE_PROMPT,
    "self_distance_v2": SELF_DISTANCE_PROMPT_V2,
    "self_distance_v3": SELF_DISTANCE_PROMPT_V3,
    "control": NEUTRAL_CONTROL_PROMPT,
}
