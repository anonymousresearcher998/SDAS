from __future__ import annotations

import math
from typing import Any, Dict

from pydantic.v1 import BaseModel, Field, validator


class JudgeResult(BaseModel):
    score: float = Field(
        ...,
        description="Real number in [0,1]. 0 = identical; 0.5 = somewhat different; 1 = unrelated/contradictory."
    )
    explanation: str = Field(
        ...,
        description="Concise, neutral English reason for the score (1-2 short sentences)."
    )

    @validator("score")
    def clamp_score(cls, v):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Score must be finite.")
        return max(0.0, min(1.0, float(v)))

    @validator("explanation")
    def strip_explanation(cls, v):
        return v.strip()

JUDGE_JSON_SCHEMA: Dict[str, Any] = {
    "name": "judge_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": "Real number in [0,1]. 0 = identical; 1 = unrelated/contradictory.",
            },
            "explanation": {
                "type": "string",
                "description": "1-2 short, neutral English sentences explaining the score.",
            },
        },
        "required": ["score", "explanation"],
        "additionalProperties": False,
    },
}
