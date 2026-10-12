#!/usr/bin/env python3
import json
import os
import sys
from typing import Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.judge.schema import JudgeResult


class LocalJudge:
    _TEMPLATE = """{system}

Return ONLY a valid JSON object (no prose, no markdown, no extra keys), in the following format:
{{
  "score": <number in [0,1]>,
  "explanation": <1-2 short, neutral English sentences>
}}

Sentence A: {sentence_a}
Sentence B: {sentence_b}
"""

    _STRICT_TEMPLATE = """{system}

Return ONLY this JSON object (no prose, no markdown, no comments, no extra keys):
{{
  "score": 0.0,
  "explanation": "..."
}}

The value of "score" MUST be a real number in [0,1].
The value of "explanation" MUST be a short, neutral English reason (1-2 sentences).
Start with '{{' and end with '}}'. Nothing else.

Sentence A: {sentence_a}
Sentence B: {sentence_b}
"""

    def __init__(
        self,
        model_path: str,
        system_prompt: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        dtype: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.temperature = max(0.0, float(temperature))
        self.max_new_tokens = int(max_new_tokens)
        self.dtype = dtype
        self.system_prompt = system_prompt

        self.llm = self._build_llm(model_path, self.temperature, self.max_new_tokens, self.dtype)
        self._parser = PydanticOutputParser(pydantic_object=JudgeResult)
        self._format_instructions = self._parser.get_format_instructions()

        self._prompt = PromptTemplate(
            template=self._TEMPLATE,
            input_variables=["system", "sentence_a", "sentence_b"],
            partial_variables={"format_instructions": self._format_instructions},
        )
        self._strict_prompt = PromptTemplate(
            template=self._STRICT_TEMPLATE,
            input_variables=["system", "sentence_a", "sentence_b"],
            partial_variables={"format_instructions": self._format_instructions},
        )

        print(self.system_prompt.format(language="TEST"))

        self._chain = self._prompt | self.llm | self._parser
        self._strict_chain = self._strict_prompt | self.llm | self._parser

    @staticmethod
    def _build_llm(model_path: str, temperature: float, max_new_tokens: int, dtype: str):
        dtype_map = {
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, None)

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        gen_pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=gen_pipe)

    def judge(self, language: str, sentence_a: str, sentence_b: str) -> Dict[str, Any]:
        """
        Returns a dict with keys: {"score": float, "explanation": str}.
        Performs one strict retry if parsing fails.
        """
        try:
            result: JudgeResult = self._chain.invoke(
                {"system": self.system_prompt.format(language=language), "sentence_a": sentence_a, "sentence_b": sentence_b}
            )
        except Exception:
            result: JudgeResult = self._strict_chain.invoke(
                {"system": self.system_prompt.format(language=language), "sentence_a": sentence_a, "sentence_b": sentence_b}
            )

        return {"score": float(result.score), "explanation": result.explanation}

    def judge_json(self, language: str, sentence_a: str, sentence_b: str) -> str:
        out = self.judge(language, sentence_a, sentence_b)
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))
