from __future__ import annotations
import json
import os
from time import perf_counter
from typing import TypeVar
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .runtime import CallResult
from .schemas import ActorOutput, JudgeResult, QAExample, ReflectionEntry

M = TypeVar("M", bound=BaseModel)

class LLMRuntime:
    mode = "llm"

    def __init__(self, model: str | None = None, base_url: str | None = None, temperature: float = 0.0, max_retries: int = 2) -> None:
        load_dotenv()
        self.model = model or os.getenv("LLM_MODEL", "")
        if not self.model:
            raise ValueError("Missing model. Set LLM_MODEL or pass --model.")
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key and resolved_base_url and ("localhost" in resolved_base_url or "127.0.0.1" in resolved_base_url):
            api_key = "local-model"
        if not api_key:
            raise ValueError("Missing API key. Set LLM_API_KEY or OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key, base_url=resolved_base_url, timeout=float(os.getenv("LLM_TIMEOUT", "60")))
        self.temperature = temperature
        self.max_retries = max_retries
        self.json_mode = os.getenv("LLM_JSON_MODE", "true").lower() not in {"0", "false", "no"}

    @staticmethod
    def _clean_json(text: str) -> str:
        value = text.strip()
        if value.startswith("```"):
            value = value.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return value

    @staticmethod
    def _estimated_tokens(*texts: str) -> int:
        return max(1, sum(len(text) for text in texts) // 4)

    def _chat(self, system: str, user: str, structured: bool = True) -> CallResult[str]:
        kwargs = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": self.temperature}
        if structured and self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        started = perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        latency_ms = round((perf_counter() - started) * 1000)
        content = response.choices[0].message.content or ""
        usage = response.usage
        has_usage = bool(usage and usage.total_tokens is not None)
        tokens = int(usage.total_tokens) if has_usage else self._estimated_tokens(system, user, content)
        return CallResult(content, tokens, latency_ms, not has_usage)

    def _structured(self, system: str, user: str, schema: type[M]) -> CallResult[M]:
        total_tokens = 0
        total_latency = 0
        token_is_estimate = False
        last_error: ValidationError | ValueError | None = None
        request = user
        for retry in range(self.max_retries + 1):
            call = self._chat(system, request)
            total_tokens += call.token_count
            total_latency += call.latency_ms
            token_is_estimate = token_is_estimate or call.token_is_estimate
            try:
                value = schema.model_validate_json(self._clean_json(call.value))
                return CallResult(value, total_tokens, total_latency, token_is_estimate)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                request = f"{user}\n\nYour previous response was invalid. Return only JSON matching this schema:\n{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        raise RuntimeError(f"LLM did not return valid {schema.__name__} after {self.max_retries + 1} calls") from last_error

    @staticmethod
    def _context(example: QAExample) -> str:
        return json.dumps([chunk.model_dump() for chunk in example.context], ensure_ascii=False)

    def actor_answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> CallResult[str]:
        user = f"Question: {example.question}\nContext: {self._context(example)}\nAttempt: {attempt_id}\nReflection memory: {json.dumps(reflection_memory, ensure_ascii=False)}"
        call = self._structured(ACTOR_SYSTEM, user, ActorOutput)
        return CallResult(call.value.answer, call.token_count, call.latency_ms, call.token_is_estimate)

    def evaluator(self, example: QAExample, answer: str) -> CallResult[JudgeResult]:
        user = f"Question: {example.question}\nGold answer: {example.gold_answer}\nPredicted answer: {answer}"
        return self._structured(EVALUATOR_SYSTEM, user, JudgeResult)

    def reflector(self, example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> CallResult[ReflectionEntry]:
        user = f"Question: {example.question}\nContext: {self._context(example)}\nAttempt: {attempt_id}\nWrong answer: {answer}\nEvaluator feedback: {judge.model_dump_json()}"
        call = self._structured(REFLECTOR_SYSTEM, user, ReflectionEntry)
        entry = call.value.model_copy(update={"attempt_id": attempt_id})
        return CallResult(entry, call.token_count, call.latency_ms, call.token_is_estimate)
