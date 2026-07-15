from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar
from .schemas import JudgeResult, QAExample, ReflectionEntry

T = TypeVar("T")

@dataclass(frozen=True, slots=True)
class CallResult(Generic[T]):
    value: T
    token_count: int
    latency_ms: int
    token_is_estimate: bool = False

class AgentRuntime(Protocol):
    mode: str

    def actor_answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> CallResult[str]: ...
    def evaluator(self, example: QAExample, answer: str) -> CallResult[JudgeResult]: ...
    def reflector(self, example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> CallResult[ReflectionEntry]: ...
