from __future__ import annotations
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, ConfigDict, Field, model_validator

Difficulty = Literal["easy", "medium", "hard"]
FailureMode = Literal["none", "entity_drift", "incomplete_multi_hop", "wrong_final_answer", "looping", "reflection_overfit"]
MetricsMode = Literal["mock_estimate", "measured"]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class ContextChunk(StrictModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)

class QAExample(StrictModel):
    qid: str = Field(min_length=1)
    difficulty: Difficulty
    question: str = Field(min_length=1)
    gold_answer: str = Field(min_length=1)
    context: list[ContextChunk] = Field(min_length=1)

class ActorOutput(StrictModel):
    answer: str = Field(min_length=1)

class JudgeResult(StrictModel):
    score: int = Field(ge=0, le=1, strict=True)
    reason: str = Field(min_length=1)
    missing_evidence: list[str] = Field(default_factory=list)
    spurious_claims: list[str] = Field(default_factory=list)
    failure_mode: FailureMode = "wrong_final_answer"

    @model_validator(mode="after")
    def keep_score_and_failure_consistent(self) -> JudgeResult:
        if self.score == 1:
            self.failure_mode = "none"
        elif self.failure_mode == "none":
            self.failure_mode = "wrong_final_answer"
        return self

class ReflectionEntry(StrictModel):
    attempt_id: int = Field(ge=1)
    failure_reason: str = Field(min_length=1)
    lesson: str = Field(min_length=1)
    next_strategy: str = Field(min_length=1)

class AttemptTrace(StrictModel):
    attempt_id: int = Field(ge=1)
    answer: str = Field(min_length=1)
    score: int = Field(ge=0, le=1, strict=True)
    reason: str = Field(min_length=1)
    reflection: Optional[ReflectionEntry] = None
    token_estimate: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    token_is_estimate: bool = True
    metrics_mode: MetricsMode = "mock_estimate"

class RunRecord(StrictModel):
    qid: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_answer: str = Field(min_length=1)
    agent_type: Literal["react", "reflexion"]
    predicted_answer: str = Field(min_length=1)
    is_correct: bool
    attempts: int = Field(ge=1)
    token_estimate: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    token_is_estimate: bool = True
    metrics_mode: MetricsMode = "mock_estimate"
    failure_mode: FailureMode
    reflections: list[ReflectionEntry] = Field(default_factory=list)
    traces: list[AttemptTrace] = Field(default_factory=list)

class ReportPayload(StrictModel):
    meta: dict
    summary: dict
    failure_modes: dict
    examples: list[dict]
    extensions: list[str]
    discussion: str

class ReflexionState(TypedDict):
    question: str
    context: list[str]
    trajectory: list[str]
    reflection_memory: list[str]
    attempt_count: int
    success: bool
    final_answer: str
