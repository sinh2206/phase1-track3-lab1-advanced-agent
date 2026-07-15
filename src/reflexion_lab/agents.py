from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import MockRuntime
from .runtime import AgentRuntime
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: AgentRuntime | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def run(self, example: QAExample) -> RunRecord:
        runtime = self.runtime or MockRuntime()
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_failure_mode = "wrong_final_answer"
        metrics_mode = "mock_estimate" if runtime.mode == "mock" else "measured"
        for attempt_id in range(1, self.max_attempts + 1):
            actor_call = runtime.actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge_call = runtime.evaluator(example, actor_call.value)
            answer, judge = actor_call.value, judge_call.value
            token_count = actor_call.token_count + judge_call.token_count
            latency_ms = actor_call.latency_ms + judge_call.latency_ms
            token_is_estimate = actor_call.token_is_estimate or judge_call.token_is_estimate
            reflection = None
            final_answer = answer
            final_score = judge.score
            final_failure_mode = judge.failure_mode
            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection_call = runtime.reflector(example, attempt_id, answer, judge)
                reflection = reflection_call.value
                reflections.append(reflection)
                reflection_memory.append(f"Lesson: {reflection.lesson}\nNext strategy: {reflection.next_strategy}")
                token_count += reflection_call.token_count
                latency_ms += reflection_call.latency_ms
                token_is_estimate = token_is_estimate or reflection_call.token_is_estimate
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, reflection=reflection, token_estimate=token_count, latency_ms=latency_ms, token_is_estimate=token_is_estimate, metrics_mode=metrics_mode)
            traces.append(trace)
            if judge.score == 1:
                break
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else final_failure_mode
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, token_is_estimate=any(t.token_is_estimate for t in traces), metrics_mode=metrics_mode, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self, runtime: AgentRuntime | None = None) -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime=runtime)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, runtime: AgentRuntime | None = None) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, runtime=runtime)
