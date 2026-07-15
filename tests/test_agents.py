import pytest
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.mock_runtime import MockRuntime
from src.reflexion_lab.runtime import CallResult
from src.reflexion_lab.schemas import JudgeResult, QAExample, ReflectionEntry

class AlwaysWrongRuntime:
    mode = "mock"

    def actor_answer(self, example, attempt_id, agent_type, reflection_memory):
        return CallResult("wrong", 1, 1, True)

    def evaluator(self, example, answer):
        return CallResult(JudgeResult(score=0, reason="Still wrong."), 1, 1, True)

    def reflector(self, example, attempt_id, answer, judge):
        entry = ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="Change approach.", next_strategy="Check both hops.")
        return CallResult(entry, 1, 1, True)

def make_example(qid: str = "hp2") -> QAExample:
    return QAExample(qid=qid, difficulty="medium", question="What river flows through the city where Ada Lovelace was born?", gold_answer="River Thames", context=[{"title": "Ada Lovelace", "text": "Ada Lovelace was born in London."}, {"title": "London", "text": "London is crossed by the River Thames."}])

def test_react_runs_once_without_reflection():
    record = ReActAgent(MockRuntime()).run(make_example())
    assert not record.is_correct
    assert record.attempts == 1
    assert record.reflections == []
    assert record.traces[0].reflection is None

def test_reflexion_uses_memory_and_recovers():
    record = ReflexionAgent(max_attempts=3, runtime=MockRuntime()).run(make_example())
    assert record.is_correct
    assert record.attempts == 2
    assert len(record.reflections) == 1
    assert record.traces[0].reflection == record.reflections[0]
    assert record.traces[1].reflection is None
    assert record.token_estimate == sum(trace.token_estimate for trace in record.traces)

def test_reflexion_stops_immediately_when_correct():
    record = ReflexionAgent(runtime=MockRuntime()).run(make_example("new-qid"))
    assert record.is_correct
    assert record.attempts == 1
    assert record.reflections == []

def test_reflexion_rejects_invalid_attempt_limit():
    with pytest.raises(ValueError, match="at least 1"):
        ReflexionAgent(max_attempts=0)

def test_reflexion_does_not_reflect_after_last_attempt():
    record = ReflexionAgent(max_attempts=2, runtime=AlwaysWrongRuntime()).run(make_example())
    assert not record.is_correct
    assert record.attempts == 2
    assert len(record.reflections) == 1
    assert record.traces[-1].reflection is None
