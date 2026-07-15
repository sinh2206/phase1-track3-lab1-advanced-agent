import pytest
from pydantic import ValidationError
from src.reflexion_lab.schemas import JudgeResult, ReflectionEntry

def test_judge_result_normalizes_failure_mode():
    correct = JudgeResult(score=1, reason="Equivalent answer.", failure_mode="entity_drift")
    wrong = JudgeResult(score=0, reason="Wrong answer.", failure_mode="none")
    assert correct.failure_mode == "none"
    assert wrong.failure_mode == "wrong_final_answer"

def test_judge_result_rejects_invalid_score():
    with pytest.raises(ValidationError):
        JudgeResult(score=2, reason="Invalid")
    with pytest.raises(ValidationError):
        JudgeResult(score=True, reason="Boolean is not a numeric score")

def test_reflection_requires_positive_attempt():
    with pytest.raises(ValidationError):
        ReflectionEntry(attempt_id=0, failure_reason="x", lesson="x", next_strategy="x")

def test_structured_models_reject_extra_fields():
    with pytest.raises(ValidationError):
        JudgeResult(score=1, reason="Correct", unexpected="value")
