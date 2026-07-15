import json
import pytest
from src.reflexion_lab.utils import normalize_answer
from src.reflexion_lab.utils import load_dataset

def test_normalize_answer():
    assert normalize_answer("Oxford University!") == "oxford university"

def test_normalize_answer_keeps_unicode_and_removes_articles():
    assert normalize_answer("  The Đại học Quốc gia, Hà Nội! ") == "đại học quốc gia hà nội"

def test_load_dataset_rejects_duplicate_qid(tmp_path):
    item = {"qid": "same", "difficulty": "easy", "question": "Q?", "gold_answer": "A", "context": [{"title": "T", "text": "A."}]}
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps([item, item]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate qid"):
        load_dataset(path)

def test_main_dataset_has_100_unique_examples():
    examples = load_dataset("data/multihop_100.json")
    assert len(examples) == 100
    assert len({example.qid for example in examples}) == 100
