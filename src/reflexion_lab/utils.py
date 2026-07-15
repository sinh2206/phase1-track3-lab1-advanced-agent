from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable
from .schemas import QAExample, RunRecord

def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = "".join(char if char.isalnum() or char.isspace() else " " for char in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_dataset(path: str | Path) -> list[QAExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Dataset must be a non-empty JSON array.")
    examples = [QAExample.model_validate(item) for item in raw]
    qids = [example.qid for example in examples]
    if len(qids) != len(set(qids)):
        raise ValueError("Dataset contains duplicate qid values.")
    return examples

def save_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
