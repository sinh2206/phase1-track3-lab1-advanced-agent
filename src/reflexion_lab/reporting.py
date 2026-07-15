from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

FAILURE_EXPLANATIONS = {
    "incomplete_multi_hop": "dừng ở thực thể trung gian và chưa hoàn tất chuỗi bằng chứng",
    "entity_drift": "chọn thực thể cuối không được đoạn bằng chứng liên quan hỗ trợ",
    "wrong_final_answer": "hoàn tất lượt nhưng kết luận cuối không tương đương đáp án chuẩn",
    "looping": "lặp lại đáp án hoặc chiến thuật sai qua nhiều lượt",
    "reflection_overfit": "bài học quá đặc thù làm chiến thuật sau kém tổng quát",
}

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        avg_tokens = round(mean(r.token_estimate for r in rows), 2)
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_tokens": avg_tokens, "avg_token_estimate": avg_tokens, "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2), "token_estimate_rate": round(mean(1.0 if r.token_is_estimate else 0.0 for r in rows), 4)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        if record.failure_mode != "none":
            grouped[record.failure_mode][record.agent_type] += 1
    return {mode: {"react": counts.get("react", 0), "reflexion": counts.get("reflexion", 0), "total": sum(counts.values())} for mode, counts in sorted(grouped.items())}

def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    summary = summarize(records)
    failures = failure_breakdown(records)
    examples = [{"qid": r.qid, "agent_type": r.agent_type, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": r.failure_mode, "reflection_count": len(r.reflections), "token_estimate": r.token_estimate, "latency_ms": r.latency_ms} for r in records]
    delta = summary.get("delta_reflexion_minus_react", {})
    failure_analysis = "; ".join(f"{mode} ({counts['total']}): {FAILURE_EXPLANATIONS.get(mode, 'lỗi chưa phân loại chi tiết')}" for mode, counts in failures.items()) or "không có lỗi cuối cùng để phân tích"
    discussion = f"Benchmark so sánh baseline một lượt với Reflexion có bộ nhớ phản chiếu trên cùng dữ liệu. Chênh lệch EM của Reflexion so với ReAct là {delta.get('em_abs', 0)}, trong khi số lượt trung bình thay đổi {delta.get('attempts_abs', 0)}, token thay đổi {delta.get('tokens_abs', 0)} và độ trễ thay đổi {delta.get('latency_abs', 0)} ms. Phân tích lỗi: {failure_analysis}. Reflexion có lợi khi phản hồi giúp hoàn tất hop còn thiếu hoặc sửa thực thể bị trôi, nhưng phải đánh đổi thêm lời gọi Actor, Evaluator và Reflector. Kết quả còn phụ thuộc chất lượng evaluator, độ đầy đủ của context và việc token do provider trả trực tiếp hay chỉ được ước lượng."
    return ReportPayload(meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})}, summary=summary, failure_modes=failures, examples=examples, extensions=["structured_evaluator", "reflection_memory", "benchmark_report_json", "mock_mode_for_autograding"], discussion=discussion)

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg tokens | {react.get('avg_tokens', 0)} | {reflexion.get('avg_tokens', 0)} | {delta.get('tokens_abs', 0)} |
| Token estimate rate | {react.get('token_estimate_rate', 0)} | {reflexion.get('token_estimate_rate', 0)} | n/a |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
