# Lab 16 — Reflexion Agent

## Tổng quan

Dự án triển khai một pipeline hỏi đáp multi-hop gồm Actor, Evaluator và Reflector.
Hai agent chạy trên cùng dữ liệu để so sánh:

- `ReActAgent`: baseline một lần trả lời.
- `ReflexionAgent`: tối đa N lần trả lời; sau mỗi lần sai, Reflector tạo bài học và
  chiến thuật để Actor dùng ở lần kế tiếp.

Repo hỗ trợ hai backend:

- `mock`: xác định, không dùng mạng/API, phù hợp test và autograde.
- `llm`: dùng API tương thích OpenAI, hỗ trợ OpenAI hoặc endpoint local như Ollama.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS dùng `source .venv/bin/activate` để kích hoạt môi trường.

## Chạy mock

```powershell
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/sample_run --mode mock
python autograde.py --report-path outputs/sample_run/report.json
```

Mock cố ý làm sai `hp2`, `hp4`, `hp6`, `hp8` ở baseline. Reflexion chỉ sửa được
đáp án khi reflection memory đã được tạo, nhờ đó kiểm tra được flow thật của loop.

## Chạy LLM thật

Sao chép `.env.example` thành `.env`, sau đó cấu hình:

```dotenv
LLM_MODEL=your-model-name
LLM_API_KEY=replace-me
# LLM_BASE_URL=https://api.openai.com/v1
LLM_TIMEOUT=60
LLM_JSON_MODE=true
```

Với Ollama OpenAI-compatible API, ví dụ:

```dotenv
LLM_MODEL=llama3.1:8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=local-model
```

Chạy tập 100 mẫu:

```powershell
python run_benchmark.py --dataset data/multihop_100.json --out-dir outputs/llm_run --mode llm
python autograde.py --report-path outputs/llm_run/report.json
```

Có thể truyền `--model`, `--base-url`, `--temperature` và
`--reflexion-attempts`. Nếu endpoint không hỗ trợ `response_format`, đặt
`LLM_JSON_MODE=false`; prompt và Pydantic vẫn kiểm tra JSON trả về.

## Luồng xử lý

```text
QAExample
  -> Actor(question, context, reflection_memory)
  -> Evaluator(question, gold_answer, predicted_answer)
  -> đúng: dừng
  -> sai và còn lượt: Reflector -> cập nhật memory -> Actor thử lại
  -> RunRecord/AttemptTrace
  -> JSONL + report.json + report.md
```

LLM runtime validate structured output bằng Pydantic và thử lại tối đa hai lần nếu
JSON không hợp lệ. Token lấy từ `response.usage`; nếu provider không trả usage thì
dùng ước lượng ký tự và đánh dấu `token_is_estimate=true`. Latency được đo bằng
`time.perf_counter()`; mock dùng số ước lượng xác định.

## Dữ liệu

Mỗi dataset là một mảng JSON theo schema:

```json
{
  "qid": "my_q1",
  "difficulty": "medium",
  "question": "Câu hỏi multi-hop...",
  "gold_answer": "Đáp án đúng",
  "context": [
    {"title": "Nguồn 1", "text": "Thông tin liên quan..."},
    {"title": "Nguồn 2", "text": "Thông tin liên quan..."}
  ]
}
```

- `data/hotpot_mini.json`: 8 mẫu nhỏ để kiểm tra mock.
- `data/multihop_100.json`: 100 mẫu gồm 8 câu gốc và 92 câu multi-hop tự chứa,
  bao phủ chuỗi 2–3 hop về tác giả, địa lý, tiền tệ, âm nhạc và tổ chức.

Loader từ chối mảng rỗng, qid trùng, difficulty sai và context thiếu. Actor không
nhận `gold_answer`; chỉ Evaluator được dùng đáp án chuẩn.

## Đầu ra

Mỗi lần benchmark tạo trong `out_dir`:

- `react_runs.jsonl`: dấu vết baseline.
- `reflexion_runs.jsonl`: dấu vết Reflexion và reflection memory.
- `report.json`: báo cáo có cấu trúc cho autograder.
- `report.md`: bảng summary và discussion cho người đọc.

Summary gồm EM, số lượt, token, latency trung bình và delta Reflexion trừ ReAct.
Failure mode được nhóm ở cấp cao nhất theo loại lỗi để autograder đếm đúng.

## Kiểm thử

```powershell
python -m pytest
```

Test bao phủ chuẩn hóa Unicode, qid trùng, validation schema, điều kiện dừng,
reflection memory, tổng metric và cấu trúc report. Test dùng mock, không gọi API.

## Tiêu chí chấm điểm

| Phần | Điểm | Điều kiện |
|---|---:|---|
| Schema | 30 | Đủ `meta`, `summary`, `failure_modes`, `examples`, `extensions`, `discussion` |
| Experiment | 30 | Có ReAct + Reflexion, ≥100 records, ≥20 examples |
| Analysis | 20 | ≥3 failure modes, discussion ≥250 ký tự |
| Bonus | 20 | Tối đa hai extension được nhận diện |

Các extension được triển khai: `structured_evaluator`, `reflection_memory`,
`benchmark_report_json`, `mock_mode_for_autograding`.

Autograder chỉ kiểm tra cấu trúc và số lượng. Chất lượng prompt, tính đúng của phép
đo, độ sâu reasoning và chất lượng dữ liệu vẫn cần review thủ công.

## Thành phần mã nguồn

| File | Vai trò |
|---|---|
| `src/reflexion_lab/schemas.py` | Pydantic schema và validation |
| `src/reflexion_lab/runtime.py` | Protocol runtime và kết quả lời gọi có metric |
| `src/reflexion_lab/mock_runtime.py` | Backend xác định cho test/autograde |
| `src/reflexion_lab/llm_runtime.py` | Backend LLM OpenAI-compatible |
| `src/reflexion_lab/prompts.py` | System prompt Actor/Evaluator/Reflector |
| `src/reflexion_lab/agents.py` | Loop ReAct và Reflexion |
| `src/reflexion_lab/reporting.py` | Summary, failure analysis và report |
| `src/reflexion_lab/utils.py` | Load dữ liệu, normalize, ghi JSONL |
| `run_benchmark.py` | CLI chạy benchmark |
| `autograde.py` | Chấm report.json |
| `tests/` | Unit/integration test không dùng mạng |

Không commit `.env`, API key hoặc thư mục `outputs/`.
