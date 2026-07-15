from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.mock_runtime import MockRuntime
from src.reflexion_lab.reporting import build_report
from src.reflexion_lab.utils import load_dataset

def test_report_matches_autograde_shape():
    examples = load_dataset("data/multihop_100.json")
    runtime = MockRuntime()
    records = [ReActAgent(runtime).run(item) for item in examples]
    records += [ReflexionAgent(runtime=runtime).run(item) for item in examples]
    report = build_report(records, "multihop_100.json")
    assert set(report.model_dump()) == {"meta", "summary", "failure_modes", "examples", "extensions", "discussion"}
    assert {"react", "reflexion", "delta_reflexion_minus_react"} <= set(report.summary)
    assert {"incomplete_multi_hop", "entity_drift", "wrong_final_answer"} <= set(report.failure_modes)
    assert report.meta["num_records"] == 200
    assert len(report.examples) == 200
    assert len(report.discussion) >= 250
