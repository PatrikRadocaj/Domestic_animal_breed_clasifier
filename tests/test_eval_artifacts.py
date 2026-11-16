from __future__ import annotations

import json

from animal_breed.evaluation.runner import write_evaluation_artifacts
from animal_breed.utils.artifacts import create_artifact_bundle


def test_write_evaluation_artifacts(tmp_path):
    bundle = create_artifact_bundle("eval-test", artifacts_root=tmp_path)
    results = {
        "summary": {"single_stage": {"detection_map": 0.5}, "two_stage": {"breed_top1": 0.6}},
        "latency": {"single_stage": {"p95_ms": 10.0}, "two_stage": {"p95_ms": 12.0}},
        "ablation": [
            {"pipeline": "single_stage", "primary_metric": 0.5, "p95_ms": 10.0, "notes": "a"},
            {"pipeline": "two_stage", "primary_metric": 0.6, "p95_ms": 12.0, "notes": "b"},
        ],
        "promotion_gate": {"gate_passed": True},
        "qa_report": {"valid": True, "issue_count": 0, "issues": []},
    }

    write_evaluation_artifacts(bundle, results)

    summary_path = bundle.metrics_dir / "evaluation_summary.v1.json"
    latency_path = bundle.metrics_dir / "latency_report.v1.json"
    ablation_path = bundle.metrics_dir / "ablation_matrix.v1.csv"
    gate_path = bundle.metrics_dir / "promotion_gate.v1.json"
    qa_path = bundle.qa_dir / "data_quality_report.json"

    assert summary_path.exists()
    assert latency_path.exists()
    assert ablation_path.exists()
    assert gate_path.exists()
    assert qa_path.exists()

    assert json.loads(summary_path.read_text(encoding="utf-8"))["two_stage"]["breed_top1"] == 0.6
    assert "pipeline,primary_metric,p95_ms,notes" in ablation_path.read_text(encoding="utf-8")
