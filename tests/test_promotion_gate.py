from animal_breed.evaluation.metrics import relative_promotion_gate


def test_relative_promotion_gate_pass():
    gate = relative_promotion_gate(
        current={"primary_metric": 0.8, "p95_ms": 40.0},
        baseline={"primary_metric": 0.7, "p95_ms": 38.0},
        primary_metric_key="primary_metric",
        latency_key="p95_ms",
        max_latency_regression_pct=10.0,
    )
    assert gate["gate_passed"] is True


def test_relative_promotion_gate_fail_on_quality():
    gate = relative_promotion_gate(
        current={"primary_metric": 0.68, "p95_ms": 37.0},
        baseline={"primary_metric": 0.7, "p95_ms": 38.0},
        primary_metric_key="primary_metric",
        latency_key="p95_ms",
        max_latency_regression_pct=10.0,
    )
    assert gate["gate_passed"] is False
