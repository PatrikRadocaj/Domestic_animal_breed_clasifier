"""Evaluation metrics helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def topk_accuracy(y_true: list[int], y_topk: list[list[int]], k: int = 1) -> float:
    if not y_true:
        return 0.0
    hits = 0
    for target, preds in zip(y_true, y_topk):
        if target in preds[:k]:
            hits += 1
    return float(hits / len(y_true))


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def brier_score_multiclass(y_true: list[int], probs: np.ndarray, num_classes: int) -> float:
    if probs.size == 0:
        return 0.0
    y_onehot = np.zeros((len(y_true), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_true)), np.array(y_true)] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def expected_calibration_error(confidences: list[float], correct: list[int], n_bins: int = 15) -> float:
    if not confidences:
        return 0.0
    conf = np.array(confidences, dtype=np.float32)
    corr = np.array(correct, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(corr[mask].mean())
        ece += (float(mask.mean()) * abs(bin_acc - bin_conf))
    return float(ece)


def summarize_latency(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    arr = np.array(values_ms, dtype=np.float32)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }


def model_size_mb(path: str) -> float:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return 0.0
    return float(p.stat().st_size / (1024 * 1024))


def relative_promotion_gate(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    primary_metric_key: str,
    latency_key: str,
    max_latency_regression_pct: float,
) -> dict[str, Any]:
    if baseline is None:
        return {
            "gate_passed": True,
            "reason": "No baseline available; initializing baseline run.",
            "baseline_present": False,
        }

    current_primary = float(current.get(primary_metric_key, 0.0))
    baseline_primary = float(baseline.get(primary_metric_key, 0.0))
    current_latency = float(current.get(latency_key, 0.0))
    baseline_latency = float(baseline.get(latency_key, 0.0))

    quality_improved = current_primary > baseline_primary
    latency_budget = baseline_latency * (1.0 + max_latency_regression_pct / 100.0)
    latency_ok = current_latency <= latency_budget if baseline_latency > 0 else True

    return {
        "gate_passed": bool(quality_improved and latency_ok),
        "baseline_present": True,
        "quality_improved": bool(quality_improved),
        "latency_ok": bool(latency_ok),
        "current_primary": current_primary,
        "baseline_primary": baseline_primary,
        "current_latency": current_latency,
        "baseline_latency": baseline_latency,
        "latency_budget": float(latency_budget),
    }
