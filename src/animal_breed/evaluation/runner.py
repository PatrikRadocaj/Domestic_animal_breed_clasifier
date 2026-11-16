"""Unified evaluation runner for two-stage and single-stage pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import hydra
import numpy as np
import pandas as pd
import timm
import torch
from omegaconf import DictConfig
from ultralytics import YOLO

from animal_breed.data.manifest_schema import read_manifest
from animal_breed.data.quality.qa import run_quality_checks
from animal_breed.data.transforms import classification_eval_transforms
from animal_breed.evaluation.metrics import (
    brier_score_multiclass,
    expected_calibration_error,
    macro_f1,
    model_size_mb,
    relative_promotion_gate,
    summarize_latency,
    topk_accuracy,
)
from animal_breed.utils.artifacts import (
    build_run_id,
    build_snapshot_hashes,
    create_artifact_bundle,
    write_json,
    write_run_metadata,
)
from animal_breed.utils.logging import setup_logging


@dataclass
class EvalRow:
    pipeline: str
    primary_metric: float
    p95_ms: float
    notes: str


def _load_classifier_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    label_to_index = payload["label_to_index"]
    labels = [label for label, _ in sorted(label_to_index.items(), key=lambda x: x[1])]
    classifier_cfg = payload.get("config", {}).get("model", {}).get("classifier", {})
    architecture = classifier_cfg.get("architecture", "tf_efficientnetv2_s.in21k_ft_in1k")
    img_size = int(classifier_cfg.get("img_size", 384))
    model = timm.create_model(str(architecture), pretrained=False, num_classes=len(labels))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, labels, img_size


def _crop_bbox(image: np.ndarray, row: pd.Series, context_ratio: float = 0.12) -> np.ndarray:
    if any(pd.isna(row[c]) for c in ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]):
        return image
    h, w = image.shape[:2]
    xmin, ymin, xmax, ymax = [float(row[c]) for c in ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]]
    bw = max(1.0, xmax - xmin)
    bh = max(1.0, ymax - ymin)
    mx = bw * context_ratio
    my = bh * context_ratio
    x0 = int(max(0, xmin - mx))
    y0 = int(max(0, ymin - my))
    x1 = int(min(w, xmax + mx))
    y1 = int(min(h, ymax + my))
    return image[y0:y1, x0:x1]


def _latency_for_detector(model: YOLO, images: list[str], warmup: int, bench: int) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        model.predict(source=images[:1], verbose=False)
    vals = []
    for i in range(max(1, bench)):
        start = time.perf_counter()
        model.predict(source=images[i % len(images): (i % len(images)) + 1], verbose=False)
        vals.append((time.perf_counter() - start) * 1000.0)
    return summarize_latency(vals)


def _evaluate_hierarchical_breed(cfg: DictConfig, manifest: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    val = manifest[manifest["split"] == "val"].reset_index(drop=True)
    species_model, species_labels, species_img_size = _load_classifier_checkpoint(Path(cfg.eval.species_model_path), device)
    breed_cache: dict[str, tuple[Any, list[str], int] | None] = {}

    y_true: list[int] = []
    y_pred: list[int] = []
    y_topk: list[list[int]] = []
    confidences: list[float] = []
    correct: list[int] = []
    probs_rows: list[np.ndarray] = []

    unique_breeds = sorted(val["breed_id"].dropna().astype(str).unique())
    breed_to_idx = {b: i for i, b in enumerate(unique_breeds)}

    for row in val.itertuples(index=False):
        img = cv2.imread(str(row.path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop = _crop_bbox(rgb, pd.Series(row._asdict()), context_ratio=float(cfg.model.pipeline.crop_context_ratio))

        tf = classification_eval_transforms(species_img_size)
        x = tf(image=crop)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            s_logits = species_model(x)
            s_probs = torch.softmax(s_logits, dim=1)[0]
            s_idx = int(torch.argmax(s_probs).item())
            species_pred = species_labels[s_idx]

        if species_pred not in breed_cache:
            ckpt = Path(cfg.eval.breed_models_dir) / species_pred / "best.pt"
            breed_cache[species_pred] = _load_classifier_checkpoint(ckpt, device) if ckpt.exists() else None

        bundle = breed_cache[species_pred]
        if bundle is None:
            continue

        breed_model, breed_labels, breed_img_size = bundle
        tf_b = classification_eval_transforms(breed_img_size)
        xb = tf_b(image=crop)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            b_logits = breed_model(xb)
            b_probs = torch.softmax(b_logits, dim=1)[0].cpu().numpy()

        pred_idx_local = int(np.argmax(b_probs))
        pred_breed = breed_labels[pred_idx_local]
        if str(row.breed_id) not in breed_to_idx or pred_breed not in breed_to_idx:
            continue

        topk_local = np.argsort(-b_probs)[: min(5, len(b_probs))]
        topk_global = [breed_to_idx[breed_labels[int(i)]] for i in topk_local if breed_labels[int(i)] in breed_to_idx]

        true_idx = breed_to_idx[str(row.breed_id)]
        pred_idx = breed_to_idx[pred_breed]

        probs_global = np.zeros(len(breed_to_idx), dtype=np.float32)
        for local_i, label in enumerate(breed_labels):
            if label in breed_to_idx:
                probs_global[breed_to_idx[label]] = float(b_probs[local_i])

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_topk.append(topk_global)
        conf = float(np.max(b_probs))
        confidences.append(conf)
        correct.append(int(pred_idx == true_idx))
        probs_rows.append(probs_global)

    probs = np.stack(probs_rows, axis=0) if probs_rows else np.zeros((0, len(breed_to_idx)), dtype=np.float32)
    return {
        "breed_top1": topk_accuracy(y_true, [[x] for x in y_pred], k=1),
        "breed_top5": topk_accuracy(y_true, y_topk, k=5),
        "breed_macro_f1": macro_f1(y_true, y_pred),
        "breed_ece": expected_calibration_error(confidences, correct, n_bins=int(cfg.eval.calibration_bins)),
        "breed_brier": brier_score_multiclass(y_true, probs, num_classes=max(1, len(breed_to_idx))),
        "evaluated_samples": int(len(y_true)),
    }


def _evaluate_detector_map(model_path: str, data_yaml: str) -> dict[str, float]:
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=False)
    box = getattr(metrics, "box", None)
    if box is None:
        return {"map50": 0.0, "map": 0.0}
    return {"map50": float(box.map50), "map": float(box.map)}


def _load_json_or_none(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    import json

    return json.loads(p.read_text(encoding="utf-8"))


def evaluate(cfg: DictConfig) -> dict[str, Any]:
    manifest = read_manifest(cfg.data.split_manifest_path)
    _, qa_report = run_quality_checks(manifest, cfg.data.taxonomy_path, dict(cfg.data.quality))
    if not qa_report["valid"]:
        raise ValueError("Data quality gate failed before evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device != "cpu" else "cpu")

    single_map = _evaluate_detector_map(str(cfg.eval.single_stage_detector_weights), str(cfg.eval.single_stage_data_yaml))
    two_species_map = _evaluate_detector_map(str(cfg.eval.two_stage_detector_weights), str(cfg.eval.two_stage_data_yaml))

    breed_eval = _evaluate_hierarchical_breed(cfg, manifest, device)

    sample_images = [str(p) for p in manifest["path"].head(int(cfg.eval.latency_sample_images)).tolist() if Path(str(p)).exists()]
    sample_images = sample_images or [str(Path(cfg.eval.source_path))] if hasattr(cfg.eval, "source_path") else []

    single_latency = {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    two_latency = {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    if sample_images:
        single_latency = _latency_for_detector(
            YOLO(str(cfg.eval.single_stage_detector_weights)),
            sample_images,
            warmup=int(cfg.eval.latency.warmup_iters),
            bench=int(cfg.eval.latency.benchmark_iters),
        )
        two_latency = _latency_for_detector(
            YOLO(str(cfg.eval.two_stage_detector_weights)),
            sample_images,
            warmup=int(cfg.eval.latency.warmup_iters),
            bench=int(cfg.eval.latency.benchmark_iters),
        )

    summary = {
        "single_stage": {
            "detection_map50": single_map["map50"],
            "detection_map": single_map["map"],
            "p95_ms": single_latency["p95_ms"],
            "model_size_mb": model_size_mb(str(cfg.eval.single_stage_detector_weights)),
        },
        "two_stage": {
            "species_detection_map50": two_species_map["map50"],
            "species_detection_map": two_species_map["map"],
            **breed_eval,
            "p95_ms": two_latency["p95_ms"],
            "detector_model_size_mb": model_size_mb(str(cfg.eval.two_stage_detector_weights)),
            "species_model_size_mb": model_size_mb(str(cfg.eval.species_model_path)),
        },
    }

    ablation = [
        asdict(EvalRow("single_stage", summary["single_stage"]["detection_map"], summary["single_stage"]["p95_ms"], "YOLO only")),
        asdict(EvalRow("two_stage", summary["two_stage"]["breed_top1"], summary["two_stage"]["p95_ms"], "YOLO + hierarchical classifier")),
    ]

    baseline = _load_json_or_none(cfg.eval.promotion.baseline_path)
    gate = relative_promotion_gate(
        current={
            "primary_metric": float(summary["two_stage"]["breed_top1"]),
            "p95_ms": float(summary["two_stage"]["p95_ms"]),
        },
        baseline=baseline,
        primary_metric_key="primary_metric",
        latency_key="p95_ms",
        max_latency_regression_pct=float(cfg.eval.promotion.max_latency_regression_pct),
    )

    return {
        "summary": summary,
        "ablation": ablation,
        "promotion_gate": gate,
        "latency": {
            "single_stage": single_latency,
            "two_stage": two_latency,
        },
        "qa_report": qa_report,
    }


def write_evaluation_artifacts(bundle, results: dict[str, Any]) -> None:
    write_json(bundle.metrics_dir / "evaluation_summary.v1.json", results["summary"])
    write_json(bundle.metrics_dir / "latency_report.v1.json", results["latency"])
    pd.DataFrame(results["ablation"]).to_csv(bundle.metrics_dir / "ablation_matrix.v1.csv", index=False)
    write_json(bundle.metrics_dir / "promotion_gate.v1.json", results["promotion_gate"])
    write_json(bundle.qa_dir / "data_quality_report.json", results["qa_report"])


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    run_id = build_run_id("eval", cfg)
    bundle = create_artifact_bundle(run_id, artifacts_root=str(cfg.paths.artifacts_dir))
    snapshot_hashes = build_snapshot_hashes(
        cfg,
        manifest_path=cfg.data.split_manifest_path,
        taxonomy_path=cfg.data.taxonomy_path,
        source_registry_path=getattr(cfg.data, "source_registry_path", None),
    )
    write_run_metadata(bundle, cfg, snapshot_hashes)

    results = evaluate(cfg)
    write_evaluation_artifacts(bundle, results)

    logger.info("Evaluation artifacts saved under %s", bundle.metrics_dir)


if __name__ == "__main__":
    main()
