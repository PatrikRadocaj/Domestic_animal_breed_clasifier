"""Shared YOLO training core with artifact contract support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from ultralytics import YOLO

from animal_breed.utils.artifacts import (
    build_run_id,
    build_snapshot_hashes,
    create_artifact_bundle,
    write_json,
    write_run_metadata,
)


def train_yolo(cfg, logger, run_prefix: str = "yolo") -> dict[str, Any]:
    detector_cfg = cfg.model.detector
    data_yaml = Path(detector_cfg.data_yaml_path)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"YOLO data yaml not found: {data_yaml}. "
            "Create it from the split manifest before detector training."
        )

    run_id = build_run_id(run_prefix, cfg)
    bundle = create_artifact_bundle(run_id, artifacts_root=str(cfg.paths.artifacts_dir))
    snapshot_hashes = build_snapshot_hashes(
        cfg,
        manifest_path=cfg.data.split_manifest_path,
        taxonomy_path=cfg.data.taxonomy_path,
        source_registry_path=getattr(cfg.data, "source_registry_path", None),
    )
    write_run_metadata(bundle, cfg, snapshot_hashes)

    model = YOLO(str(detector_cfg.base_model))
    train_save_dir = bundle.checkpoints_dir / str(detector_cfg.name)
    results = model.train(
        data=str(data_yaml),
        imgsz=int(detector_cfg.imgsz),
        epochs=int(detector_cfg.train.epochs),
        batch=int(detector_cfg.train.batch),
        workers=int(detector_cfg.train.workers),
        patience=int(detector_cfg.train.patience),
        optimizer=str(detector_cfg.train.optimizer),
        lr0=float(detector_cfg.train.lr0),
        cos_lr=bool(detector_cfg.train.cos_lr),
        close_mosaic=int(detector_cfg.train.close_mosaic),
        amp=bool(detector_cfg.train.amp),
        cache=bool(detector_cfg.train.cache),
        seed=int(cfg.seed),
        project=str(bundle.checkpoints_dir),
        name=str(detector_cfg.name),
        device=str(cfg.train.device),
        resume=bool(cfg.train.resume),
        exist_ok=True,
        plots=True,
    )

    weights_dir = train_save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    result_payload = {
        "run_id": run_id,
        "artifact_root": str(bundle.root.resolve()),
        "checkpoints_dir": str(bundle.checkpoints_dir.resolve()),
        "train_save_dir": str(train_save_dir.resolve()),
        "best_checkpoint": str(best_pt.resolve()) if best_pt.exists() else "",
        "last_checkpoint": str(last_pt.resolve()) if last_pt.exists() else "",
        "yolo_results": str(results),
    }
    write_json(bundle.metrics_dir / "yolo_train_summary.json", result_payload)
    logger.info("YOLO training artifact root: %s", bundle.root)
    return result_payload


def train_yolo_curriculum(cfg, logger) -> dict[str, Any]:
    original_detector = cfg.model.detector
    detector_species_cfg = OmegaConf.load("configs/model/detector/yolo11_species.yaml")
    detector_breed_cfg = OmegaConf.load("configs/model/detector/yolo11_breed.yaml")

    # Stage A: species pretrain
    cfg.model.detector = detector_species_cfg
    stage_a = train_yolo(cfg, logger, run_prefix="yolo-curriculum-a")

    # Stage B: breed finetune initialized from stage A best
    cfg.model.detector = detector_breed_cfg
    checkpoints_dir = Path(stage_a["checkpoints_dir"]) / str(detector_species_cfg.name) / "weights"
    best_pt = checkpoints_dir / "best.pt"
    if best_pt.exists():
        cfg.model.detector.base_model = str(best_pt)
    stage_b = train_yolo(cfg, logger, run_prefix="yolo-curriculum-b")

    # Stage C: long-tail recovery with stronger augmentation/class balancing hints
    cfg.model.detector = detector_breed_cfg
    cfg.model.detector.train.close_mosaic = max(0, int(cfg.model.detector.train.close_mosaic) - 3)
    stage_c = train_yolo(cfg, logger, run_prefix="yolo-curriculum-c")

    cfg.model.detector = original_detector
    return {"stage_a": stage_a, "stage_b": stage_b, "stage_c": stage_c}
