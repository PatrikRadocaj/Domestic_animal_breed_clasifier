"""Train a YOLO11 species-level detector with Ultralytics."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from animal_breed.utils.logging import setup_logging
from animal_breed.utils.reproducibility import run_metadata, seed_everything


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))

    detector_cfg = cfg.model.detector
    logger.info("Training detector with config:\n%s", OmegaConf.to_yaml(detector_cfg))
    logger.info("Run metadata: %s", run_metadata(cfg, repo_root="."))

    data_yaml = Path(detector_cfg.data_yaml_path)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"YOLO data yaml not found: {data_yaml}. "
            "Create it from the split manifest before detector training."
        )

    model = YOLO(str(detector_cfg.base_model))
    model.train(
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
        project=str(Path(cfg.train.checkpoint_dir) / "detector"),
        name=str(detector_cfg.name),
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
