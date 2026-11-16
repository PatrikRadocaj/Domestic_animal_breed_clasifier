"""Train YOLO detector through shared trainer core."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from animal_breed.data.manifest_schema import read_manifest
from animal_breed.data.quality.qa import run_quality_checks
from animal_breed.trainers.yolo_trainer import train_yolo
from animal_breed.utils.artifacts import write_json
from animal_breed.utils.logging import setup_logging
from animal_breed.utils.reproducibility import run_metadata, seed_everything


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
    detector_cfg = cfg.model.detector
    logger.info("Training detector with config:\n%s", OmegaConf.to_yaml(detector_cfg))
    logger.info("Run metadata: %s", run_metadata(cfg, repo_root="."))

    manifest = read_manifest(cfg.data.split_manifest_path)
    _, qa_report = run_quality_checks(manifest, cfg.data.taxonomy_path, dict(cfg.data.quality))
    if not qa_report["valid"]:
        raise ValueError("Data quality gate failed before detector training")

    result = train_yolo(cfg, logger, run_prefix="detector")
    write_json(f"{result['artifact_root']}/qa/data_quality_report.json", qa_report)


if __name__ == "__main__":
    main()
