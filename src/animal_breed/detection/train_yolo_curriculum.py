"""Train YOLO single-stage model using species->breed curriculum."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from animal_breed.trainers.yolo_trainer import train_yolo_curriculum
from animal_breed.utils.logging import setup_logging
from animal_breed.utils.reproducibility import seed_everything


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
    summary = train_yolo_curriculum(cfg, logger)
    logger.info("YOLO curriculum done: %s", summary)


if __name__ == "__main__":
    main()
