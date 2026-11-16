"""Validate cross-dataset label mapping and regenerate unmapped-label reports."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from animal_breed.data.label_mapping import CrossDatasetLabelMap
from animal_breed.data.manifests import build_manifest_from_tree
from animal_breed.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    map_path = Path(str(cfg.data.cross_dataset_label_map_path))
    CrossDatasetLabelMap.from_file(map_path)
    logger.info("Cross-dataset label map validated: %s", map_path)

    _ = build_manifest_from_tree(cfg)

    unmapped_path = Path(str(cfg.data.unmapped_labels_report_path))
    manual_path = Path(str(cfg.data.manual_review_report_path))
    unmapped_count = 0
    manual_count = 0
    if unmapped_path.exists():
        unmapped_count = max(0, sum(1 for _ in unmapped_path.open("r", encoding="utf-8")) - 1)
    if manual_path.exists():
        manual_count = max(0, sum(1 for _ in manual_path.open("r", encoding="utf-8")) - 1)

    logger.info("Unmapped labels report: %s (rows=%d)", unmapped_path, unmapped_count)
    logger.info("Manual review report: %s (rows=%d)", manual_path, manual_count)


if __name__ == "__main__":
    main()

