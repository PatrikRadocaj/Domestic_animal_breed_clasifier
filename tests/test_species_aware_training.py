from pathlib import Path
import logging

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from animal_breed.detection.species_aware_training import build_species_aware_training


def _mk_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((64, 64, 3), 180, dtype=np.uint8))


def _mk_dataset(root: Path, names: list[str], train_labels: list[str], val_labels: list[str]) -> None:
    (root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (root / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (root / "valid" / "labels").mkdir(parents=True, exist_ok=True)
    _mk_image(root / "train" / "images" / "a.jpg")
    _mk_image(root / "valid" / "images" / "b.jpg")
    (root / "train" / "labels" / "a.txt").write_text("\n".join(train_labels) + "\n", encoding="utf-8")
    (root / "valid" / "labels" / "b.txt").write_text("\n".join(val_labels) + "\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                f"names: {names}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_species_aware_eligibility_and_split_counts(tmp_path):
    yolo_root = tmp_path / "yolo_sources"
    ds1 = yolo_root / "dataset-dog-cat"
    ds2 = yolo_root / "dataset-horse-only"
    _mk_dataset(
        ds1,
        names=["dog", "cat"],
        train_labels=["0 0.5 0.5 0.5 0.5"],
        val_labels=["1 0.5 0.5 0.5 0.5"],
    )
    _mk_dataset(
        ds2,
        names=["horse"],
        train_labels=["0 0.5 0.5 0.5 0.5"],
        val_labels=["0 0.5 0.5 0.5 0.5"],
    )

    registry = tmp_path / "datasets_registry.csv"
    pd.DataFrame(
        [
            {"dataset_id": "dataset-dog-cat", "species_value": "multi"},
            {"dataset_id": "dataset-horse-only", "species_value": "horse"},
        ]
    ).to_csv(registry, index=False)

    cfg = OmegaConf.create(
        {
            "logging": {"level": "INFO"},
            "data": {
                "taxonomy_path": "data/label_maps/taxonomy_v0.yaml",
                "cross_dataset_label_map_path": "configs/mappings/cross_dataset_label_map.yaml",
                "yolo_sources_root": str(yolo_root),
                "species_aware": {
                    "registry_metadata_path": str(registry),
                    "output_root": str(tmp_path / "processed"),
                    "report_root": str(tmp_path / "reports"),
                    "train": {"enabled": False},
                },
            },
        }
    )
    logger = logging.getLogger("test_species_aware")
    result = build_species_aware_training(cfg, logger)

    report_path = Path(result["training_report_json"])
    assert report_path.exists()
    report = __import__("json").loads(report_path.read_text(encoding="utf-8"))

    dog_info = report["species"]["dog"]
    assert "dataset-dog-cat" in dog_info["included_dataset_ids"]
    assert any(x["dataset_id"] == "dataset-horse-only" for x in dog_info["excluded_dataset_ids"])
    assert int(dog_info["image_counts_by_split"].get("train", 0)) == 1

