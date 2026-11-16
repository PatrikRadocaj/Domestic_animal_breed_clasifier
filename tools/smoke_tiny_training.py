"""Tiny CPU smoke training for CI sanity checks."""

from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
from hydra import compose, initialize_config_dir
import numpy as np
import pandas as pd

from animal_breed.trainers.classifier_trainer import train_classifier
from animal_breed.utils.logging import setup_logging


def main() -> None:
    logger = setup_logging("INFO")
    repo_root = Path(__file__).resolve().parents[1]
    conf_dir = str((repo_root / "configs").resolve())

    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=[
            "train.device=cpu",
            "train.deterministic=true",
            "model/classifier=convnextv2_tiny",
            "model.classifier.pretrained=false",
            "model.classifier.img_size=64",
            "model.classifier.train.epochs=1",
            "model.classifier.train.batch_size=2",
            "model.classifier.train.num_workers=0",
            "eval.batch_size=2",
            "eval.num_workers=0",
            "paths.artifacts_dir=artifacts/smoke",
            "data.quality.dedup.enabled=false",
        ])

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        imgs = []
        labels = ["dog:beagle", "dog:boxer", "dog:beagle", "dog:boxer"]
        splits = ["train", "train", "val", "val"]
        for i in range(4):
            p = root / f"{i}.jpg"
            cv2.imwrite(str(p), np.full((64, 64, 3), 40 + i * 20, dtype=np.uint8))
            imgs.append(str(p))

        manifest = pd.DataFrame(
            {
                "image_id": [f"id{i}" for i in range(4)],
                "path": imgs,
                "source": ["smoke"] * 4,
                "license": ["unknown"] * 4,
                "species_id": ["dog"] * 4,
                "breed_id": labels,
                "width": [64] * 4,
                "height": [64] * 4,
                "bbox_xmin": [8] * 4,
                "bbox_ymin": [8] * 4,
                "bbox_xmax": [56] * 4,
                "bbox_ymax": [56] * 4,
                "is_crowd": [False] * 4,
                "split_group": [f"g{i}" for i in range(4)],
                "duplicate_cluster": [f"d{i}" for i in range(4)],
                "quality_tags": [""] * 4,
                "split": splits,
            }
        )
        manifest_path = root / "smoke_manifest.csv"
        manifest.to_csv(manifest_path, index=False)

        cfg.data.split_manifest_path = str(manifest_path)
        cfg.data.manifest_path = str(manifest_path)

        result = train_classifier(
            cfg=cfg,
            manifest=manifest,
            label_to_index={"dog:beagle": 0, "dog:boxer": 1},
            label_column="breed_id",
            run_prefix="smoke",
            logger=logger,
        )
        print(f"smoke_run_id={result.run_id}")


if __name__ == "__main__":
    main()
