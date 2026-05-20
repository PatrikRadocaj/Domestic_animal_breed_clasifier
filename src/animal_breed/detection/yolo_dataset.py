"""Prepare YOLO-format detection datasets from the canonical manifest."""

from __future__ import annotations

from pathlib import Path
import shutil

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from animal_breed.data.manifest_schema import read_manifest
from animal_breed.data.taxonomy import load_taxonomy
from animal_breed.utils.logging import setup_logging


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _class_names(df: pd.DataFrame, class_level: str, taxonomy_path: str) -> list[str]:
    taxonomy = load_taxonomy(taxonomy_path)
    if class_level == "species":
        return [row["species_id"] for row in taxonomy.species_records() if row["species_id"] in set(df["species_id"])]
    if class_level == "breed":
        return [row["breed_id"] for row in taxonomy.breed_records() if row["breed_id"] in set(df["breed_id"])]
    raise ValueError(f"Unsupported class_level: {class_level}")


def _bbox_to_yolo(row: pd.Series) -> tuple[float, float, float, float] | None:
    bbox_cols = ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
    if any(pd.isna(row[col]) for col in bbox_cols):
        return None

    width = float(row.width)
    height = float(row.height)
    xmin = max(0.0, float(row.bbox_xmin))
    ymin = max(0.0, float(row.bbox_ymin))
    xmax = min(width, float(row.bbox_xmax))
    ymax = min(height, float(row.bbox_ymax))
    if xmax <= xmin or ymax <= ymin or width <= 0 or height <= 0:
        return None

    cx = ((xmin + xmax) / 2.0) / width
    cy = ((ymin + ymax) / 2.0) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return cx, cy, box_w, box_h


def prepare_yolo_dataset(
    manifest: pd.DataFrame,
    output_dir: str | Path,
    yaml_path: str | Path,
    taxonomy_path: str | Path,
    class_level: str = "species",
) -> dict[str, int]:
    output_dir = Path(output_dir)
    yaml_path = Path(yaml_path)
    names = _class_names(manifest, class_level, str(taxonomy_path))
    class_to_index = {name: idx for idx, name in enumerate(names)}
    stats = {"written": 0, "skipped_no_box": 0, "skipped_unknown_class": 0}

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Preparing YOLO"):
        row_series = pd.Series(row._asdict())
        split = str(row_series.get("split", "train"))
        class_name = str(row_series.species_id if class_level == "species" else row_series.breed_id)
        class_idx = class_to_index.get(class_name)
        if class_idx is None:
            stats["skipped_unknown_class"] += 1
            continue

        yolo_box = _bbox_to_yolo(row_series)
        if yolo_box is None:
            stats["skipped_no_box"] += 1
            continue

        src = Path(str(row_series.path))
        image_name = f"{row_series.image_id}{src.suffix.lower()}"
        image_dst = output_dir / "images" / split / image_name
        label_dst = output_dir / "labels" / split / f"{row_series.image_id}.txt"

        _link_or_copy(src, image_dst)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        cx, cy, box_w, box_h = yolo_box
        label_dst.write_text(
            f"{class_idx} {cx:.8f} {cy:.8f} {box_w:.8f} {box_h:.8f}\n",
            encoding="utf-8",
        )
        stats["written"] += 1

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return stats


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    manifest = read_manifest(cfg.data.split_manifest_path)
    class_level = str(cfg.model.detector.classes)
    output_dir = (
        cfg.data.prepared_yolo_species_dir
        if class_level == "species"
        else cfg.data.prepared_yolo_breed_dir
    )
    stats = prepare_yolo_dataset(
        manifest=manifest,
        output_dir=output_dir,
        yaml_path=cfg.model.detector.data_yaml_path,
        taxonomy_path=cfg.data.taxonomy_path,
        class_level=class_level,
    )
    logger.info("Prepared YOLO dataset: %s", stats)


if __name__ == "__main__":
    main()
