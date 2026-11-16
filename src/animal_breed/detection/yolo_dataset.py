"""Prepare YOLO-format detection datasets from the canonical manifest."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import cv2
import numpy as np
import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from animal_breed.data.datasets import BreedClassificationDataset
from animal_breed.data.manifest_schema import read_manifest
from animal_breed.data.taxonomy import load_taxonomy
from animal_breed.utils.logging import setup_logging


def _load_detection_rgb_uint8(src: Path) -> np.ndarray:
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image for YOLO export: {src}")
    return BreedClassificationDataset._normalize_image(image)


def _export_detection_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    image = _load_detection_rgb_uint8(src)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(dst), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise RuntimeError(f"Failed to write normalized YOLO export image: {dst}")


def _class_names(df: pd.DataFrame, class_level: str, taxonomy_path: str) -> list[str]:
    if class_level == "species":
        taxonomy = load_taxonomy(taxonomy_path)
        return [row["species_id"] for row in taxonomy.species_records() if row["species_id"] in set(df["species_id"])]
    if class_level == "breed":
        taxonomy = load_taxonomy(taxonomy_path)
        return [row["breed_id"] for row in taxonomy.breed_records() if row["breed_id"] in set(df["breed_id"])]
    if class_level == "species_or_breed":
        if "direct_label_id" not in df.columns:
            raise ValueError("species_or_breed export requires a direct_label_id column in the manifest.")
        taxonomy = load_taxonomy(taxonomy_path)
        direct_labels = set(df["direct_label_id"].dropna().astype(str))
        species_with_breeds = {row["species_id"] for row in taxonomy.breed_records()}
        names: list[str] = []
        for row in taxonomy.species_records():
            species_id = row["species_id"]
            if species_id in direct_labels and species_id not in species_with_breeds:
                names.append(species_id)
        for row in taxonomy.breed_records():
            breed_id = row["breed_id"]
            if breed_id in direct_labels:
                names.append(breed_id)
        leftovers = sorted(direct_labels - set(names))
        names.extend(leftovers)
        return names
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
    stats = {
        "written": 0,
        "skipped_no_box": 0,
        "skipped_unknown_class": 0,
        "skipped_bad_image": 0,
    }
    label_lines: dict[Path, list[str]] = {}
    linked_images: set[Path] = set()

    index_rows: list[dict[str, object]] = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Preparing YOLO"):
        split = str(getattr(row, "split", "train"))
        if class_level == "species":
            class_name = str(row.species_id)
        elif class_level == "breed":
            class_name = str(row.breed_id)
        elif class_level == "species_or_breed":
            class_name = str(row.direct_label_id)
        else:
            raise ValueError(f"Unsupported class_level: {class_level}")
        class_idx = class_to_index.get(class_name)
        if class_idx is None:
            stats["skipped_unknown_class"] += 1
            continue

        yolo_box = _bbox_to_yolo(pd.Series(row._asdict()))
        if yolo_box is None:
            stats["skipped_no_box"] += 1
            continue

        src = Path(str(row.path))
        image_key = sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:20]
        image_name = f"{image_key}.jpg"
        image_dst = output_dir / "images" / split / image_name
        label_dst = output_dir / "labels" / split / f"{image_key}.txt"

        if image_dst not in linked_images:
            try:
                _export_detection_image(src, image_dst)
            except Exception:
                stats["skipped_bad_image"] += 1
                continue
            linked_images.add(image_dst)
        cx, cy, box_w, box_h = yolo_box
        label_lines.setdefault(label_dst, []).append(
            f"{class_idx} {cx:.8f} {cy:.8f} {box_w:.8f} {box_h:.8f}"
        )
        index_rows.append(
            {
                "image_id": image_key,
                "annotation_id": str(row.image_id),
                "split": split,
                "class_name": class_name,
                "class_idx": int(class_idx),
                "image_path": str(image_dst.resolve()),
                "label_path": str(label_dst.resolve()),
            }
        )
        stats["written"] += 1

    for label_dst, lines in label_lines.items():
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    if index_rows:
        pd.DataFrame(index_rows).to_parquet(output_dir / "prepare_yolo_index.parquet", index=False)
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
