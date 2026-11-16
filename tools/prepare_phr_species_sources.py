"""Prepare horse and cattle YOLO sources for inclusion in phr-labeled-images."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re


REPO_ROOT = Path(r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje")
HORSE_YOLO_ROOT = REPO_ROOT / "data" / "raw" / "yolo_sources" / "horse-breeds-olgabelitskaya-yolo"
HORSE_CSV_PATH = (
    REPO_ROOT
    / "data"
    / "raw"
    / "non_yolo_sources"
    / "horse-breeds-olgabelitskaya"
    / "labels_my-project-name_2026-06-02-11-55-25.csv"
)
CATTLE_YOLO_ROOT = REPO_ROOT / "data" / "raw" / "yolo_sources" / "cattle-breeds-anandkumarsahu09-yolo"


def slugify(text: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in str(text).strip()]
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out


def load_horse_breed_map() -> dict[str, tuple[str, str]]:
    prefix_to_label: dict[str, str] = {}
    with HORSE_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = str(row.get("image_name", ""))
            label_name = str(row.get("label_name", "")).strip()
            match = re.match(r"(\d+)_", image_name)
            if not match or not label_name:
                continue
            prefix = match.group(1)
            prefix_to_label.setdefault(prefix, label_name)
    return {prefix: (label, slugify(label)) for prefix, label in prefix_to_label.items()}


def rename_horse_files(prefix_map: dict[str, tuple[str, str]]) -> None:
    for split in ("train", "valid", "test"):
        image_dir = HORSE_YOLO_ROOT / split / "images"
        label_dir = HORSE_YOLO_ROOT / split / "labels"
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue
            stem = image_path.stem
            match = re.match(r"(\d+)_([0-9]+)$", stem)
            if not match:
                continue
            prefix, index = match.groups()
            if prefix not in prefix_map:
                continue
            _, breed_slug = prefix_map[prefix]
            new_stem = f"{breed_slug}_{index}"
            new_image_path = image_path.with_name(f"{new_stem}{image_path.suffix.lower()}")
            label_path = label_dir / f"{stem}.txt"
            new_label_path = label_dir / f"{new_stem}.txt"
            if image_path != new_image_path:
                if new_image_path.exists():
                    raise FileExistsError(f"Refusing to overwrite existing horse image: {new_image_path}")
                image_path.rename(new_image_path)
            if label_path.exists() and label_path != new_label_path:
                if new_label_path.exists():
                    raise FileExistsError(f"Refusing to overwrite existing horse label: {new_label_path}")
                label_path.rename(new_label_path)


def read_yolo_boxes(label_path: Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


def write_species_metadata(
    dataset_root: Path,
    species_name: str,
    infer_subspecies,
) -> None:
    for split in ("train", "valid", "test"):
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        metadata_dir = dataset_root / split / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        for label_path in sorted(label_dir.glob("*.txt")):
            image_matches = list(image_dir.glob(f"{label_path.stem}.*"))
            if len(image_matches) != 1:
                continue
            image_path = image_matches[0]
            subspecies, source_label_name = infer_subspecies(label_path.stem)
            objects = [
                {
                    "class_id": 0,
                    "class_name": species_name,
                    "species_name": species_name,
                    "subspecies": subspecies,
                    "source_label_name": source_label_name,
                    "bbox_xywhn": [round(v, 6) for v in bbox],
                }
                for bbox in read_yolo_boxes(label_path)
            ]
            if not objects:
                continue
            payload = {
                "image_name": image_path.name,
                "folder_name": dataset_root.name,
                "split": split,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "objects": objects,
            }
            (metadata_dir / f"{label_path.stem}.json").write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )


def infer_horse_subspecies(stem: str) -> tuple[str, str]:
    match = re.match(r"([a-z0-9_]+)_\d+$", stem)
    breed_slug = match.group(1) if match else "unknown"
    source_label = breed_slug.replace("_", " ").title()
    source_label = source_label.replace("Akhal Teke", "Akhal-Teke")
    return breed_slug, source_label


def infer_cattle_subspecies(stem: str) -> tuple[str, str]:
    lowered = stem.lower()
    mapping = [
        ("ayrshirecattle", "ayrshire", "Ayrshire cattle"),
        ("brownswisscattle", "brown_swiss", "Brown Swiss cattle"),
        ("holsteinfriesiancattle", "holstein_friesian", "Holstein Friesian cattle"),
        ("jerseycattle", "jersey", "Jersey cattle"),
        ("reddanecattle", "red_dane", "Red Dane cattle"),
    ]
    for prefix, slug, label in mapping:
        if lowered.startswith(prefix):
            return slug, label
    return "unknown", "cow"


def main() -> None:
    prefix_map = load_horse_breed_map()
    rename_horse_files(prefix_map)
    write_species_metadata(HORSE_YOLO_ROOT, "horse", infer_horse_subspecies)
    write_species_metadata(CATTLE_YOLO_ROOT, "cattle", infer_cattle_subspecies)
    print(
        json.dumps(
            {
                "horse_prefix_map": {k: {"label": v[0], "slug": v[1]} for k, v in sorted(prefix_map.items())},
                "horse_root": str(HORSE_YOLO_ROOT),
                "cattle_root": str(CATTLE_YOLO_ROOT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
