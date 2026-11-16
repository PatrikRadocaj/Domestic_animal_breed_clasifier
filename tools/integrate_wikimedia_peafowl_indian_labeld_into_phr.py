"""Re-integrate the Wikimedia Indian peafowl `labeld` batch into PHR.

This targets only the `indian_peafowl/labeld` folder and replaces the existing
outputs for that source_dataset_id while leaving the other peafowl buckets
untouched.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import zipfile

import yaml


REPO_ROOT = Path(r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje")
SRC_DIR = REPO_ROOT / "data" / "raw" / "non_yolo_sources" / "wikimedia" / "peafowl" / "indian_peafowl" / "labeld"
OUT_ROOT = REPO_ROOT / "data" / "raw" / "yolo_sources" / "phr-labeled-images"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "dataset_integration" / "wikimedia_peafowl_indian_labeld_reimport.json"
SOURCE_DATASET_ID = "wikimedia-peafowl-indian_peafowl-labeld"
SPECIES = "peafowl"
SUBSPECIES = "indian_peafowl"
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MATCH_TOLERANCE = 0.03


@dataclass
class CsvRow:
    label_name: str
    bbox_xywhn: list[float]
    image_name: str


@dataclass
class YoloRow:
    class_id: int
    bbox_xywhn: list[float]


def normalize_label(value: str) -> str:
    out = []
    last_was_sep = False
    for ch in str(value).strip().lower():
        if ch.isalnum():
            out.append(ch)
            last_was_sep = False
        elif not last_was_sep:
            out.append("_")
            last_was_sep = True
    text = "".join(out).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def split_for_key(key: str) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def build_stem(image_stem: str) -> str:
    merged = f"{SOURCE_DATASET_ID}__{normalize_label(SUBSPECIES)}__{image_stem}"
    if len(merged) <= 140:
        return merged
    digest = hashlib.sha1(merged.encode("utf-8")).hexdigest()[:16]
    short_image = image_stem[:80].rstrip("_ ")
    return f"{SOURCE_DATASET_ID}__{normalize_label(SUBSPECIES)}__{short_image}__{digest}"


def load_species_class_ids() -> dict[str, int]:
    payload = yaml.safe_load((OUT_ROOT / "data.yaml").read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        ordered = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    else:
        ordered = [str(item) for item in names]
    if SPECIES not in ordered:
        ordered.append(SPECIES)
    return {name: idx for idx, name in enumerate(ordered)}


def update_data_yaml() -> None:
    path = OUT_ROOT / "data.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        ordered = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    else:
        ordered = [str(item) for item in names]
    if SPECIES not in ordered:
        ordered.append(SPECIES)
    payload["nc"] = len(ordered)
    payload["names"] = ordered
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def remove_existing_outputs() -> int:
    removed = 0
    for meta_path in (OUT_ROOT / "metadata").rglob("*.json"):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("source_dataset_id") != SOURCE_DATASET_ID:
            continue
        for path in (meta_path, Path(payload.get("image_path", "")), Path(payload.get("label_path", ""))):
            if path.exists():
                path.unlink()
                removed += 1
    return removed


def image_paths_in_dir(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS)


def read_csv_rows(csv_paths: list[Path]) -> dict[str, list[CsvRow]]:
    rows_by_image: dict[str, list[CsvRow]] = {}
    for csv_path in sorted(csv_paths):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                try:
                    width = float(raw["image_width"])
                    height = float(raw["image_height"])
                    xmin = float(raw["bbox_x"])
                    ymin = float(raw["bbox_y"])
                    bw = float(raw["bbox_width"])
                    bh = float(raw["bbox_height"])
                    xc = (xmin + (bw / 2.0)) / width
                    yc = (ymin + (bh / 2.0)) / height
                    wn = bw / width
                    hn = bh / height
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    continue
                image_name = str(raw.get("image_name", "")).strip()
                if not image_name:
                    continue
                rows_by_image.setdefault(image_name, []).append(
                    CsvRow(
                        label_name=str(raw.get("label_name", "")).strip(),
                        bbox_xywhn=[xc, yc, wn, hn],
                        image_name=image_name,
                    )
                )
    return rows_by_image


def read_zip_labels(zip_paths: list[Path]) -> dict[str, list[YoloRow]]:
    labels_by_image: dict[str, list[YoloRow]] = {}
    for zip_path in sorted(zip_paths):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".txt"):
                    continue
                stem = Path(info.filename).stem
                text = zf.read(info.filename).decode("utf-8", errors="ignore").strip()
                entries: list[YoloRow] = []
                for line in text.splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                        bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    except ValueError:
                        continue
                    entries.append(YoloRow(class_id=class_id, bbox_xywhn=bbox))
                if entries:
                    labels_by_image[stem] = entries
    return labels_by_image


def bbox_distance(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def resolve_objects(image_name: str, yolo_rows: list[YoloRow], csv_rows: list[CsvRow] | None, species_class_id: int) -> tuple[list[dict[str, object]] | None, str | None]:
    if csv_rows:
        matched: list[tuple[YoloRow, CsvRow]] = []
        used_csv_indexes: set[int] = set()
        for yolo_row in yolo_rows:
            candidates = []
            for idx, csv_row in enumerate(csv_rows):
                if idx in used_csv_indexes:
                    continue
                dist = bbox_distance(yolo_row.bbox_xywhn, csv_row.bbox_xywhn)
                if dist <= MATCH_TOLERANCE:
                    candidates.append((dist, idx, csv_row))
            if not candidates:
                return None, f"unresolved_csv_match:{image_name}"
            candidates.sort(key=lambda item: item[0])
            _, idx, csv_row = candidates[0]
            used_csv_indexes.add(idx)
            matched.append((yolo_row, csv_row))
        objects = []
        for yolo_row, csv_row in matched:
            objects.append(
                {
                    "class_id": species_class_id,
                    "class_name": SPECIES,
                    "species_name": SPECIES,
                    "subspecies": SUBSPECIES,
                    "source_label_name": str(csv_row.label_name or SUBSPECIES),
                    "bbox_xywhn": [round(value, 6) for value in yolo_row.bbox_xywhn],
                    "source_class_name": str(csv_row.label_name or SUBSPECIES),
                }
            )
        return objects, None

    if any(row.class_id != 0 for row in yolo_rows):
        return None, f"unexpected_nonzero_class_id:{image_name}"
    objects = []
    for yolo_row in yolo_rows:
        objects.append(
            {
                "class_id": species_class_id,
                "class_name": SPECIES,
                "species_name": SPECIES,
                "subspecies": SUBSPECIES,
                "source_label_name": SUBSPECIES,
                "bbox_xywhn": [round(value, 6) for value in yolo_row.bbox_xywhn],
                "source_class_name": SUBSPECIES,
            }
        )
    return objects, None


def integrate() -> dict[str, object]:
    species_class_ids = load_species_class_ids()
    if SPECIES not in species_class_ids:
        update_data_yaml()
        species_class_ids = load_species_class_ids()

    removed = remove_existing_outputs()
    image_paths = image_paths_in_dir(SRC_DIR)
    csv_paths = list(SRC_DIR.glob("*.csv"))
    zip_paths = list(SRC_DIR.glob("*.zip"))
    csv_map = read_csv_rows(csv_paths)
    label_map = read_zip_labels(zip_paths)

    imported = 0
    multi_object_images = 0
    exclusions: list[dict[str, str]] = []
    imported_by_split = {"train": 0, "valid": 0, "test": 0}
    source_label_counts: dict[str, int] = {}
    species_class_id = species_class_ids[SPECIES]

    for image_path in image_paths:
        yolo_rows = label_map.get(image_path.stem)
        if not yolo_rows:
            exclusions.append({"image": str(image_path), "reason": "missing_yolo_label"})
            continue
        csv_rows = csv_map.get(image_path.name)
        objects, error = resolve_objects(image_path.name, yolo_rows, csv_rows, species_class_id)
        if error or not objects:
            exclusions.append({"image": str(image_path), "reason": error or "empty_objects"})
            continue
        split = split_for_key(f"{SOURCE_DATASET_ID}::{image_path.stem}")
        merged_stem = build_stem(image_path.stem)
        dst_image = OUT_ROOT / "images" / split / f"{merged_stem}{image_path.suffix.lower()}"
        dst_label = OUT_ROOT / "labels" / split / f"{merged_stem}.txt"
        dst_meta = OUT_ROOT / "metadata" / split / f"{merged_stem}.json"
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dst_image)
        label_lines = [
            f"{obj['class_id']} {obj['bbox_xywhn'][0]:.6f} {obj['bbox_xywhn'][1]:.6f} {obj['bbox_xywhn'][2]:.6f} {obj['bbox_xywhn'][3]:.6f}"
            for obj in objects
        ]
        dst_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        payload = {
            "image_name": image_path.name,
            "folder_name": image_path.parent.name,
            "split": split,
            "image_path": str(dst_image.resolve()),
            "label_path": str(dst_label.resolve()),
            "objects": objects,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_species": SPECIES,
            "source_subspecies": SUBSPECIES,
            "source_dir": str(SRC_DIR.resolve()),
        }
        dst_meta.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        imported += 1
        imported_by_split[split] += 1
        if len(objects) > 1:
            multi_object_images += 1
        for obj in objects:
            key = str(obj.get("source_label_name") or SUBSPECIES)
            source_label_counts[key] = source_label_counts.get(key, 0) + 1

    integrity = {}
    for split in ("train", "valid", "test"):
        integrity[split] = {
            "images": len(list((OUT_ROOT / "images" / split).glob("*"))),
            "labels": len(list((OUT_ROOT / "labels" / split).glob("*.txt"))),
            "metadata": len(list((OUT_ROOT / "metadata" / split).glob("*.json"))),
        }

    return {
        "source_dataset_id": SOURCE_DATASET_ID,
        "removed_existing_triplet_files": removed,
        "imported_total": imported,
        "imported_by_split": imported_by_split,
        "multi_object_images": multi_object_images,
        "source_label_counts": dict(sorted(source_label_counts.items())),
        "excluded_total": len(exclusions),
        "excluded_examples": exclusions[:200],
        "integrity": integrity,
    }


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = integrate()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
