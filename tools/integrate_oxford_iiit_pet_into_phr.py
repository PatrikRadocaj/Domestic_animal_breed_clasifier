"""Integrate Oxford-IIIT Pet `images_1` into the merged PHR YOLO source.

This importer keeps YOLO labels at species level (`cat` / `dog`) while preserving
breed provenance in metadata JSON files, matching the existing `phr-labeled-images`
layout and semantics.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Iterable
import zipfile

import yaml


REPO_ROOT = Path(r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje")
SRC_ROOT = REPO_ROOT / "data" / "raw" / "non_yolo_sources" / "oxford-iiit-pet-vgg" / "images_1"
OUT_ROOT = REPO_ROOT / "data" / "raw" / "yolo_sources" / "phr-labeled-images"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "dataset_integration" / "oxford_iiit_pet_images_1_to_phr_labeled_images.json"
DATASET_ID = "oxford-iiit-pet-vgg-images_1"
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MATCH_TOLERANCE = 0.03
SPECIES_ALIASES = {
    "cat": "cat",
    "dog": "dog",
    "duck": "duck",
    "goose": "goose",
    "chicken": "chicken",
    "cattle": "cattle",
    "cow": "cattle",
    "horse": "horse",
    "pigeon": "pigeon",
    "turkey": "turkey",
    "alpaca": "alpaca",
    "goat": "goat",
    "sheep": "sheep",
}


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


def build_stem(folder_hint: str, image_stem: str) -> str:
    merged = f"{DATASET_ID}__{folder_hint}__{image_stem}"
    if len(merged) <= 140:
        return merged
    digest = hashlib.sha1(merged.encode("utf-8")).hexdigest()[:16]
    short_hint = folder_hint[:40].rstrip("_ ")
    short_image = image_stem[:60].rstrip("_ ")
    return f"{DATASET_ID}__{short_hint}__{short_image}__{digest}"


def load_species_class_ids() -> dict[str, int]:
    payload = yaml.safe_load((OUT_ROOT / "data.yaml").read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        ordered = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    else:
        ordered = [str(item) for item in names]
    return {name: idx for idx, name in enumerate(ordered)}


def remove_existing_oxford_outputs() -> int:
    removed = 0
    for meta_path in (OUT_ROOT / "metadata").rglob("*.json"):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload.get("source_dataset_id") != DATASET_ID:
            continue
        image_path = Path(payload["image_path"])
        label_path = Path(payload["label_path"])
        for path in (meta_path, image_path, label_path):
            if path.exists():
                path.unlink()
                removed += 1
    return removed


def collect_source_batches() -> list[dict[str, object]]:
    batches: list[dict[str, object]] = []
    for species in ("cat", "dog"):
        species_root = SRC_ROOT / species
        for breed_dir in sorted(path for path in species_root.iterdir() if path.is_dir()):
            batches.append(
                {
                    "species": species,
                    "primary_label": breed_dir.name,
                    "folder_hint": normalize_label(breed_dir.name),
                    "dir": breed_dir,
                    "allow_folder_primary": True,
                }
            )
    for species in ("cat", "dog"):
        missing_root = SRC_ROOT / "missing_labels" / species
        batches.append(
                {
                    "species": species,
                    "primary_label": None,
                    "folder_hint": f"missing_labels_{species}",
                    "dir": missing_root,
                    "allow_folder_primary": False,
                    "recursive_images": True,
                }
            )
    return batches


def image_paths_in_dir(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS
    )


def read_csv_rows(csv_paths: Iterable[Path]) -> dict[str, list[CsvRow]]:
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


def read_zip_labels(zip_paths: Iterable[Path]) -> dict[str, list[YoloRow]]:
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
                if not entries:
                    continue
                existing = labels_by_image.get(stem)
                if existing is None:
                    labels_by_image[stem] = entries
                    continue
                existing_key = sorted((row.class_id, tuple(round(x, 6) for x in row.bbox_xywhn)) for row in existing)
                new_key = sorted((row.class_id, tuple(round(x, 6) for x in row.bbox_xywhn)) for row in entries)
                if existing_key != new_key:
                    raise ValueError(f"Conflicting YOLO label entries for {stem} across ZIP files")
    return labels_by_image


def bbox_distance(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def canonical_subspecies(species: str, raw_label: str) -> str | None:
    norm = normalize_label(raw_label)
    if norm in {"", species}:
        return species
    return norm


def infer_species_for_label(batch_species: str, raw_label: str) -> str | None:
    norm = normalize_label(raw_label)
    alias = SPECIES_ALIASES.get(norm)
    if alias is not None:
        return alias if alias in SPECIES_CLASS_IDS else None
    return batch_species


def resolve_objects(
    *,
    species: str,
    primary_label: str | None,
    allow_folder_primary: bool,
    image_name: str,
    yolo_rows: list[YoloRow],
    csv_rows: list[CsvRow] | None,
) -> tuple[list[dict[str, object]] | None, str | None]:
    if csv_rows:
        remaining_csv = list(enumerate(csv_rows))
        matched: list[tuple[YoloRow, CsvRow]] = []
        used_csv_indexes: set[int] = set()
        for yolo_row in yolo_rows:
            candidates = []
            for idx, csv_row in remaining_csv:
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
            resolved_species = infer_species_for_label(species, csv_row.label_name)
            if resolved_species is None or resolved_species not in SPECIES_CLASS_IDS:
                continue
            subspecies = canonical_subspecies(resolved_species, csv_row.label_name)
            objects.append(
                {
                    "class_id": SPECIES_CLASS_IDS[resolved_species],
                    "class_name": resolved_species,
                    "species_name": resolved_species,
                    "subspecies": subspecies,
                    "source_label_name": csv_row.label_name,
                    "bbox_xywhn": [round(value, 6) for value in yolo_row.bbox_xywhn],
                    "source_class_name": csv_row.label_name,
                }
            )
        if not objects:
            return None, f"all_csv_labels_out_of_taxonomy:{image_name}"
        return objects, None

    if not allow_folder_primary or not primary_label:
        return None, f"missing_csv_for_non_primary_batch:{image_name}"
    if any(row.class_id != 0 for row in yolo_rows):
        return None, f"missing_csv_for_multiclass_batch:{image_name}"

    objects = []
    for yolo_row in yolo_rows:
        objects.append(
            {
                "class_id": SPECIES_CLASS_IDS[species],
                "class_name": species,
                "species_name": species,
                "subspecies": canonical_subspecies(species, primary_label),
                "source_label_name": primary_label,
                "bbox_xywhn": [round(value, 6) for value in yolo_row.bbox_xywhn],
                "source_class_name": primary_label,
            }
        )
    return objects, None


def global_integrity_summary() -> dict[str, object]:
    counts: dict[str, dict[str, int]] = {}
    missing_metadata: list[str] = []
    orphan_metadata: list[str] = []
    invalid_metadata = 0
    for split in ("train", "valid", "test"):
        image_stems = {path.stem for path in (OUT_ROOT / "images" / split).glob("*") if path.is_file()}
        label_stems = {path.stem for path in (OUT_ROOT / "labels" / split).glob("*.txt")}
        meta_paths = list((OUT_ROOT / "metadata" / split).glob("*.json"))
        meta_stems = {path.stem for path in meta_paths}
        counts[split] = {
            "images": len(image_stems),
            "labels": len(label_stems),
            "metadata": len(meta_stems),
        }
        missing_metadata.extend(sorted(image_stems - meta_stems))
        orphan_metadata.extend(sorted(meta_stems - image_stems))
        for meta_path in meta_paths:
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                invalid_metadata += 1
                continue
            if not Path(payload.get("image_path", "")).exists() or not Path(payload.get("label_path", "")).exists():
                invalid_metadata += 1
    return {
        "split_counts": counts,
        "missing_metadata_stems": missing_metadata[:20],
        "orphan_metadata_stems": orphan_metadata[:20],
        "invalid_metadata_count": invalid_metadata,
    }


def integrate() -> dict[str, object]:
    removed = remove_existing_oxford_outputs()
    imported = 0
    imported_by_species = {"cat": 0, "dog": 0}
    imported_by_breed: dict[str, int] = {}
    imported_by_batch_type = {"regular": 0, "missing_labels": 0}
    exclusions: list[dict[str, str]] = []
    csv_backed = 0
    folder_primary_only = 0
    multi_object_images = 0
    multi_label_images = 0
    filtered_out_of_taxonomy_objects = 0

    for batch in collect_source_batches():
        batch_dir = Path(batch["dir"])
        species = str(batch["species"])
        primary_label = batch["primary_label"]
        allow_folder_primary = bool(batch["allow_folder_primary"])
        base_folder_hint = str(batch["folder_hint"])
        recursive_images = bool(batch.get("recursive_images", False))
        image_paths = image_paths_in_dir(batch_dir, recursive_images)
        zip_paths = list(batch_dir.glob("*.zip"))
        csv_paths = list(batch_dir.glob("*.csv"))
        label_map = read_zip_labels(zip_paths)
        csv_map = read_csv_rows(csv_paths)

        for image_path in image_paths:
            image_stem = image_path.stem
            yolo_rows = label_map.get(image_stem)
            if not yolo_rows:
                exclusions.append({"image": str(image_path), "reason": "missing_yolo_label"})
                continue
            csv_rows = csv_map.get(image_path.name)
            objects, error = resolve_objects(
                species=species,
                primary_label=primary_label,
                allow_folder_primary=allow_folder_primary,
                image_name=image_path.name,
                yolo_rows=yolo_rows,
                csv_rows=csv_rows,
            )
            if error or not objects:
                exclusions.append({"image": str(image_path), "reason": error or "empty_objects"})
                continue
            filtered_out_of_taxonomy_objects += max(0, len(yolo_rows) - len(objects))

            folder_hint = normalize_label(image_path.parent.name) if recursive_images else base_folder_hint
            split = split_for_key(f"{species}::{folder_hint}::{image_stem}")
            merged_stem = build_stem(folder_hint, image_stem)
            dst_image = OUT_ROOT / "images" / split / f"{merged_stem}{image_path.suffix.lower()}"
            dst_label = OUT_ROOT / "labels" / split / f"{merged_stem}.txt"
            dst_meta = OUT_ROOT / "metadata" / split / f"{merged_stem}.json"
            if dst_image.exists() or dst_label.exists() or dst_meta.exists():
                raise FileExistsError(f"Destination collision for {merged_stem}")

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
                "source_dataset_id": DATASET_ID,
                "source_species": species,
                "source_dir": str(image_path.parent.resolve()),
            }
            dst_meta.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

            imported += 1
            imported_by_species[species] += 1
            batch_type = "missing_labels" if "missing_labels" in batch_dir.parts else "regular"
            imported_by_batch_type[batch_type] += 1
            breed_keys = {
                str(obj.get("subspecies") or species)
                for obj in objects
            }
            for breed_key in breed_keys:
                imported_by_breed[breed_key] = imported_by_breed.get(breed_key, 0) + 1
            if len(objects) > 1:
                multi_object_images += 1
            if len({str(obj.get("source_label_name")) for obj in objects}) > 1:
                multi_label_images += 1
            if csv_rows:
                csv_backed += 1
            else:
                folder_primary_only += 1

    return {
        "removed_existing_triplet_files": removed,
        "imported_total": imported,
        "imported_by_species": imported_by_species,
        "imported_by_breed": dict(sorted(imported_by_breed.items())),
        "imported_by_batch_type": imported_by_batch_type,
        "csv_backed_images": csv_backed,
        "folder_primary_only_images": folder_primary_only,
        "multi_object_images": multi_object_images,
        "multi_label_images": multi_label_images,
        "filtered_out_of_taxonomy_objects": filtered_out_of_taxonomy_objects,
        "excluded_total": len(exclusions),
        "excluded_examples": exclusions[:200],
        "integrity": global_integrity_summary(),
    }


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = integrate()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


SPECIES_CLASS_IDS = load_species_class_ids()


if __name__ == "__main__":
    main()
