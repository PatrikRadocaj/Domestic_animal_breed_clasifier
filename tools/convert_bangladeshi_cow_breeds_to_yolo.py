"""Convert Bangladeshi cow breed folder+label export into a multispecies YOLO source."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import yaml


SRC_ROOT = Path(
    r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\raw\non_yolo_sources\bangladeshi-cow-breeds-yd84s35d73\A comprehensive image dataset of Bangladeshi cow b\Cow Breed\Cow Breed"
)
CSV_PATH = SRC_ROOT / "labels_my-project-name_2026-06-06-01-40-02.csv"
OUT_ROOT = Path(
    r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\raw\yolo_sources\bangladeshi-cow-breeds-yd84s35d73-yolo-multispecies"
)
CLASS_NAMES = ["holstain_friesian", "jersey", "local_deshi", "sahiwal", "pigeon"]
LABEL_TO_CLASS = {name: name for name in CLASS_NAMES}
SPECIES_BY_CLASS = {
    "holstain_friesian": "cattle",
    "jersey": "cattle",
    "local_deshi": "cattle",
    "sahiwal": "cattle",
    "pigeon": "pigeon",
}
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def slugify(value: str) -> str:
    value = value.strip().lower()
    return value.replace("-", "_").replace(" ", "_")


def split_for_name(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def ensure_dirs() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "valid", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "metadata" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml() -> None:
    payload = {
        "path": str(OUT_ROOT.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    (OUT_ROOT / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def resolve_class_id(label_name: str) -> int:
    normalized = slugify(label_name)
    class_name = LABEL_TO_CLASS.get(normalized)
    if class_name is None:
        raise KeyError(f"Unmapped label: {label_name!r}")
    return CLASS_NAMES.index(class_name)


def class_to_species(class_name: str) -> str:
    return SPECIES_BY_CLASS[class_name]


def build_subspecies(label_name: str, class_name: str) -> str | None:
    normalized = slugify(label_name)
    species_name = class_to_species(class_name)
    if species_name == class_name:
        return None
    return normalized


def build_image_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for breed_dir in sorted([p for p in SRC_ROOT.iterdir() if p.is_dir()]):
        for image_path in breed_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTS:
                index[image_path.name] = image_path
    return index


def convert() -> dict[str, object]:
    image_index = build_image_index()
    boxes_by_image: dict[str, list[str]] = {}
    metadata_by_image: dict[str, list[dict[str, object]]] = {}
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                x = float(row["bbox_x"])
                y = float(row["bbox_y"])
                w = float(row["bbox_width"])
                h = float(row["bbox_height"])
                img_w = float(row["image_width"])
                img_h = float(row["image_height"])
            except (KeyError, TypeError, ValueError):
                continue
            if min(w, h, img_w, img_h) <= 0:
                continue
            image_name = str(row.get("image_name", "")).strip()
            if not image_name:
                continue
            class_id = resolve_class_id(str(row.get("label_name", "")))
            class_name = CLASS_NAMES[class_id]
            species_name = class_to_species(class_name)
            xc = (x + (w / 2.0)) / img_w
            yc = (y + (h / 2.0)) / img_h
            wn = w / img_w
            hn = h / img_h
            boxes_by_image.setdefault(image_name, []).append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            metadata_by_image.setdefault(image_name, []).append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "species_name": species_name,
                    "subspecies": build_subspecies(str(row.get("label_name", "")), class_name),
                    "source_label_name": str(row.get("label_name", "")),
                    "bbox_xywhn": [round(xc, 6), round(yc, 6), round(wn, 6), round(hn, 6)],
                }
            )

    written = 0
    skipped = 0
    for image_name, label_lines in sorted(boxes_by_image.items()):
        image_path = image_index.get(image_name)
        if image_path is None:
            skipped += 1
            continue
        split = split_for_name(Path(image_name).stem)
        dst_image = OUT_ROOT / "images" / split / image_path.name
        dst_label = OUT_ROOT / "labels" / split / f"{Path(image_name).stem}.txt"
        dst_meta = OUT_ROOT / "metadata" / split / f"{Path(image_name).stem}.json"
        shutil.copy2(image_path, dst_image)
        dst_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        dst_meta.write_text(
            json.dumps(
                {
                    "image_name": image_path.name,
                    "folder_name": image_path.parent.name,
                    "split": split,
                    "image_path": str(dst_image.resolve()),
                    "label_path": str(dst_label.resolve()),
                    "objects": metadata_by_image[image_name],
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        written += 1

    return {"written": written, "skipped": skipped}


def main() -> None:
    ensure_dirs()
    conversion = convert()
    write_data_yaml()
    split_counts = {split: len(list((OUT_ROOT / "labels" / split).glob("*.txt"))) for split in ("train", "valid", "test")}
    report = {
        "output_dir": str(OUT_ROOT),
        "class_names": CLASS_NAMES,
        "total_written": conversion["written"],
        "total_skipped": conversion["skipped"],
        "split_counts": split_counts,
    }
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
