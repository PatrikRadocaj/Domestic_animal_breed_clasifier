"""Convert the labeled Wikimedia duck-breed folders into a multispecies YOLO source."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import yaml


SRC_ROOT = Path(
    r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\raw\non_yolo_sources\wikimedia\duck\by_category"
)
LABELED_FOLDERS = [
    "American_Pekin",
    "Aylesbury_duck",
    "Black_East_Indian",
    "Call_Duck",
    "Campbell_Ducks",
    "Cayuga_duck",
    "Crested_Duck_domestic_breed",
    "Ducclair_duck",
    "Indian_Runner_Ducks",
    "Muscovy_ducks_domesticated",
    "Rouen_duck",
    "Saxony_duck",
    "Silver_Appleyard",
    "Swedish_Duck",
    "Welsh_Harlequin",
]
OUT_ROOT = Path(
    r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\raw\yolo_sources\wikimedia-duck-breeds-yolo-multispecies"
)
CLASS_NAMES = ["duck", "chicken", "goose", "cattle", "cat", "dog", "pigeon", "turkey"]
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_TO_CLASS = {
    "duck": "duck",
    "unknown duck": "duck",
    "duck unknown": "duck",
    "duck unknown ": "duck",
    "american pekin": "duck",
    "aylesbury duck": "duck",
    "black east indian": "duck",
    "call duck": "duck",
    "campbell ducks": "duck",
    "cayuga duck": "duck",
    "crested duck": "duck",
    "crested duck domestic": "duck",
    "ducclair duck": "duck",
    "indian runner": "duck",
    "muscovy duck": "duck",
    "rouen duck": "duck",
    "saxony duck": "duck",
    "silver appleyard": "duck",
    "chicken": "chicken",
    "goose": "goose",
    "cattle": "cattle",
    "cat": "cat",
    "dog": "dog",
    "pigeon": "pigeon",
    "turkey": "turkey",
}


def normalize_label(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def split_for_name(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def safe_stem(folder_name: str, image_name: str) -> str:
    return f"{folder_name}__{Path(image_name).stem}"


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
    normalized = normalize_label(label_name)
    target_name = LABEL_TO_CLASS.get(normalized, "duck")
    return CLASS_NAMES.index(target_name)


def infer_subspecies(label_name: str, folder_name: str, default_species: str) -> str:
    normalized = normalize_label(label_name)
    target_name = LABEL_TO_CLASS.get(normalized, default_species)
    if target_name != default_species:
        return "unknown"
    if normalized in {"duck", "unknown duck", "duck unknown"}:
        return "unknown"
    return normalized.replace(" ", "_") if normalized else folder_name.lower().replace(" ", "_")


def copy_sample(
    image_path: Path,
    label_lines: list[str],
    metadata_objects: list[dict[str, object]],
    folder_name: str,
    image_name: str,
    split: str,
) -> bool:
    if not image_path.exists() or image_path.suffix.lower() not in VALID_IMAGE_EXTS:
        return False
    label_lines = [line for line in label_lines if line.strip()]
    if not label_lines or not metadata_objects or len(label_lines) != len(metadata_objects):
        return False
    stem = safe_stem(folder_name, image_name)
    dst_image = OUT_ROOT / "images" / split / f"{stem}{image_path.suffix.lower()}"
    dst_label = OUT_ROOT / "labels" / split / f"{stem}.txt"
    dst_meta = OUT_ROOT / "metadata" / split / f"{stem}.json"
    shutil.copy2(image_path, dst_image)
    dst_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    dst_meta.write_text(
        json.dumps(
            {
                "image_name": image_name,
                "folder_name": folder_name,
                "split": split,
                "image_path": str(dst_image.resolve()),
                "label_path": str(dst_label.resolve()),
                "objects": metadata_objects,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return True


def convert_zip_only_folder(folder: Path) -> tuple[int, int]:
    zip_files = sorted(folder.glob("*.zip"))
    if not zip_files:
        return 0, 0
    image_index = {p.name: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS}
    written = 0
    skipped = 0
    with zipfile.ZipFile(zip_files[0]) as zf:
        for member in zf.namelist():
            if not member.endswith(".txt"):
                continue
            image_path = next((p for p in image_index.values() if p.stem == Path(member).stem), None)
            if image_path is None:
                skipped += 1
                continue
            label_lines = []
            metadata_objects = []
            for raw in zf.read(member).decode("utf-8").splitlines():
                parts = raw.strip().split()
                if len(parts) == 5:
                    label_lines.append("0 " + " ".join(parts[1:]))
                    metadata_objects.append(
                        {
                            "class_id": 0,
                            "class_name": "duck",
                            "species_name": "duck",
                            "subspecies": folder.name.lower(),
                            "source_label_name": folder.name,
                            "bbox_xywhn": [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                        }
                    )
            split = split_for_name(safe_stem(folder.name, image_path.name))
            if copy_sample(image_path, label_lines, metadata_objects, folder.name, image_path.name, split):
                written += 1
            else:
                skipped += 1
    return written, skipped


def convert_csv_folder(folder: Path) -> tuple[int, int]:
    csv_path = sorted(folder.glob("*.csv"))[0]
    boxes_by_image: dict[str, list[str]] = {}
    metadata_by_image: dict[str, list[dict[str, object]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
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
            class_id = resolve_class_id(row.get("label_name", ""))
            class_name = CLASS_NAMES[class_id]
            xc = (x + (w / 2.0)) / img_w
            yc = (y + (h / 2.0)) / img_h
            wn = w / img_w
            hn = h / img_h
            image_name = row["image_name"]
            boxes_by_image.setdefault(image_name, []).append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            metadata_by_image.setdefault(image_name, []).append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "species_name": class_name,
                    "subspecies": infer_subspecies(str(row.get("label_name", "")), folder.name, class_name),
                    "source_label_name": str(row.get("label_name", "")),
                    "bbox_xywhn": [round(xc, 6), round(yc, 6), round(wn, 6), round(hn, 6)],
                }
            )
    written = 0
    skipped = 0
    for image_name, label_lines in boxes_by_image.items():
        image_path = folder / image_name
        split = split_for_name(safe_stem(folder.name, image_name))
        if copy_sample(image_path, label_lines, metadata_by_image.get(image_name, []), folder.name, image_name, split):
            written += 1
        else:
            skipped += 1
    return written, skipped


def main() -> None:
    ensure_dirs()
    total_written = 0
    total_skipped = 0
    per_folder = []
    for folder_name in LABELED_FOLDERS:
        folder = SRC_ROOT / folder_name
        has_csv = any(folder.glob("*.csv"))
        written, skipped = convert_csv_folder(folder) if has_csv else convert_zip_only_folder(folder)
        total_written += written
        total_skipped += skipped
        per_folder.append({"folder": folder_name, "mode": "csv" if has_csv else "zip", "written": written, "skipped": skipped})
    write_data_yaml()
    split_counts = {split: len(list((OUT_ROOT / "labels" / split).glob("*.txt"))) for split in ("train", "valid", "test")}
    report = {
        "output_dir": str(OUT_ROOT),
        "class_names": CLASS_NAMES,
        "total_written": total_written,
        "total_skipped": total_skipped,
        "split_counts": split_counts,
        "folders": per_folder,
    }
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
