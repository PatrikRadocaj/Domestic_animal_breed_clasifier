"""Convert rabbit breed Roboflow folder+label export into a YOLO source with breed metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import yaml


SRC_ROOT = Path(
    r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\raw\non_yolo_sources\rabbit-breed-classification-roboflow"
)
TRAIN_ROOT = SRC_ROOT / "train"
ZIP_PATH = TRAIN_ROOT / "labels_my-project-name_2026-06-05-06-48-39.zip"
OUT_ROOT = Path(
    r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje\data\raw\yolo_sources\rabbit-breeds-roboflow-yolo-multispecies"
)
CLASS_NAMES = ["rabbit"]
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def slugify(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


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


def rename_source_files() -> None:
    for breed_dir in sorted([p for p in TRAIN_ROOT.iterdir() if p.is_dir()]):
        breed_slug = slugify(breed_dir.name)
        images = sorted([p for p in breed_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS])
        rename_plan: list[tuple[Path, Path]] = []
        for idx, src in enumerate(images, start=1):
            dst_name = f"{breed_slug}_{idx:03d}{src.suffix.lower()}"
            dst = breed_dir / dst_name
            rename_plan.append((src, dst))
        # two-phase rename avoids collisions if rerun
        temp_plan: list[tuple[Path, Path]] = []
        for src, dst in rename_plan:
            if src == dst:
                continue
            tmp = src.with_name(f"__tmp__{src.name}")
            if tmp.exists():
                tmp.unlink()
            src.rename(tmp)
            temp_plan.append((tmp, dst))
        for tmp, dst in temp_plan:
            if dst.exists():
                dst.unlink()
            tmp.rename(dst)


def rewrite_label_zip() -> Path:
    rebuilt_zip = TRAIN_ROOT / "labels_integrated_rabbit_breeds.zip"
    breed_entries: dict[str, list[tuple[str, bytes]]] = {}
    breed_aliases: dict[str, list[str]] = {}
    for p in TRAIN_ROOT.iterdir():
        if not p.is_dir():
            continue
        slug = slugify(p.name)
        breed_aliases[slug] = [slug, slug.replace("_", "-")]
    with zipfile.ZipFile(ZIP_PATH, "r") as zin:
        for info in zin.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".txt"):
                continue
            stem = Path(info.filename).stem.lower()
            matched = None
            for breed_slug, aliases in breed_aliases.items():
                if any(stem.startswith(alias) for alias in aliases):
                    matched = breed_slug
                    break
            if matched is None:
                continue
            breed_entries.setdefault(matched, []).append((info.filename, zin.read(info.filename)))
    with zipfile.ZipFile(rebuilt_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for breed_dir in sorted([p for p in TRAIN_ROOT.iterdir() if p.is_dir()]):
            breed_slug = slugify(breed_dir.name)
            images = sorted([p for p in breed_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS])
            entries = sorted(breed_entries.get(breed_slug, []), key=lambda item: item[0].lower())
            usable = min(len(images), len(entries))
            for image_path, (_, payload) in zip(images[:usable], entries[:usable]):
                zout.writestr(f"{image_path.stem}.txt", payload)
    return rebuilt_zip


def convert(label_zip: Path) -> dict[str, object]:
    written = 0
    with zipfile.ZipFile(label_zip, "r") as zf:
        label_entries = {Path(info.filename).stem: info for info in zf.infolist() if not info.is_dir() and info.filename.lower().endswith(".txt")}
        for breed_dir in sorted([p for p in TRAIN_ROOT.iterdir() if p.is_dir()]):
            breed_slug = slugify(breed_dir.name)
            for image_path in sorted([p for p in breed_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]):
                stem = image_path.stem
                entry = label_entries.get(stem)
                if entry is None:
                    continue
                split = split_for_name(stem)
                dst_image = OUT_ROOT / "images" / split / image_path.name
                dst_label = OUT_ROOT / "labels" / split / f"{stem}.txt"
                dst_meta = OUT_ROOT / "metadata" / split / f"{stem}.json"
                shutil.copy2(image_path, dst_image)
                label_text = zf.read(entry).decode("utf-8").strip()
                if label_text:
                    label_lines = []
                    objects = []
                    for line in label_text.splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        _, xc, yc, w, h = parts
                        label_lines.append(f"0 {xc} {yc} {w} {h}")
                        objects.append(
                            {
                                "class_id": 0,
                                "class_name": "rabbit",
                                "species_name": "rabbit",
                                "subspecies": breed_slug,
                                "source_label_name": breed_dir.name,
                                "bbox_xywhn": [float(xc), float(yc), float(w), float(h)],
                            }
                        )
                    if not label_lines:
                        continue
                    dst_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                    dst_meta.write_text(
                        json.dumps(
                            {
                                "image_name": image_path.name,
                                "folder_name": breed_dir.name,
                                "split": split,
                                "image_path": str(dst_image.resolve()),
                                "label_path": str(dst_label.resolve()),
                                "objects": objects,
                            },
                            ensure_ascii=True,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    written += 1
    return {"written": written}


def main() -> None:
    ensure_dirs()
    rename_source_files()
    rebuilt_zip = rewrite_label_zip()
    report = convert(rebuilt_zip)
    write_data_yaml()
    split_counts = {split: len(list((OUT_ROOT / "labels" / split).glob("*.txt"))) for split in ("train", "valid", "test")}
    payload = {
        "output_dir": str(OUT_ROOT),
        "label_zip": str(rebuilt_zip),
        "class_names": CLASS_NAMES,
        "total_written": report["written"],
        "split_counts": split_counts,
    }
    print(yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
