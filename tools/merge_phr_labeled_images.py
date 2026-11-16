"""Merge selected YOLO sources into one combined species-level PHR source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import yaml


REPO_ROOT = Path(r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje")
SOURCES = [
    REPO_ROOT / "data" / "raw" / "yolo_sources" / "wikimedia-duck-breeds-yolo-multispecies",
    REPO_ROOT / "data" / "raw" / "yolo_sources" / "wikimedia-goose-breeds-yolo-multispecies",
    REPO_ROOT / "data" / "raw" / "yolo_sources" / "horse-breeds-olgabelitskaya-yolo",
    REPO_ROOT / "data" / "raw" / "yolo_sources" / "cattle-breeds-anandkumarsahu09-yolo",
    REPO_ROOT / "data" / "raw" / "yolo_sources" / "bangladeshi-cow-breeds-yd84s35d73-yolo-multispecies",
]
OUT_ROOT = REPO_ROOT / "data" / "raw" / "yolo_sources" / "phr-labeled-images"
MERGED_CLASS_NAMES = ["duck", "goose", "chicken", "cattle", "cat", "dog", "pigeon", "turkey", "alpaca", "horse"]


def ensure_dirs() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "valid", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "metadata" / split).mkdir(parents=True, exist_ok=True)


def split_key_for_yaml(split: str) -> str:
    return "val" if split == "valid" else split


def load_data_yaml(root: Path) -> dict:
    for candidate in (root / "data.yaml", root / "data.yaml.disabled"):
        if candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing data.yaml for {root}")


def normalize_names(raw_names) -> list[str]:
    if isinstance(raw_names, list):
        return [str(x) for x in raw_names]
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names, key=lambda x: int(x))]
    raise TypeError(f"Unsupported names payload: {type(raw_names)!r}")


def resolve_source_split_dirs(root: Path, data: dict, split: str) -> tuple[Path, Path, Path]:
    split_key = split_key_for_yaml(split)
    image_rel = Path(str(data[split_key]))
    image_dir = root / image_rel
    if len(image_rel.parts) >= 2 and image_rel.parts[0] == "images":
        label_dir = root / "labels" / image_rel.parts[1]
        metadata_dir = root / "metadata" / image_rel.parts[1]
    else:
        label_dir = root / image_rel.parent / "labels"
        metadata_dir = root / image_rel.parent / "metadata"
    return image_dir, label_dir, metadata_dir


def remap_metadata(meta_path: Path, split: str, image_path: Path, label_path: Path, source_dataset: str) -> tuple[dict, list[str]]:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    objects = []
    label_lines: list[str] = []
    for obj in payload.get("objects", []):
        species_name = str(obj.get("species_name") or obj.get("class_name"))
        merged_class_id = MERGED_CLASS_NAMES.index(species_name)
        bbox = obj.get("bbox_xywhn", [])
        if len(bbox) != 4:
            continue
        obj = dict(obj)
        obj["class_id"] = merged_class_id
        obj["class_name"] = species_name
        obj["species_name"] = species_name
        obj["source_class_name"] = str(obj.get("source_class_name") or obj.get("source_label_name") or obj.get("subspecies") or species_name)
        objects.append(obj)
        label_lines.append(f"{merged_class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
    return ({
        **payload,
        "split": split,
        "image_path": str(image_path.resolve()),
        "label_path": str(label_path.resolve()),
        "source_dataset_id": source_dataset,
        "objects": objects,
    }, label_lines)




def build_merged_stem(source_dataset: str, stem: str) -> str:
    merged = f"{source_dataset}__{stem}"
    if len(merged) <= 120:
        return merged
    digest = hashlib.sha1(merged.encode("utf-8")).hexdigest()[:16]
    short_stem = stem[:70].rstrip('_ ')
    return f"{source_dataset}__{short_stem}__{digest}"

def copy_source(root: Path) -> dict[str, int]:
    data = load_data_yaml(root)
    _ = normalize_names(data["names"])
    counts = {"images": 0, "labels": 0, "metadata": 0}
    source_dataset = root.name
    for split in ("train", "valid", "test"):
        image_dir, label_dir, metadata_dir = resolve_source_split_dirs(root, data, split)
        for label_path in label_dir.glob("*.txt"):
            stem = label_path.stem
            image_candidates = list(image_dir.glob(f"{stem}.*"))
            meta_path = metadata_dir / f"{stem}.json"
            if len(image_candidates) != 1 or not meta_path.exists():
                continue
            image_src = image_candidates[0]
            merged_stem = build_merged_stem(source_dataset, stem)
            image_dst = OUT_ROOT / "images" / split / f"{merged_stem}{image_src.suffix.lower()}"
            label_dst = OUT_ROOT / "labels" / split / f"{merged_stem}.txt"
            meta_dst = OUT_ROOT / "metadata" / split / f"{merged_stem}.json"
            if image_dst.exists() or label_dst.exists() or meta_dst.exists():
                raise FileExistsError(f"Collision while merging: {merged_stem}")
            shutil.copy2(image_src, image_dst)
            meta_payload, remapped_label_lines = remap_metadata(meta_path, split, image_dst, label_dst, source_dataset)
            label_dst.write_text("\n".join(remapped_label_lines) + "\n", encoding="utf-8")
            meta_dst.write_text(
                json.dumps(meta_payload, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            counts["images"] += 1
            counts["labels"] += 1
            counts["metadata"] += 1
    return counts


def write_data_yaml() -> None:
    payload = {
        "path": str(OUT_ROOT.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(MERGED_CLASS_NAMES),
        "names": MERGED_CLASS_NAMES,
    }
    (OUT_ROOT / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    per_source = {}
    for root in SOURCES:
        per_source[root.name] = copy_source(root)
    write_data_yaml()
    split_counts = {
        split: len(list((OUT_ROOT / "labels" / split).glob("*.txt"))) for split in ("train", "valid", "test")
    }
    report = {
        "output_dir": str(OUT_ROOT),
        "class_names": MERGED_CLASS_NAMES,
        "split_counts": split_counts,
        "sources": per_source,
    }
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
