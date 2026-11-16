"""Utilities for indexing heterogeneous YOLO-exported source folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class YoloSourceDataset:
    dataset_id: str
    root_dir: Path
    data_yaml_path: Path
    names: list[str]
    split_image_dirs: dict[str, Path]
    split_label_dirs: dict[str, Path]
    label_file_count: int


def parse_names_field(payload: dict[str, Any]) -> list[str]:
    names_raw = payload.get("names", [])
    if isinstance(names_raw, list):
        return [str(x) for x in names_raw]
    if isinstance(names_raw, dict):
        return [str(names_raw[k]) for k in sorted(names_raw, key=lambda x: int(x) if str(x).isdigit() else str(x))]
    return []


def resolve_split_dir(base: Path, data_path_value: str, split_value: str) -> Path:
    split_path = Path(str(split_value))
    if split_path.is_absolute():
        return split_path

    data_base_raw = Path(str(data_path_value)) if str(data_path_value).strip() else Path(".")
    candidate_bases: list[Path] = []
    if data_base_raw.is_absolute():
        candidate_bases.append(data_base_raw.resolve())
    else:
        candidate_bases.extend(
            [
                (base / data_base_raw).resolve(),
                (Path.cwd() / data_base_raw).resolve(),
                base.resolve(),
            ]
        )

    # Also try stripping leading "../" tokens used by some Roboflow exports.
    split_parts = list(split_path.parts)
    while split_parts and split_parts[0] == "..":
        split_parts.pop(0)
    stripped_split = Path(*split_parts) if split_parts else Path(".")

    fallback = (candidate_bases[0] / split_path).resolve()
    for data_base in candidate_bases:
        c1 = (data_base / split_path).resolve()
        if c1.exists():
            return c1
        c2 = (data_base / stripped_split).resolve()
        if c2.exists():
            return c2
    return fallback


def count_label_files(label_dir: Path) -> int:
    if not label_dir.exists():
        return 0
    return sum(1 for p in label_dir.rglob("*.txt") if p.is_file())


def infer_label_dir_from_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    lowered = [p.lower() for p in parts]
    if "images" in lowered:
        idx = lowered.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return Path(str(image_dir).replace("images", "labels"))


def discover_yolo_sources(yolo_sources_root: str | Path) -> list[YoloSourceDataset]:
    root = Path(yolo_sources_root)
    if not root.exists():
        return []

    datasets: list[YoloSourceDataset] = []
    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        yaml_candidates = sorted(dataset_dir.rglob("data.yaml"))
        if not yaml_candidates:
            yaml_candidates = sorted(dataset_dir.rglob("data.yml"))
        best: YoloSourceDataset | None = None
        for ypath in yaml_candidates:
            try:
                payload = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
            except Exception:
                continue

            names = parse_names_field(payload)
            if not names:
                continue

            split_image_dirs: dict[str, Path] = {}
            split_label_dirs: dict[str, Path] = {}
            label_count = 0
            split_aliases = {
                "train": ["train"],
                "val": ["val", "valid"],
                "test": ["test"],
            }
            for split_key, aliases in split_aliases.items():
                split_value = None
                for alias in aliases:
                    if alias in payload and payload.get(alias):
                        split_value = payload.get(alias)
                        break
                if not split_value:
                    continue
                image_dir = resolve_split_dir(ypath.parent, str(payload.get("path", "")), str(split_value))
                split_image_dirs[split_key] = image_dir
                label_dir = infer_label_dir_from_image_dir(image_dir)
                split_label_dirs[split_key] = label_dir
                label_count += count_label_files(label_dir)

            candidate = YoloSourceDataset(
                dataset_id=dataset_dir.name,
                root_dir=dataset_dir,
                data_yaml_path=ypath,
                names=names,
                split_image_dirs=split_image_dirs,
                split_label_dirs=split_label_dirs,
                label_file_count=label_count,
            )
            if best is None or candidate.label_file_count > best.label_file_count:
                best = candidate

        if best is not None:
            datasets.append(best)
    return datasets


def resolve_image_for_label(label_path: Path, image_dir: Path) -> Path | None:
    stem = label_path.stem
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    return None
