"""On-the-fly hybrid YOLO dataset generation for long-tail multi-species training.

This module builds a deterministic sample plan plus a PyTorch dataset that renders
synthetic balanced composites, original real images, and strongly augmented
single-species images at access time.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import ceil, sqrt
from pathlib import Path
import json
import math
import os
import random
import shutil
import sys
from typing import Any

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import DecompressionBombError
import torch
from torch.utils.data import Dataset
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances

REPO_ROOT = Path(__file__).resolve().parents[3]


def _round_to_stride(value: int, stride: int = 32) -> int:
    return int(ceil(max(value, stride) / stride) * stride)


def _sha_like_int(parts: list[int | str]) -> int:
    text = "|".join(str(p) for p in parts)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))


def _normalize_box_xywhn_to_xyxy(box_xywhn: list[float], width: int, height: int) -> list[float]:
    cx, cy, bw, bh = box_xywhn
    x0 = (cx - bw / 2.0) * width
    y0 = (cy - bh / 2.0) * height
    x1 = (cx + bw / 2.0) * width
    y1 = (cy + bh / 2.0) * height
    return [float(x0), float(y0), float(x1), float(y1)]


def _clip_box_xyxy(box: list[float], width: int, height: int) -> list[float] | None:
    x0, y0, x1, y1 = box
    x0 = min(max(0.0, x0), float(width))
    y0 = min(max(0.0, y0), float(height))
    x1 = min(max(0.0, x1), float(width))
    y1 = min(max(0.0, y1), float(height))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _resize_boxes_xyxy(boxes: list[list[float]], scale_x: float, scale_y: float, off_x: float = 0.0, off_y: float = 0.0) -> list[list[float]]:
    out: list[list[float]] = []
    for x0, y0, x1, y1 in boxes:
        out.append([x0 * scale_x + off_x, y0 * scale_y + off_y, x1 * scale_x + off_x, y1 * scale_y + off_y])
    return out


def _limit_boxes_by_area(
    boxes: list[list[float]],
    class_ids: list[int],
    subspecies: list[str],
    limit: int,
) -> tuple[list[list[float]], list[int], list[str]]:
    if limit <= 0 or len(boxes) <= limit:
        return boxes, class_ids, subspecies
    order = sorted(
        range(len(boxes)),
        key=lambda idx: max(0.0, boxes[idx][2] - boxes[idx][0]) * max(0.0, boxes[idx][3] - boxes[idx][1]),
        reverse=True,
    )[:limit]
    order.sort()
    return [boxes[i] for i in order], [class_ids[i] for i in order], [subspecies[i] for i in order]


def _sanitize_labeled_boxes(
    boxes: list[list[float]],
    class_ids: list[int],
    subspecies: list[str],
    width: int,
    height: int,
    min_size: float = 1.0,
) -> tuple[list[list[float]], list[int], list[str]]:
    clean_boxes: list[list[float]] = []
    clean_class_ids: list[int] = []
    clean_subspecies: list[str] = []
    for box, class_id, subspecies_name in zip(boxes, class_ids, subspecies, strict=True):
        clipped = _clip_box_xyxy(list(map(float, box)), width, height)
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        if (x1 - x0) < min_size or (y1 - y0) < min_size:
            continue
        clean_boxes.append([x0, y0, x1, y1])
        clean_class_ids.append(int(class_id))
        clean_subspecies.append(str(subspecies_name))
    return clean_boxes, clean_class_ids, clean_subspecies


def _crop_image_to_box_union(
    image: np.ndarray,
    boxes: list[list[float]],
    margin_px: int,
) -> tuple[np.ndarray, list[list[float]]]:
    if not boxes:
        return image, boxes
    height, width = image.shape[:2]
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    crop_x0 = max(0, int(math.floor(x0 - margin_px)))
    crop_y0 = max(0, int(math.floor(y0 - margin_px)))
    crop_x1 = min(width, int(math.ceil(x1 + margin_px)))
    crop_y1 = min(height, int(math.ceil(y1 + margin_px)))
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return image, boxes
    cropped = image[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    cropped_boxes = [[bx0 - crop_x0, by0 - crop_y0, bx1 - crop_x0, by1 - crop_y0] for bx0, by0, bx1, by1 in boxes]
    return cropped, cropped_boxes


def _grid_track_sizes(tiles: list[dict[str, Any]], rows: int, cols: int) -> tuple[list[int], list[int]]:
    col_widths = [0] * cols
    row_heights = [0] * rows
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        if row >= rows:
            break
        col_widths[col] = max(col_widths[col], int(tile["width"]))
        row_heights[row] = max(row_heights[row], int(tile["height"]))
    return col_widths, row_heights


def _resolve_background_paths(background_dir: str | Path) -> list[str]:
    path = Path(background_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
    return sorted(str(p) for p in path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts)


def _xyxy_to_xywhn(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / float(width)
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / float(height)
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / float(width)
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / float(height)
    return out


def _species_slug(text: str) -> str:
    text = str(text).strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "unknown"


@dataclass(frozen=True)
class HybridDatasetConfig:
    seed: int = 42
    synthetic_ratio: float = 0.40
    real_ratio: float = 0.40
    augmented_ratio: float = 0.20
    min_outer_padding: int = 1
    min_inter_image_padding: int = 1
    min_canvas_size: int = 1024
    max_canvas_size: int = 4096
    absolute_max_canvas_size: int = 8192
    allow_canvas_expand_beyond_max: bool = True
    min_cell_size: int = 260
    max_cell_size: int = 720
    placement_scale_min: float = 0.88
    placement_scale_max: float = 1.20
    grid_jitter: float = 0.06
    synthetic_crop_to_objects: bool = True
    synthetic_crop_margin_px: int = 10
    synthetic_min_box_size_px: float = 15.0
    synthetic_injection_gap_min: int = 5
    synthetic_injection_gap_max: int = 10
    injection_count_min: int = 1
    injection_count_max: int = 3
    extra_synthetic_budget_fraction: float = 0.05
    background_texture_probability_min: float = 0.05
    background_texture_probability_max: float = 0.15
    synthetic_backgrounds_dir: str = "data/raw/synthetic_backgrounds"
    decoded_image_cache_size: int = 256
    resized_image_cache_size: int = 256
    debug_subset_count: int = 12
    test_subset_only: bool = False
    test_subset_train_count: int = 32
    test_subset_val_count: int = 16
    allow_scale_up: bool = True
    strong_rotation_limit: int = 30
    mild_rotation_limit: int = 10
    random_crop_scale_min: float = 0.70
    random_crop_scale_max: float = 1.00
    brightness_contrast_limit: float = 0.22
    gaussian_noise_std_fraction: float = 0.03
    val_mode: str = "real_only"
    train_metadata_write: bool = True
    placement_retry_count: int = 12
    greedy_min_scale: float = 0.45
    max_train_objects_per_image: int = 8
    max_eval_objects_per_image: int = 16

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "HybridDatasetConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        values = {k: payload[k] for k in payload if k in known}
        safe_workers = int(os.environ.get("HYBRID_YOLO_WORKERS", "0") or "0")
        # Keep synthetic samples near the final training resolution. Larger
        # intermediate canvases caused Windows DataLoader shared-memory failures
        # with many workers.
        values["max_canvas_size"] = min(int(values.get("max_canvas_size", cls.max_canvas_size)), 1536)
        values["absolute_max_canvas_size"] = min(
            int(values.get("absolute_max_canvas_size", cls.absolute_max_canvas_size)),
            2048,
        )
        values["allow_canvas_expand_beyond_max"] = False
        values["max_cell_size"] = min(int(values.get("max_cell_size", cls.max_cell_size)), 560)
        values["max_train_objects_per_image"] = min(
            max(1, int(values.get("max_train_objects_per_image", cls.max_train_objects_per_image))),
            8,
        )
        values["max_eval_objects_per_image"] = min(
            max(1, int(values.get("max_eval_objects_per_image", cls.max_eval_objects_per_image))),
            16,
        )
        if safe_workers > 0:
            values["decoded_image_cache_size"] = min(int(values.get("decoded_image_cache_size", 0)), 8)
            values["resized_image_cache_size"] = min(int(values.get("resized_image_cache_size", 0)), 8)
        return cls(**values)


@dataclass(frozen=True)
class SourceObject:
    class_id: int
    species_id: str
    subspecies: str
    source_class_name: str
    bbox_xywhn: list[float]


@dataclass(frozen=True)
class SourceImageRecord:
    record_id: str
    dataset_id: str
    split: str
    image_path: str
    width: int
    height: int
    source_species_ids: list[str]
    objects: list[SourceObject]
    pure_species_id: str | None = None


@dataclass(frozen=True)
class SamplePlan:
    sample_id: str
    split: str
    sample_kind: str
    synthetic_index: int = -1
    record_id: str | None = None
    species_id: str | None = None
    variant_seed: int = 0


@dataclass
class RenderedSample:
    image: np.ndarray
    class_ids: list[int]
    species_ids: list[str]
    subspecies: list[str]
    boxes_xyxy: list[list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)


class _LruArrayCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max(0, int(max_items))
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> np.ndarray | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value.copy()

    def put(self, key: str, value: np.ndarray) -> None:
        if self.max_items <= 0:
            return
        self._items[key] = value.copy()
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_box_transform(
    cfg: HybridDatasetConfig,
    mode: str,
    image_shape: tuple[int, int],
    *,
    include_noise: bool = True,
) -> A.Compose:
    h, w = image_shape
    bbox_params = A.BboxParams(
        format="pascal_voc",
        label_fields=["class_ids", "subspecies"],
        min_visibility=0.20,
        clip=True,
        filter_invalid_bboxes=True,
    )
    if mode == "single_strong":
        strong_ops: list[Any] = [
            A.HorizontalFlip(p=0.50),
            A.VerticalFlip(p=0.50),
            A.Rotate(limit=cfg.strong_rotation_limit, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.85),
            A.RandomResizedCrop(
                size=(h, w),
                scale=(cfg.random_crop_scale_min, cfg.random_crop_scale_max),
                ratio=(0.75, 1.25),
                p=0.85,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_contrast_limit,
                contrast_limit=cfg.brightness_contrast_limit,
                p=0.80,
            ),
        ]
        return A.Compose(
            strong_ops,
            bbox_params=bbox_params,
        )
    return A.Compose(
        [
            A.HorizontalFlip(p=0.40),
            A.VerticalFlip(p=0.15),
            A.Rotate(limit=cfg.mild_rotation_limit, border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.50),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_contrast_limit * 0.55,
                contrast_limit=cfg.brightness_contrast_limit * 0.55,
                p=0.45,
            ),
        ],
        bbox_params=bbox_params,
    )


def _choose_weighted_species(rng: random.Random, species_ids: list[str], counts: dict[str, int]) -> str:
    max_count = max(counts.values())
    weights = []
    for species_id in species_ids:
        n = max(1, counts.get(species_id, 1))
        weights.append(float(max_count) / float(n))
    return rng.choices(species_ids, weights=weights, k=1)[0]


def build_hybrid_sample_plans(
    class_names: list[str],
    pure_records_by_species: dict[str, list[SourceImageRecord]],
    real_records: list[SourceImageRecord],
    split: str,
    cfg: HybridDatasetConfig,
) -> list[SamplePlan]:
    if split != "train":
        subset_limit = None
        if cfg.test_subset_only:
            subset_limit = int(cfg.test_subset_val_count if split == "val" else cfg.test_subset_val_count)
        rows = [
            SamplePlan(
                sample_id=f"{split}_{idx:07d}",
                split=split,
                sample_kind="real",
                record_id=record.record_id,
                variant_seed=_sha_like_int([cfg.seed, split, idx, record.record_id]),
            )
            for idx, record in enumerate(real_records)
        ]
        return rows[:subset_limit] if subset_limit is not None else rows

    if not pure_records_by_species:
        raise ValueError("No pure single-species records available for hybrid synthetic generation.")
    missing = [species_id for species_id in class_names if not pure_records_by_species.get(species_id)]
    if missing:
        raise ValueError(f"Hybrid synthetic generation requires at least one pure record per species. Missing: {missing}")

    main_synthetic_count = max(len(records) for records in pure_records_by_species.values())
    if cfg.test_subset_only:
        main_synthetic_count = min(main_synthetic_count, int(cfg.test_subset_train_count))

    total_count = max(main_synthetic_count, int(math.ceil(main_synthetic_count / max(cfg.synthetic_ratio, 1e-6))))
    desired = {
        "synthetic_total": int(round(total_count * cfg.synthetic_ratio)),
        "real": int(round(total_count * cfg.real_ratio)),
        "augmented_single": int(round(total_count * cfg.augmented_ratio)),
    }
    desired["synthetic_total"] = max(desired["synthetic_total"], main_synthetic_count)
    current_total = desired["synthetic_total"] + desired["real"] + desired["augmented_single"]
    if current_total != total_count:
        order = ["synthetic_total", "real", "augmented_single"]
        diff = total_count - current_total
        idx = 0
        while diff != 0:
            key = order[idx % len(order)]
            if diff > 0:
                desired[key] += 1
                diff -= 1
            elif desired[key] > (main_synthetic_count if key == "synthetic_total" else 0):
                desired[key] -= 1
                diff += 1
            idx += 1

    extra_synth_budget = max(0, desired["synthetic_total"] - main_synthetic_count)
    real_count = desired["real"]
    aug_count = desired["augmented_single"]

    rng = random.Random(cfg.seed + _sha_like_int([split, "plans"]))
    real_cycle = list(real_records)
    rng.shuffle(real_cycle)
    if not real_cycle:
        raise ValueError("Hybrid dataset requires at least one real record for the train split.")

    plans: list[SamplePlan] = []
    real_cursor = 0
    gap_remaining = rng.randint(cfg.synthetic_injection_gap_min, cfg.synthetic_injection_gap_max)
    remaining = {
        "real": real_count,
        "augmented_single": aug_count,
        "synthetic_extra": extra_synth_budget,
    }

    def _take_real_plan() -> SamplePlan:
        nonlocal real_cursor
        record = real_cycle[real_cursor % len(real_cycle)]
        real_cursor += 1
        sample_idx = len(plans)
        return SamplePlan(
            sample_id=f"{split}_{sample_idx:07d}",
            split=split,
            sample_kind="real",
            record_id=record.record_id,
            variant_seed=_sha_like_int([cfg.seed, split, sample_idx, record.record_id]),
        )

    species_counts = {species_id: len(records) for species_id, records in pure_records_by_species.items()}
    per_species_aug_cursor = {species_id: 0 for species_id in pure_records_by_species}

    def _take_augmented_plan() -> SamplePlan:
        species_id = _choose_weighted_species(rng, class_names, species_counts)
        pool = pure_records_by_species[species_id]
        cursor = per_species_aug_cursor[species_id]
        record = pool[cursor % len(pool)]
        per_species_aug_cursor[species_id] = cursor + 1
        sample_idx = len(plans)
        return SamplePlan(
            sample_id=f"{split}_{sample_idx:07d}",
            split=split,
            sample_kind="augmented_single",
            record_id=record.record_id,
            species_id=species_id,
            variant_seed=_sha_like_int([cfg.seed, split, sample_idx, record.record_id, "aug"]),
        )

    def _take_extra_synth_plan(main_idx: int) -> SamplePlan:
        sample_idx = len(plans)
        return SamplePlan(
            sample_id=f"{split}_{sample_idx:07d}",
            split=split,
            sample_kind="synthetic_extra",
            synthetic_index=main_idx,
            variant_seed=_sha_like_int([cfg.seed, split, sample_idx, "synthetic_extra"]),
        )

    def _append_injections(main_idx: int) -> None:
        nonlocal gap_remaining
        if sum(remaining.values()) <= 0:
            return
        if gap_remaining > 0:
            return
        inject_n = rng.randint(cfg.injection_count_min, cfg.injection_count_max)
        for _ in range(inject_n):
            options = [kind for kind, count in remaining.items() if count > 0]
            if not options:
                break
            weights = [remaining[k] for k in options]
            kind = rng.choices(options, weights=weights, k=1)[0]
            if kind == "real":
                plans.append(_take_real_plan())
            elif kind == "augmented_single":
                plans.append(_take_augmented_plan())
            else:
                plans.append(_take_extra_synth_plan(main_idx))
            remaining[kind] -= 1
        gap_remaining = rng.randint(cfg.synthetic_injection_gap_min, cfg.synthetic_injection_gap_max)

    for main_idx in range(main_synthetic_count):
        plans.append(
            SamplePlan(
                sample_id=f"{split}_{len(plans):07d}",
                split=split,
                sample_kind="synthetic_main",
                synthetic_index=main_idx,
                variant_seed=_sha_like_int([cfg.seed, split, main_idx, "synthetic_main"]),
            )
        )
        gap_remaining -= 1
        _append_injections(main_idx)

    while sum(remaining.values()) > 0:
        _append_injections(main_synthetic_count)
        if gap_remaining > 0:
            gap_remaining = 0

    expected_total = main_synthetic_count + extra_synth_budget + real_count + aug_count
    if len(plans) != expected_total:
        raise RuntimeError(f"Hybrid planner produced {len(plans)} samples, expected {expected_total}")
    return plans


def build_sample_metadata_rows(plans: list[SamplePlan]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_order, plan in enumerate(plans):
        rows.append(
            {
                "sample_id": plan.sample_id,
                "sample_order": sample_order,
                "split": plan.split,
                "sample_kind": plan.sample_kind,
                "synthetic_index": plan.synthetic_index,
                "record_id": plan.record_id,
                "species_id": plan.species_id,
                "variant_seed": plan.variant_seed,
            }
        )
    return rows


class HybridSpeciesYoloDataset(Dataset):
    """On-the-fly hybrid dataset that returns Ultralytics-compatible detection batches."""

    collate_fn = staticmethod(YOLODataset.collate_fn)

    def __init__(
        self,
        class_names: list[str],
        records: list[SourceImageRecord],
        plans: list[SamplePlan],
        split: str,
        imgsz: int,
        cfg: HybridDatasetConfig,
        augment: bool,
    ) -> None:
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.records_by_id = {record.record_id: record for record in records}
        self.plans = plans
        self.split = split
        self.imgsz = int(imgsz)
        self.cfg = cfg
        self.augment = bool(augment)
        self.rect = False
        self.epoch = 0
        self.decoded_cache = _LruArrayCache(cfg.decoded_image_cache_size)
        self.resized_cache = _LruArrayCache(cfg.resized_image_cache_size)
        self.background_paths = _resolve_background_paths(cfg.synthetic_backgrounds_dir)
        self.records_by_species: dict[str, list[SourceImageRecord]] = {species_id: [] for species_id in class_names}
        self.real_records: list[SourceImageRecord] = []
        for record in records:
            if record.pure_species_id is not None and record.pure_species_id in self.records_by_species:
                self.records_by_species[record.pure_species_id].append(record)
            self.real_records.append(record)
        self.fallback_record_ids = [record.record_id for record in records if record.objects]
        self.transforms = Compose(
            [
                LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=True),
                Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0),
            ]
        )

    def __len__(self) -> int:
        return len(self.plans)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_rng(self, index: int, variant_seed: int) -> random.Random:
        return random.Random(_sha_like_int([self.cfg.seed, self.epoch, index, variant_seed]))

    def _load_image(self, image_path: str) -> np.ndarray:
        cached = self.decoded_cache.get(image_path)
        if cached is not None:
            return cached
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            try:
                with Image.open(image_path) as pil_img:
                    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
                    img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            except (UnidentifiedImageError, OSError, DecompressionBombError, SyntaxError, ValueError):
                img = None
        if img is None:
            raise FileNotFoundError(f"Failed to load source image: {image_path}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Source image did not decode to 3-channel BGR: {image_path}")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        height, width = img.shape[:2]
        max_side = max(height, width)
        if max_side > 2048:
            scale = 2048.0 / float(max_side)
            img = cv2.resize(
                img,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        self.decoded_cache.put(image_path, img)
        return img

    def _resize_image_cached(self, image: np.ndarray, width: int, height: int, cache_key: str) -> np.ndarray:
        cached = self.resized_cache.get(cache_key)
        if cached is not None:
            return cached
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC if width >= image.shape[1] or height >= image.shape[0] else cv2.INTER_AREA)
        self.resized_cache.put(cache_key, resized)
        return resized

    def _render_background_canvas(self, canvas_w: int, canvas_h: int, synthetic_index: int, rng: random.Random) -> tuple[np.ndarray, str | None]:
        if not self.background_paths:
            return np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8), None
        bg_path = self.background_paths[synthetic_index % len(self.background_paths)]
        bg = self._load_image(bg_path)
        bg_h, bg_w = bg.shape[:2]
        scale = max(canvas_w / max(bg_w, 1), canvas_h / max(bg_h, 1))
        scaled_w = max(canvas_w, int(round(bg_w * scale)))
        scaled_h = max(canvas_h, int(round(bg_h * scale)))
        resized = self._resize_image_cached(bg, scaled_w, scaled_h, f"bg:{bg_path}:{scaled_w}:{scaled_h}")
        max_x = max(0, scaled_w - canvas_w)
        max_y = max(0, scaled_h - canvas_h)
        off_x = rng.randint(0, max_x) if max_x > 0 else 0
        off_y = rng.randint(0, max_y) if max_y > 0 else 0
        canvas = resized[off_y : off_y + canvas_h, off_x : off_x + canvas_w].copy()
        return canvas, bg_path

    def _fit_tile_to_canvas(
        self,
        tile_info: dict[str, Any],
        paste_x: int,
        paste_y: int,
        canvas_w: int,
        canvas_h: int,
    ) -> tuple[np.ndarray, list[list[float]], int, int] | None:
        if paste_x < 0 or paste_y < 0 or paste_x >= canvas_w or paste_y >= canvas_h:
            return None
        tile = tile_info["tile"]
        tile_h, tile_w = tile.shape[:2]
        avail_w = canvas_w - paste_x
        avail_h = canvas_h - paste_y
        if avail_w < 32 or avail_h < 32:
            return None
        if tile_w <= avail_w and tile_h <= avail_h:
            return tile, tile_info["boxes"], tile_w, tile_h

        scale = min(avail_w / max(1, tile_w), avail_h / max(1, tile_h))
        if scale <= 0:
            return None
        new_w = max(1, int(round(tile_w * scale)))
        new_h = max(1, int(round(tile_h * scale)))
        if new_w < 32 or new_h < 32:
            return None
        resized = self._resize_image_cached(tile, new_w, new_h, f"fit:{id(tile)}:{paste_x}:{paste_y}:{new_w}:{new_h}")
        boxes = _resize_boxes_xyxy(tile_info["boxes"], new_w / max(1, tile_w), new_h / max(1, tile_h))
        boxes, class_ids, subspecies = _sanitize_labeled_boxes(
            boxes,
            tile_info["class_ids"],
            tile_info["subspecies"],
            new_w,
            new_h,
            min_size=float(self.cfg.synthetic_min_box_size_px),
        )
        if not boxes:
            return None
        tile_info["class_ids"] = class_ids
        tile_info["subspecies"] = subspecies
        return resized, boxes, new_w, new_h

    def _valid_placements(
        self,
        tiles: list[dict[str, Any]],
        placements: list[tuple[int, int]] | None,
        canvas_w: int,
        canvas_h: int,
    ) -> bool:
        if placements is None or len(placements) != len(tiles):
            return False
        for tile, (x, y) in zip(tiles, placements, strict=True):
            h, w = tile["tile"].shape[:2]
            if x < 0 or y < 0 or x + w > canvas_w or y + h > canvas_h:
                return False
        return True

    def _render_real(self, record: SourceImageRecord) -> RenderedSample:
        image = self._load_image(record.image_path).copy()
        boxes: list[list[float]] = []
        class_ids: list[int] = []
        species_ids: list[str] = []
        subspecies: list[str] = []
        for obj in record.objects:
            class_ids.append(int(obj.class_id))
            species_ids.append(obj.species_id)
            subspecies.append(obj.subspecies)
            boxes.append(_normalize_box_xywhn_to_xyxy(obj.bbox_xywhn, record.width, record.height))
        boxes, class_ids, subspecies = _sanitize_labeled_boxes(boxes, class_ids, subspecies, record.width, record.height)
        species_ids = [self.class_names[class_id] for class_id in class_ids]
        return RenderedSample(
            image=image,
            class_ids=class_ids,
            species_ids=species_ids,
            subspecies=subspecies,
            boxes_xyxy=boxes,
            metadata={
                "sample_kind": "real",
                "source_image": record.image_path,
                "dataset_id": record.dataset_id,
                "source_species_ids": record.source_species_ids,
            },
        )

    def _render_augmented_single(self, record: SourceImageRecord, rng: random.Random) -> RenderedSample:
        rendered = self._render_real(record)
        if not rendered.boxes_xyxy:
            return rendered
        crop_image, crop_boxes = _crop_image_to_box_union(
            rendered.image,
            rendered.boxes_xyxy,
            max(8, int(self.cfg.synthetic_crop_margin_px)),
        )
        compose = _build_box_transform(self.cfg, "single_strong", crop_image.shape[:2], include_noise=False)
        transformed = compose(
            image=crop_image,
            bboxes=crop_boxes,
            class_ids=rendered.class_ids,
            subspecies=rendered.subspecies,
        )
        boxes = [list(map(float, box)) for box in transformed["bboxes"]]
        boxes, class_ids, subspecies = _sanitize_labeled_boxes(
            boxes,
            [int(x) for x in transformed["class_ids"]],
            [str(x) for x in transformed["subspecies"]],
            transformed["image"].shape[1],
            transformed["image"].shape[0],
        )
        if not boxes:
            return rendered
        rendered.image = transformed["image"]
        rendered.boxes_xyxy = boxes
        rendered.class_ids = class_ids
        rendered.subspecies = subspecies
        rendered.species_ids = [self.class_names[int(x)] for x in rendered.class_ids]
        rendered.metadata = {
            **rendered.metadata,
            "sample_kind": "augmented_single",
            "augmentations": [
                "horizontal_flip",
                "vertical_flip",
                f"rotate_{self.cfg.strong_rotation_limit}",
                "random_resized_crop",
                "brightness_contrast",
            ],
            "memory_fallback_no_noise": True,
            "rng_marker": rng.randint(0, 2**31 - 1),
        }
        return rendered

    def _render_synthetic(self, synthetic_index: int, rng: random.Random, sample_kind: str) -> RenderedSample:
        species_ids = list(self.class_names)
        rng.shuffle(species_ids)
        n_species = len(species_ids)
        cols_base = max(1, int(round(sqrt(n_species))))
        cols_choices = sorted({cols_base, cols_base + 1, max(1, cols_base - 1)})
        cols = next(cols for cols in cols_choices if cols * ceil(n_species / cols) >= n_species)
        rows = ceil(n_species / cols)

        cell_w = rng.randint(self.cfg.min_cell_size, self.cfg.max_cell_size)
        cell_h = rng.randint(self.cfg.min_cell_size, self.cfg.max_cell_size)
        cell_w = min(cell_w, 560)
        cell_h = min(cell_h, 560)
        gap = max(self.cfg.min_inter_image_padding, self.cfg.min_outer_padding)
        all_boxes: list[list[float]] = []
        all_class_ids: list[int] = []
        all_species_ids: list[str] = []
        all_subspecies: list[str] = []
        source_images: list[str] = []
        tiles: list[dict[str, Any]] = []
        dropped_tiny_boxes = 0
        skipped_tiny_only_tiles: list[str] = []

        for _, species_id in enumerate(species_ids):
            pool = self.records_by_species[species_id]
            record = pool[synthetic_index % len(pool)]
            rendered = self._render_real(record)
            if not rendered.boxes_xyxy:
                continue
            if self.cfg.synthetic_crop_to_objects:
                tile_image, tile_boxes = _crop_image_to_box_union(
                    rendered.image,
                    rendered.boxes_xyxy,
                    max(0, int(self.cfg.synthetic_crop_margin_px)),
                )
                rendered.image = tile_image
                rendered.boxes_xyxy = tile_boxes
            compose = _build_box_transform(self.cfg, "tile_mild", rendered.image.shape[:2])
            transformed = compose(
                image=rendered.image,
                bboxes=rendered.boxes_xyxy,
                class_ids=rendered.class_ids,
                subspecies=rendered.subspecies,
            )
            tile = transformed["image"]
            boxes = [list(map(float, box)) for box in transformed["bboxes"]]
            boxes, class_ids, subspecies = _sanitize_labeled_boxes(
                boxes,
                [int(x) for x in transformed["class_ids"]],
                [str(x) for x in transformed["subspecies"]],
                tile.shape[1],
                tile.shape[0],
            )
            if not boxes:
                tile = rendered.image
                boxes = rendered.boxes_xyxy
                class_ids = rendered.class_ids
                subspecies = rendered.subspecies

            max_tile_w = max(64, cell_w - self.cfg.min_inter_image_padding)
            max_tile_h = max(64, cell_h - self.cfg.min_inter_image_padding)
            scale = min(max_tile_w / max(1, tile.shape[1]), max_tile_h / max(1, tile.shape[0]))
            scale *= rng.uniform(self.cfg.placement_scale_min, self.cfg.placement_scale_max)
            if not self.cfg.allow_scale_up:
                scale = min(scale, 1.0)
            new_w = max(32, int(round(tile.shape[1] * scale)))
            new_h = max(32, int(round(tile.shape[0] * scale)))
            cache_key = f"{record.record_id}:{new_w}:{new_h}"
            tile_resized = self._resize_image_cached(tile, new_w, new_h, cache_key)
            boxes_resized = _resize_boxes_xyxy(boxes, new_w / max(1, tile.shape[1]), new_h / max(1, tile.shape[0]))
            boxes_resized, class_ids, subspecies = _sanitize_labeled_boxes(
                boxes_resized,
                class_ids,
                subspecies,
                new_w,
                new_h,
                min_size=float(self.cfg.synthetic_min_box_size_px),
            )
            dropped_tiny_boxes += len(boxes) - len(boxes_resized)
            if not boxes_resized:
                skipped_tiny_only_tiles.append(record.record_id)
                continue
            # Species-level synthetic mosaics need broad species coverage, not
            # every crowd instance. Keeping the largest object per tile prevents
            # YOLO's assigner from allocating huge bs x boxes x anchors tensors.
            boxes_resized, class_ids, subspecies = _limit_boxes_by_area(
                boxes_resized,
                class_ids,
                subspecies,
                limit=1,
            )
            tiles.append(
                {
                    "record": record,
                    "tile": tile_resized,
                    "boxes": boxes_resized,
                    "class_ids": class_ids,
                    "subspecies": subspecies,
                    "width": new_w,
                    "height": new_h,
                }
            )

        col_widths, row_heights = _grid_track_sizes(tiles, rows, cols)
        preferred_w = max(self.cfg.min_canvas_size, sum(col_widths) + (cols + 1) * gap)
        preferred_h = max(self.cfg.min_canvas_size, sum(row_heights) + (rows + 1) * gap)
        canvas_limit = min(int(self.cfg.absolute_max_canvas_size), int(self.cfg.max_canvas_size), 1536)
        canvas_w = _round_to_stride(min(max(preferred_w, self.cfg.min_canvas_size), canvas_limit))
        canvas_h = _round_to_stride(min(max(preferred_h, self.cfg.min_canvas_size), canvas_limit))

        placements: list[tuple[int, int]] | None = None
        used_fallback = False
        for attempt in range(self.cfg.placement_retry_count):
            placements = self._try_grid_layout(tiles, rows, cols, canvas_w, canvas_h, gap, rng)
            if placements is not None:
                break
            growth = 1.0 + 0.08 * (attempt + 1)
            grown_w = min(canvas_limit, _round_to_stride(int(canvas_w * growth)))
            grown_h = min(canvas_limit, _round_to_stride(int(canvas_h * growth)))
            if grown_w == canvas_w and grown_h == canvas_h:
                break
            canvas_w, canvas_h = grown_w, grown_h

        if not self._valid_placements(tiles, placements, canvas_w, canvas_h):
            placements, canvas_w, canvas_h = self._greedy_pack_layout(tiles, canvas_w, canvas_h, gap)
            used_fallback = True

        canvas, background_path = self._render_background_canvas(canvas_w, canvas_h, synthetic_index, rng)
        if background_path is None and rng.random() < rng.uniform(
            self.cfg.background_texture_probability_min,
            self.cfg.background_texture_probability_max,
        ):
            noise = rng.randint(1, 5)
            texture = np.random.default_rng(_sha_like_int([self.cfg.seed, self.epoch, synthetic_index])).normal(
                loc=0.0,
                scale=noise,
                size=canvas.shape,
            )
            canvas = np.clip(canvas.astype(np.int16) + texture.astype(np.int16), 245, 255).astype(np.uint8)

        skipped_out_of_bounds_tiles: list[str] = []
        for tile_info, (paste_x, paste_y) in zip(tiles, placements or [], strict=True):
            fitted = self._fit_tile_to_canvas(tile_info, int(paste_x), int(paste_y), canvas_w, canvas_h)
            if fitted is None:
                skipped_out_of_bounds_tiles.append(tile_info["record"].record_id)
                continue
            tile_resized, fitted_boxes, new_w, new_h = fitted
            canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = tile_resized
            offset_boxes = _resize_boxes_xyxy(fitted_boxes, 1.0, 1.0, paste_x, paste_y)
            all_boxes.extend(offset_boxes)
            all_class_ids.extend(tile_info["class_ids"])
            all_species_ids.extend([self.class_names[class_id] for class_id in tile_info["class_ids"]])
            all_subspecies.extend(tile_info["subspecies"])
            source_images.append(tile_info["record"].image_path)

        all_boxes, all_class_ids, all_subspecies = _sanitize_labeled_boxes(
            all_boxes,
            all_class_ids,
            all_subspecies,
            canvas_w,
            canvas_h,
            min_size=float(self.cfg.synthetic_min_box_size_px),
        )
        all_species_ids = [self.class_names[class_id] for class_id in all_class_ids]
        if not all_boxes and tiles:
            record = tiles[0]["record"]
            rendered = self._render_real(record)
            rendered.metadata["sample_kind"] = f"{sample_kind}_fallback_real"
            rendered.metadata["synthetic_fallback_reason"] = "no_fit_tiles"
            return rendered

        return RenderedSample(
            image=canvas,
            class_ids=all_class_ids,
            species_ids=all_species_ids,
            subspecies=all_subspecies,
            boxes_xyxy=all_boxes,
            metadata={
                "sample_kind": sample_kind,
                "synthetic_index": synthetic_index,
                "canvas_size": [int(canvas_w), int(canvas_h)],
                "grid": {"rows": rows, "cols": cols},
                "source_images": source_images,
                "background_texture": bool(background_path is None and np.any(canvas < 255)),
                "background_image": background_path,
                "placement_strategy": "greedy_fallback" if used_fallback else "grid",
                "synthetic_crop_to_objects": bool(self.cfg.synthetic_crop_to_objects),
                "synthetic_crop_margin_px": int(self.cfg.synthetic_crop_margin_px),
                "synthetic_min_box_size_px": float(self.cfg.synthetic_min_box_size_px),
                "dropped_tiny_boxes": int(dropped_tiny_boxes),
                "skipped_tiny_only_tiles": skipped_tiny_only_tiles,
                "skipped_out_of_bounds_tiles": skipped_out_of_bounds_tiles,
            },
        )

    def _try_grid_layout(
        self,
        tiles: list[dict[str, Any]],
        rows: int,
        cols: int,
        canvas_w: int,
        canvas_h: int,
        gap: int,
        rng: random.Random,
    ) -> list[tuple[int, int]] | None:
        placements: list[tuple[int, int]] = []
        col_widths, row_heights = _grid_track_sizes(tiles, rows, cols)
        packed_w = sum(col_widths) + (cols + 1) * gap
        packed_h = sum(row_heights) + (rows + 1) * gap
        if packed_w > canvas_w or packed_h > canvas_h:
            return None
        start_x = max(gap, (canvas_w - packed_w) // 2 + gap)
        start_y = max(gap, (canvas_h - packed_h) // 2 + gap)
        rects: list[tuple[int, int, int, int]] = []
        for idx, tile in enumerate(tiles):
            row = idx // cols
            col = idx % cols
            tile_w = int(tile["width"])
            tile_h = int(tile["height"])
            if row >= rows or col >= cols:
                return None
            cell_w = col_widths[col]
            cell_h = row_heights[row]
            if tile_w > cell_w or tile_h > cell_h:
                return None
            base_x = start_x + sum(col_widths[:col]) + col * gap
            base_y = start_y + sum(row_heights[:row]) + row * gap
            max_x = base_x + max(0, cell_w - tile_w)
            max_y = base_y + max(0, cell_h - tile_h)
            jitter_x = int(max(0, max_x - base_x) * self.cfg.grid_jitter)
            jitter_y = int(max(0, max_y - base_y) * self.cfg.grid_jitter)
            paste_x = min(max_x, max(base_x, base_x + rng.randint(-jitter_x, jitter_x)))
            paste_y = min(max_y, max(base_y, base_y + rng.randint(-jitter_y, jitter_y)))
            rect = (paste_x, paste_y, paste_x + tile_w, paste_y + tile_h)
            if rect[2] > canvas_w - gap or rect[3] > canvas_h - gap:
                return None
            for ox0, oy0, ox1, oy1 in rects:
                if not (
                    rect[2] + self.cfg.min_inter_image_padding <= ox0
                    or rect[0] >= ox1 + self.cfg.min_inter_image_padding
                    or rect[3] + self.cfg.min_inter_image_padding <= oy0
                    or rect[1] >= oy1 + self.cfg.min_inter_image_padding
                ):
                    return None
            rects.append(rect)
            placements.append((paste_x, paste_y))
        return placements

    def _greedy_pack_layout(
        self,
        tiles: list[dict[str, Any]],
        canvas_w: int,
        canvas_h: int,
        gap: int,
    ) -> tuple[list[tuple[int, int]], int, int]:
        order = sorted(range(len(tiles)), key=lambda idx: tiles[idx]["height"], reverse=True)
        placements: list[tuple[int, int] | None] = [None] * len(tiles)
        current_y = gap
        row_height = 0
        current_x = gap
        needed_w = canvas_w
        needed_h = canvas_h
        canvas_cap = min(int(self.cfg.absolute_max_canvas_size), int(self.cfg.max_canvas_size), 1536)

        for idx in order:
            tile = tiles[idx]
            w = int(tile["width"])
            h = int(tile["height"])
            if current_x + w > canvas_w - gap:
                current_x = gap
                current_y += row_height + gap
                row_height = 0
            if current_y + h > canvas_h - gap:
                scale = max(self.cfg.greedy_min_scale, min((canvas_w - 2 * gap) / max(w, 1), (canvas_h - 2 * gap) / max(h, 1)))
                if scale < 1.0:
                    new_w = max(32, int(round(w * scale)))
                    new_h = max(32, int(round(h * scale)))
                    record_id = tiles[idx]["record"].record_id
                    cache_key = f"{record_id}:{new_w}:{new_h}:greedy"
                    original_tile = tiles[idx]["tile"]
                    resized = self._resize_image_cached(original_tile, new_w, new_h, cache_key)
                    tiles[idx]["tile"] = resized
                    tiles[idx]["boxes"] = _resize_boxes_xyxy(tiles[idx]["boxes"], new_w / max(w, 1), new_h / max(h, 1))
                    tiles[idx]["width"] = new_w
                    tiles[idx]["height"] = new_h
                    w = new_w
                    h = new_h
                if current_y + h > canvas_h - gap:
                    needed_h = max(needed_h, _round_to_stride(current_y + h + gap))
                    canvas_h = min(canvas_cap, needed_h)
            placements[idx] = (current_x, current_y)
            current_x += w + gap
            row_height = max(row_height, h)
            needed_w = max(needed_w, _round_to_stride(current_x + gap))
        finalized = [p for p in placements if p is not None]
        return finalized, int(min(canvas_cap, needed_w)), int(min(canvas_cap, max(needed_h, current_y + row_height + gap)))

    def _to_ultralytics_label(self, rendered: RenderedSample, sample_id: str) -> dict[str, Any]:
        height, width = rendered.image.shape[:2]
        clipped_boxes: list[list[float]] = []
        class_ids: list[int] = []
        species_ids: list[str] = []
        subspecies: list[str] = []
        for box, class_id, species_id, subtype in zip(
            rendered.boxes_xyxy,
            rendered.class_ids,
            rendered.species_ids,
            rendered.subspecies,
            strict=True,
        ):
            clipped = _clip_box_xyxy(box, width, height)
            if clipped is None:
                continue
            clipped_boxes.append(clipped)
            class_ids.append(int(class_id))
            species_ids.append(species_id)
            subspecies.append(subtype)

        object_cap = self.cfg.max_train_objects_per_image if self.split == "train" else self.cfg.max_eval_objects_per_image
        if object_cap > 0 and len(clipped_boxes) > object_cap:
            order = sorted(
                range(len(clipped_boxes)),
                key=lambda idx: max(0.0, clipped_boxes[idx][2] - clipped_boxes[idx][0])
                * max(0.0, clipped_boxes[idx][3] - clipped_boxes[idx][1]),
                reverse=True,
            )[: int(object_cap)]
            order.sort()
            clipped_boxes = [clipped_boxes[i] for i in order]
            class_ids = [class_ids[i] for i in order]
            species_ids = [species_ids[i] for i in order]
            subspecies = [subspecies[i] for i in order]

        boxes_np = np.asarray(clipped_boxes, dtype=np.float32) if clipped_boxes else np.zeros((0, 4), dtype=np.float32)
        cls_np = np.asarray(class_ids, dtype=np.float32).reshape(-1, 1) if class_ids else np.zeros((0, 1), dtype=np.float32)
        segments_np = np.zeros((len(clipped_boxes), 0, 2), dtype=np.float32)
        labels = {
            "img": rendered.image,
            "cls": cls_np,
            "instances": Instances(boxes_np, segments=segments_np, bbox_format="xyxy", normalized=False),
            "im_file": f"hybrid://{self.split}/{sample_id}.jpg",
            "ori_shape": (height, width),
            "resized_shape": (height, width),
            "ratio_pad": (1.0, 1.0),
            "sample_kind": rendered.metadata.get("sample_kind", "unknown"),
            "species_ids": species_ids,
            "subspecies": subspecies,
            "metadata": rendered.metadata,
        }
        return self.transforms(labels)

    def _validate_training_label(self, labels: dict[str, Any], sample_id: str) -> None:
        cls = labels.get("cls")
        if cls is None:
            raise ValueError(f"Rendered sample has no cls tensor: {sample_id}")
        if isinstance(cls, torch.Tensor):
            if cls.numel() == 0:
                raise ValueError(f"Rendered sample has zero labels after transform: {sample_id}")
            cls_min = float(cls.min().item())
            cls_max = float(cls.max().item())
        else:
            arr = np.asarray(cls)
            if arr.size == 0:
                raise ValueError(f"Rendered sample has zero labels after transform: {sample_id}")
            cls_min = float(arr.min())
            cls_max = float(arr.max())
        if cls_min < 0 or cls_max >= len(self.class_names):
            raise ValueError(
                f"Rendered sample has invalid class id range {cls_min}..{cls_max} "
                f"for {len(self.class_names)} classes: {sample_id}"
            )

    def _render_plan(self, plan: SamplePlan, index: int) -> dict[str, Any]:
        rng = self._sample_rng(index, plan.variant_seed)
        if plan.sample_kind == "real":
            assert plan.record_id is not None
            rendered = self._render_real(self.records_by_id[plan.record_id])
        elif plan.sample_kind == "augmented_single":
            assert plan.record_id is not None
            rendered = self._render_augmented_single(self.records_by_id[plan.record_id], rng)
        elif plan.sample_kind in {"synthetic_main", "synthetic_extra"}:
            rendered = self._render_synthetic(plan.synthetic_index, rng, plan.sample_kind)
        else:
            raise ValueError(f"Unsupported sample kind: {plan.sample_kind}")
        rendered.metadata["sample_id"] = plan.sample_id
        rendered.metadata["split"] = plan.split
        labels = self._to_ultralytics_label(rendered, plan.sample_id)
        self._validate_training_label(labels, plan.sample_id)
        return labels

    def _render_fallback_real(self, index: int, reason: str) -> dict[str, Any]:
        if not self.fallback_record_ids:
            raise ValueError("No fallback labeled real records are available.")
        total = len(self.fallback_record_ids)
        for offset in range(min(total, 64)):
            record_id = self.fallback_record_ids[(index + offset) % total]
            rendered = self._render_real(self.records_by_id[record_id])
            rendered.metadata["sample_id"] = f"fallback_{index}_{offset}"
            rendered.metadata["split"] = self.split
            rendered.metadata["fallback_reason"] = reason
            labels = self._to_ultralytics_label(rendered, f"fallback_{index}_{offset}")
            try:
                self._validate_training_label(labels, f"fallback_{index}_{offset}")
            except ValueError:
                continue
            return labels
        raise ValueError(f"No valid fallback labeled real sample found after {min(total, 64)} attempts.")

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_exc: Exception | None = None
        total = max(1, len(self.plans))
        for attempt in range(min(16, total)):
            candidate_index = (index + attempt) % total
            plan = self.plans[candidate_index]
            try:
                return self._render_plan(plan, candidate_index)
            except Exception as exc:
                last_exc = exc
                print(
                    f"[hybrid-dataset] split={self.split} sample={plan.sample_id} kind={plan.sample_kind} "
                    f"index={candidate_index} attempt={attempt + 1} error={exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
        return self._render_fallback_real(index, str(last_exc) if last_exc is not None else "unknown")

    def export_debug_subset(self, output_dir: str | Path, count: int | None = None) -> dict[str, Any]:
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
        limit = min(len(self), count if count is not None else self.cfg.debug_subset_count)
        manifest: list[dict[str, Any]] = []

        for idx in range(limit):
            plan = self.plans[idx]
            rng = self._sample_rng(idx, plan.variant_seed)
            if plan.sample_kind == "real":
                rendered = self._render_real(self.records_by_id[plan.record_id or ""])
            elif plan.sample_kind == "augmented_single":
                rendered = self._render_augmented_single(self.records_by_id[plan.record_id or ""], rng)
            else:
                rendered = self._render_synthetic(plan.synthetic_index, rng, plan.sample_kind)
            img_path = output_dir / "images" / f"{plan.sample_id}.jpg"
            label_path = output_dir / "labels" / f"{plan.sample_id}.txt"
            meta_path = output_dir / "metadata" / f"{plan.sample_id}.json"
            cv2.imwrite(str(img_path), rendered.image)
            width = rendered.image.shape[1]
            height = rendered.image.shape[0]
            label_lines = []
            for class_id, box in zip(rendered.class_ids, rendered.boxes_xyxy, strict=True):
                clipped = _clip_box_xyxy(box, width, height)
                if clipped is None:
                    continue
                xywhn = _xyxy_to_xywhn(np.asarray([clipped], dtype=np.float32), width, height)[0]
                label_lines.append(
                    f"{int(class_id)} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}"
                )
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            meta_payload = {
                "sample_id": plan.sample_id,
                "sample_kind": plan.sample_kind,
                "split": plan.split,
                "image_path": str(img_path.resolve()),
                "label_path": str(label_path.resolve()),
                "class_names": [self.class_names[x] for x in rendered.class_ids],
                "subspecies": rendered.subspecies,
                "metadata": rendered.metadata,
            }
            meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
            manifest.append(meta_payload)

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "output_dir": str(output_dir.resolve()),
            "count": limit,
            "manifest_path": str(manifest_path.resolve()),
        }
