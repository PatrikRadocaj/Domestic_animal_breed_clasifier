"""Cross-dataset mining for unlabeled animal candidates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import cv2
import hydra
import pandas as pd
from omegaconf import DictConfig
from ultralytics import YOLO

from animal_breed.data.label_mapping import CrossDatasetLabelMap
from animal_breed.detection.yolo_source_index import discover_yolo_sources, resolve_image_for_label
from animal_breed.utils.logging import setup_logging


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ExistingLabel:
    species_id: str
    bbox_xyxy_norm: tuple[float, float, float, float]
    source_dataset: str


def _xywhn_to_xyxy_abs(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    bw = max(0.0, w * img_w)
    bh = max(0.0, h * img_h)
    cx = x * img_w
    cy = y * img_h
    x0 = max(0.0, cx - bw / 2.0)
    y0 = max(0.0, cy - bh / 2.0)
    x1 = min(float(img_w), cx + bw / 2.0)
    y1 = min(float(img_h), cy + bh / 2.0)
    return x0, y0, x1, y1


def _xyxy_abs_to_norm(box: tuple[float, float, float, float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    if img_w <= 0 or img_h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _nms(dets: list[dict[str, Any]], iou_thr: float) -> list[dict[str, Any]]:
    if not dets:
        return []
    ordered = sorted(dets, key=lambda x: float(x["confidence"]), reverse=True)
    kept: list[dict[str, Any]] = []
    while ordered:
        head = ordered.pop(0)
        kept.append(head)
        head_box = tuple(head["bbox_xyxy"])  # type: ignore[arg-type]
        rest: list[dict[str, Any]] = []
        for item in ordered:
            box = tuple(item["bbox_xyxy"])  # type: ignore[arg-type]
            if _iou_xyxy(head_box, box) <= iou_thr:
                rest.append(item)
        ordered = rest
    return kept


def _flip_box_to_original(
    box_xyxy_flipped: tuple[float, float, float, float], image_w: int
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box_xyxy_flipped
    return float(image_w) - x1, y0, float(image_w) - x0, y1


def _parse_model_boxes(
    result: Any, species_id: str, conf_default_breed: str = ""
) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None:
        return []
    names = result.names if hasattr(result, "names") else {}
    out: list[dict[str, Any]] = []
    for i in range(len(boxes)):
        x0, y0, x1, y1 = [float(x) for x in boxes.xyxy[i].tolist()]
        cls_id = int(float(boxes.cls[i].item())) if hasattr(boxes, "cls") else 0
        class_name = ""
        if isinstance(names, dict):
            class_name = str(names.get(cls_id, "")).strip()
        predicted_breed = conf_default_breed
        if class_name and class_name.lower() != species_id.lower():
            predicted_breed = class_name
        out.append(
            {
                "predicted_species": species_id,
                "predicted_breed": predicted_breed,
                "confidence": float(boxes.conf[i].item()),
                "bbox_xyxy": [x0, y0, x1, y1],
            }
        )
    return out


def _parse_shared_model_boxes(
    result: Any,
    allowed_species: set[str] | None = None,
) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None:
        return []
    names = result.names if hasattr(result, "names") else {}
    out: list[dict[str, Any]] = []
    for i in range(len(boxes)):
        x0, y0, x1, y1 = [float(x) for x in boxes.xyxy[i].tolist()]
        cls_id = int(float(boxes.cls[i].item())) if hasattr(boxes, "cls") else 0
        class_name = ""
        if isinstance(names, dict):
            class_name = str(names.get(cls_id, "")).strip().lower()
        if not class_name:
            continue
        if allowed_species is not None and class_name not in allowed_species:
            continue
        out.append(
            {
                "predicted_species": class_name,
                "predicted_breed": "",
                "confidence": float(boxes.conf[i].item()),
                "bbox_xyxy": [x0, y0, x1, y1],
            }
        )
    return out


def _predict_species_detections(
    model: YOLO,
    image_bgr,
    species_id: str,
    conf_thr: float,
    nms_iou: float,
    min_area: float,
    min_side: float,
    max_area_frac: float,
    max_side_frac: float,
    max_per_species: int,
) -> list[dict[str, Any]]:
    results = model.predict(source=[image_bgr], conf=conf_thr, iou=nms_iou, verbose=False)
    if not results:
        return []
    img_h, img_w = image_bgr.shape[:2]
    img_area = float(max(1, img_h * img_w))
    detections = _parse_model_boxes(results[0], species_id)
    filtered: list[dict[str, Any]] = []
    for det in detections:
        x0, y0, x1, y1 = [float(v) for v in det["bbox_xyxy"]]
        bw = max(0.0, x1 - x0)
        bh = max(0.0, y1 - y0)
        area = bw * bh
        if bw < min_side or bh < min_side or area < min_area:
            continue
        if max_area_frac > 0 and (area / img_area) > max_area_frac:
            continue
        if max_side_frac > 0:
            width_frac = bw / max(1.0, float(img_w))
            height_frac = bh / max(1.0, float(img_h))
            if width_frac > max_side_frac or height_frac > max_side_frac:
                continue
        filtered.append(det)
    filtered = sorted(filtered, key=lambda x: float(x["confidence"]), reverse=True)
    if max_per_species > 0:
        filtered = filtered[:max_per_species]
    return filtered


def _predict_shared_detections(
    model: YOLO,
    image_bgr,
    conf_thr: float,
    nms_iou: float,
    min_area: float,
    min_side: float,
    max_area_frac: float,
    max_side_frac: float,
    max_per_image: int,
    allowed_species: set[str] | None = None,
) -> list[dict[str, Any]]:
    results = model.predict(source=[image_bgr], conf=conf_thr, iou=nms_iou, verbose=False)
    if not results:
        return []
    img_h, img_w = image_bgr.shape[:2]
    img_area = float(max(1, img_h * img_w))
    detections = _parse_shared_model_boxes(results[0], allowed_species=allowed_species)
    filtered: list[dict[str, Any]] = []
    for det in detections:
        x0, y0, x1, y1 = [float(v) for v in det["bbox_xyxy"]]
        bw = max(0.0, x1 - x0)
        bh = max(0.0, y1 - y0)
        area = bw * bh
        if bw < min_side or bh < min_side or area < min_area:
            continue
        if max_area_frac > 0 and (area / img_area) > max_area_frac:
            continue
        if max_side_frac > 0:
            width_frac = bw / max(1.0, float(img_w))
            height_frac = bh / max(1.0, float(img_h))
            if width_frac > max_side_frac or height_frac > max_side_frac:
                continue
        filtered.append(det)
    filtered = _nms(filtered, nms_iou)
    filtered = sorted(filtered, key=lambda x: float(x["confidence"]), reverse=True)
    if max_per_image > 0:
        filtered = filtered[:max_per_image]
    return filtered


def _agreement_iou_for_detection(
    model: YOLO,
    image_bgr,
    det_box_xyxy: tuple[float, float, float, float],
    species_id: str,
    conf_thr: float,
    nms_iou: float,
    min_agreement_iou: float,
) -> float:
    h, w = image_bgr.shape[:2]
    flipped = cv2.flip(image_bgr, 1)
    results = model.predict(source=[flipped], conf=conf_thr, iou=nms_iou, verbose=False)
    if not results:
        return 0.0
    flipped_dets = _parse_model_boxes(results[0], species_id)
    if not flipped_dets:
        return 0.0
    best = 0.0
    for d in flipped_dets:
        box_f = tuple(float(v) for v in d["bbox_xyxy"])
        box = _flip_box_to_original(box_f, w)
        iou = _iou_xyxy(det_box_xyxy, box)
        best = max(best, iou)
    if best < min_agreement_iou:
        return best
    return best


def _agreement_iou_for_shared_detection(
    model: YOLO,
    image_bgr,
    det_box_xyxy: tuple[float, float, float, float],
    predicted_species: str,
    conf_thr: float,
    nms_iou: float,
    min_agreement_iou: float,
    allowed_species: set[str] | None = None,
) -> float:
    _, w = image_bgr.shape[:2]
    flipped = cv2.flip(image_bgr, 1)
    results = model.predict(source=[flipped], conf=conf_thr, iou=nms_iou, verbose=False)
    if not results:
        return 0.0
    flipped_dets = _parse_shared_model_boxes(results[0], allowed_species=allowed_species)
    if not flipped_dets:
        return 0.0
    best = 0.0
    for d in flipped_dets:
        if str(d["predicted_species"]) != predicted_species:
            continue
        box_f = tuple(float(v) for v in d["bbox_xyxy"])
        box = _flip_box_to_original(box_f, w)
        iou = _iou_xyxy(det_box_xyxy, box)
        best = max(best, iou)
    if best < min_agreement_iou:
        return best
    return best


def _source_dataset_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for marker in ("yolo_sources", "non_yolo_sources"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(path.parts):
                return path.parts[idx + 1]
    return "unknown"


def _collect_all_images(roots: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        p = Path(str(root))
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            out.append(f.resolve())
    return out


def _collect_yolo_images(yolo_sources_root: str | Path) -> list[Path]:
    out: list[Path] = []
    for ds in discover_yolo_sources(yolo_sources_root):
        for image_dir in ds.split_image_dirs.values():
            if not image_dir.exists():
                continue
            for f in image_dir.rglob("*"):
                if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                    continue
                out.append(f.resolve())
    return out


def _resolve_output_stem(base_name: str, suffix: str) -> str:
    clean_suffix = suffix.strip()
    return f"{base_name}_{clean_suffix}" if clean_suffix else base_name


def _limit_images_balanced(image_paths: list[Path], max_images: int, balanced_by_dataset: bool) -> list[Path]:
    if max_images <= 0 or len(image_paths) <= max_images:
        return image_paths
    if not balanced_by_dataset:
        return image_paths[:max_images]

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        groups[_source_dataset_from_path(path)].append(path)

    dataset_ids = sorted(groups.keys())
    out: list[Path] = []
    offset = 0
    while len(out) < max_images:
        added = False
        for dataset_id in dataset_ids:
            items = groups[dataset_id]
            if offset < len(items):
                out.append(items[offset])
                added = True
                if len(out) >= max_images:
                    return out
        if not added:
            break
        offset += 1
    return out


def _build_existing_label_index(cfg: DictConfig, label_map: CrossDatasetLabelMap, logger) -> dict[str, list[ExistingLabel]]:
    yolo_root = Path(str(cfg.data.yolo_sources_root))
    datasets = discover_yolo_sources(yolo_root)
    existing: dict[str, list[ExistingLabel]] = defaultdict(list)
    for ds in datasets:
        class_to_species: dict[int, str] = {}
        for idx, cls_name in enumerate(ds.names):
            resolved = label_map.resolve_species(cls_name, ds.dataset_id)
            if resolved.status == "resolved" and resolved.canonical_id:
                class_to_species[idx] = resolved.canonical_id

        for split, label_dir in ds.split_label_dirs.items():
            image_dir = ds.split_image_dirs.get(split)
            if image_dir is None or not label_dir.exists():
                continue
            for label_path in label_dir.rglob("*.txt"):
                image_path = resolve_image_for_label(label_path, image_dir)
                if image_path is None:
                    continue
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cls_idx = int(float(parts[0]))
                        x, y, bw, bh = [float(v) for v in parts[1:]]
                    except ValueError:
                        continue
                    sp = class_to_species.get(cls_idx)
                    if sp is None:
                        continue
                    box = _xywhn_to_xyxy_abs(x, y, bw, bh, w, h)
                    existing[str(image_path)].append(
                        ExistingLabel(
                            species_id=sp,
                            bbox_xyxy_norm=_xyxy_abs_to_norm(box, w, h),
                            source_dataset=ds.dataset_id,
                        )
                    )
    logger.info("Built existing-label index for %d images from YOLO-labeled datasets", len(existing))
    return existing


def _load_species_detectors(cfg: DictConfig, logger) -> dict[str, YOLO]:
    detectors: dict[str, YOLO] = {}
    weights_root = Path(str(cfg.eval.mining.detectors_weights_root))
    if weights_root.exists():
        for sub in sorted([p for p in weights_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            candidates = [
                sub / "weights" / "best.pt",
                sub / "best.pt",
            ]
            chosen = next((c for c in candidates if c.exists()), None)
            if chosen is None:
                continue
            detectors[sub.name] = YOLO(str(chosen))
    logger.info("Loaded %d species detectors from %s", len(detectors), weights_root)
    return detectors


def _load_shared_detector(cfg: DictConfig, logger) -> YOLO:
    weights_path = Path(str(cfg.eval.mining.shared_detector_weights))
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Shared detector weights not found at {weights_path}. "
            "Train the shared species detector first or update eval.mining.shared_detector_weights."
        )
    logger.info("Loaded shared detector from %s", weights_path)
    return YOLO(str(weights_path))


def run_unlabeled_mining(cfg: DictConfig, logger) -> dict[str, Any]:
    label_map = CrossDatasetLabelMap.from_file(str(cfg.data.cross_dataset_label_map_path))
    existing_by_image = _build_existing_label_index(cfg, label_map, logger)
    detector_mode = str(getattr(cfg.eval.mining, "detector_mode", "per_species")).strip().lower()
    detectors: dict[str, YOLO] = {}
    shared_detector: YOLO | None = None
    allowed_species_cfg = list(getattr(cfg.eval.mining, "allowed_species", []) or [])
    allowed_species = {str(x).strip().lower() for x in allowed_species_cfg if str(x).strip()} or None
    if detector_mode == "per_species":
        detectors = _load_species_detectors(cfg, logger)
        if not detectors:
            raise FileNotFoundError(
                f"No species detector weights found in {cfg.eval.mining.detectors_weights_root}. "
                "Train species-aware detectors first or set detectors path."
            )
    elif detector_mode == "shared_multi":
        shared_detector = _load_shared_detector(cfg, logger)
    else:
        raise ValueError(
            f"Unsupported eval.mining.detector_mode={detector_mode!r}. "
            "Expected 'per_species' or 'shared_multi'."
        )

    source_mode = str(getattr(cfg.eval.mining, "source_mode", "yolo_only")).strip().lower()
    if source_mode == "yolo_only":
        image_paths = _collect_yolo_images(
            cfg.data.yolo_sources_root,
        )
    elif source_mode == "all_raw":
        image_paths = _collect_all_images(
            [Path(str(x)) for x in cfg.eval.mining.image_roots],
        )
    else:
        raise ValueError(
            f"Unsupported eval.mining.source_mode={source_mode!r}. Expected 'yolo_only' or 'all_raw'."
        )
    if bool(getattr(cfg.eval.mining, "sort_images", True)):
        image_paths = sorted(image_paths, key=lambda p: str(p).lower())
    allowed_source_datasets_cfg = list(getattr(cfg.eval.mining, "allowed_source_datasets", []) or [])
    excluded_source_datasets_cfg = list(getattr(cfg.eval.mining, "excluded_source_datasets", []) or [])
    allowed_source_datasets = {str(x).strip() for x in allowed_source_datasets_cfg if str(x).strip()}
    excluded_source_datasets = {str(x).strip() for x in excluded_source_datasets_cfg if str(x).strip()}
    if allowed_source_datasets:
        image_paths = [p for p in image_paths if _source_dataset_from_path(p) in allowed_source_datasets]
    if excluded_source_datasets:
        image_paths = [p for p in image_paths if _source_dataset_from_path(p) not in excluded_source_datasets]
    shard_count = max(1, int(getattr(cfg.eval.mining, "shard_count", 1)))
    shard_index = int(getattr(cfg.eval.mining, "shard_index", 0))
    if shard_count > 1:
        shard_index = max(0, min(shard_index, shard_count - 1))
        image_paths = [p for i, p in enumerate(image_paths) if i % shard_count == shard_index]
    max_images = int(cfg.eval.mining.max_images)
    balanced_limit = bool(getattr(cfg.eval.mining, "balanced_max_images_across_datasets", source_mode == "yolo_only"))
    image_paths = _limit_images_balanced(image_paths, max_images=max_images, balanced_by_dataset=balanced_limit)
    logger.info("Mining across %d images using source_mode=%s", len(image_paths), source_mode)

    conf_default = float(cfg.eval.mining.conf_threshold_default)
    conf_overrides = {str(k): float(v) for k, v in dict(cfg.eval.mining.conf_thresholds).items()}
    min_area = float(cfg.eval.mining.min_box_area_px)
    min_side = float(cfg.eval.mining.min_box_side_px)
    max_area_frac = float(getattr(cfg.eval.mining, "max_box_area_frac", 0.0))
    max_side_frac = float(getattr(cfg.eval.mining, "max_box_side_frac", 0.0))
    max_per_species = int(getattr(cfg.eval.mining, "max_detections_per_species_per_image", 0))
    max_per_image = int(getattr(cfg.eval.mining, "max_detections_per_image", 0))
    require_existing_labels = bool(getattr(cfg.eval.mining, "require_existing_labels", True))
    include_same_species_extra = bool(getattr(cfg.eval.mining, "include_same_species_extra", False))
    nms_iou = float(cfg.eval.mining.nms_iou)
    label_iou_match = float(cfg.eval.mining.label_match_iou)
    agreement_cfg = cfg.eval.mining.agreement_check
    agreement_enabled = bool(getattr(agreement_cfg, "enabled", False))
    agreement_required = bool(getattr(agreement_cfg, "required", False))
    agreement_iou_min = float(getattr(agreement_cfg, "min_iou", 0.5))
    agreement_conf_scale = float(getattr(agreement_cfg, "conf_threshold_scale", 1.0))
    agreement_conf_min = float(getattr(agreement_cfg, "conf_threshold_min", 0.25))
    spotcheck_size = int(getattr(cfg.eval.mining, "spotcheck_sample_size", 0))
    spotcheck_seed = int(getattr(cfg.eval.mining, "spotcheck_seed", 42))
    high_priority_conflict_types_cfg = list(
        getattr(cfg.eval.mining, "high_priority_conflict_types", ["background_unlabeled"]) or ["background_unlabeled"]
    )
    high_priority_conflict_types = {str(x).strip() for x in high_priority_conflict_types_cfg if str(x).strip()}
    output_suffix = str(getattr(cfg.eval.mining, "output_name_suffix", "")).strip()
    if not output_suffix and shard_count > 1:
        output_suffix = f"shard_{shard_index:03d}_of_{shard_count:03d}"

    out_rows: list[dict[str, Any]] = []
    filtered_low_agreement = 0
    samples_written = 0
    visual_dir_base = Path(str(cfg.eval.mining.visual_samples_dir))
    visual_dir = visual_dir_base / output_suffix if output_suffix else visual_dir_base
    visual_dir.mkdir(parents=True, exist_ok=True)
    sample_limit = int(cfg.eval.mining.visual_samples_limit)

    for image_path in image_paths:
        existing = existing_by_image.get(str(image_path), [])
        if require_existing_labels and not existing:
            continue
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        detections: list[dict[str, Any]] = []
        if detector_mode == "per_species":
            for species_id, model in detectors.items():
                conf_thr = conf_overrides.get(species_id, conf_default)
                species_dets = _predict_species_detections(
                    model=model,
                    image_bgr=img,
                    species_id=species_id,
                    conf_thr=conf_thr,
                    nms_iou=nms_iou,
                    min_area=min_area,
                    min_side=min_side,
                    max_area_frac=max_area_frac,
                    max_side_frac=max_side_frac,
                    max_per_species=max_per_species,
                )
                if agreement_enabled and species_dets:
                    agreement_conf_thr = max(agreement_conf_min, conf_thr * agreement_conf_scale)
                    agreed_dets: list[dict[str, Any]] = []
                    for det in species_dets:
                        det_box = tuple(float(v) for v in det["bbox_xyxy"])
                        agreement_iou = _agreement_iou_for_detection(
                            model=model,
                            image_bgr=img,
                            det_box_xyxy=det_box,
                            species_id=species_id,
                            conf_thr=agreement_conf_thr,
                            nms_iou=nms_iou,
                            min_agreement_iou=agreement_iou_min,
                        )
                        det["agreement_iou"] = float(agreement_iou)
                        det["agreement_pass"] = bool(agreement_iou >= agreement_iou_min)
                        if agreement_required and not det["agreement_pass"]:
                            filtered_low_agreement += 1
                            continue
                        agreed_dets.append(det)
                    species_dets = agreed_dets
                detections.extend(species_dets)

            detections = _nms(detections, nms_iou)
            if max_per_image > 0 and detections:
                detections = sorted(detections, key=lambda x: float(x["confidence"]), reverse=True)[:max_per_image]
        else:
            assert shared_detector is not None
            detections = _predict_shared_detections(
                model=shared_detector,
                image_bgr=img,
                conf_thr=conf_default,
                nms_iou=nms_iou,
                min_area=min_area,
                min_side=min_side,
                max_area_frac=max_area_frac,
                max_side_frac=max_side_frac,
                max_per_image=max_per_image,
                allowed_species=allowed_species,
            )
            if agreement_enabled and detections:
                agreement_conf_thr = max(agreement_conf_min, conf_default * agreement_conf_scale)
                agreed_dets = []
                for det in detections:
                    det_box = tuple(float(v) for v in det["bbox_xyxy"])
                    agreement_iou = _agreement_iou_for_shared_detection(
                        model=shared_detector,
                        image_bgr=img,
                        det_box_xyxy=det_box,
                        predicted_species=str(det["predicted_species"]),
                        conf_thr=agreement_conf_thr,
                        nms_iou=nms_iou,
                        min_agreement_iou=agreement_iou_min,
                        allowed_species=allowed_species,
                    )
                    det["agreement_iou"] = float(agreement_iou)
                    det["agreement_pass"] = bool(agreement_iou >= agreement_iou_min)
                    if agreement_required and not det["agreement_pass"]:
                        filtered_low_agreement += 1
                        continue
                    agreed_dets.append(det)
                detections = agreed_dets
        if not detections:
            continue

        existing_species = sorted(set(x.species_id for x in existing))
        existing_species_text = ";".join(existing_species)

        to_draw: list[tuple[tuple[float, float, float, float], str, float]] = []
        for det in detections:
            pred_species = str(det["predicted_species"])
            box_abs = tuple(float(x) for x in det["bbox_xyxy"])

            matched_same_species = False
            max_iou = 0.0
            overlapping_other_species = False
            for gt in existing:
                gt_box_abs = (
                    gt.bbox_xyxy_norm[0] * w,
                    gt.bbox_xyxy_norm[1] * h,
                    gt.bbox_xyxy_norm[2] * w,
                    gt.bbox_xyxy_norm[3] * h,
                )
                iou = _iou_xyxy(box_abs, gt_box_abs)
                max_iou = max(max_iou, iou)
                if iou >= label_iou_match and gt.species_id == pred_species:
                    matched_same_species = True
                    break
                if iou >= label_iou_match and gt.species_id != pred_species:
                    overlapping_other_species = True

            if matched_same_species:
                continue

            if not existing:
                conflict_type = "background_unlabeled"
            elif overlapping_other_species:
                conflict_type = "possible_missed_label"
            elif pred_species in existing_species:
                conflict_type = "same_species_extra"
            elif max_iou > 0:
                conflict_type = "ambiguous"
            else:
                conflict_type = "background_unlabeled"

            if conflict_type == "same_species_extra" and not include_same_species_extra:
                continue

            source_dataset = _source_dataset_from_path(image_path)
            agreement_iou = float(det.get("agreement_iou", 1.0 if not agreement_enabled else 0.0))
            agreement_pass = bool(det.get("agreement_pass", True if not agreement_enabled else False))
            likely_false_positive = bool(conflict_type in {"ambiguous", "same_species_extra"} or not agreement_pass)
            row = {
                "image_path": str(image_path),
                "predicted_species": pred_species,
                "predicted_breed": str(det["predicted_breed"]),
                "confidence": float(det["confidence"]),
                "bbox_xmin": float(box_abs[0]),
                "bbox_ymin": float(box_abs[1]),
                "bbox_xmax": float(box_abs[2]),
                "bbox_ymax": float(box_abs[3]),
                "source_dataset": source_dataset,
                "existing_labels_in_image": existing_species_text,
                "conflict_type": conflict_type,
                "agreement_iou": agreement_iou,
                "agreement_pass": int(agreement_pass),
                "likely_false_positive": int(likely_false_positive),
            }
            out_rows.append(row)
            to_draw.append((box_abs, pred_species, float(det["confidence"])))

        if to_draw and samples_written < sample_limit:
            canvas = img.copy()
            for box_abs, sp, conf in to_draw[:6]:
                x0, y0, x1, y1 = [int(v) for v in box_abs]
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(
                    canvas,
                    f"{sp}:{conf:.2f}",
                    (x0, max(16, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            out_path = visual_dir / f"sample_{samples_written:05d}.jpg"
            cv2.imwrite(str(out_path), canvas)
            samples_written += 1

    df = pd.DataFrame(out_rows)
    output_dir = Path(str(cfg.eval.mining.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_csv = output_dir / f"{_resolve_output_stem('unlabeled_candidates', output_suffix)}.csv"
    high_csv = output_dir / f"{_resolve_output_stem('unlabeled_candidates_high_priority', output_suffix)}.csv"
    summary_json = output_dir / f"{_resolve_output_stem('per_species_summary', output_suffix)}.json"

    if df.empty:
        df = pd.DataFrame(
            columns=[
                "image_path",
                "predicted_species",
                "predicted_breed",
                "confidence",
                "bbox_xmin",
                "bbox_ymin",
                "bbox_xmax",
                "bbox_ymax",
                "source_dataset",
                "existing_labels_in_image",
                "conflict_type",
                "agreement_iou",
                "agreement_pass",
                "likely_false_positive",
            ]
        )
    df.to_csv(all_csv, index=False)

    high_df = df[
        (df["confidence"] >= float(cfg.eval.mining.high_priority_min_conf))
        & (df["conflict_type"].isin(sorted(high_priority_conflict_types)))
        & (df["likely_false_positive"] == 0)
    ].copy()
    high_df = high_df.sort_values(["confidence"], ascending=False)
    max_high = int(cfg.eval.mining.high_priority_max_rows)
    if max_high > 0:
        high_df = high_df.head(max_high)
    high_df.to_csv(high_csv, index=False)

    species_counts = df["predicted_species"].value_counts().to_dict() if not df.empty else {}
    ds_counts = (
        df.groupby(["predicted_species", "source_dataset"]).size().reset_index(name="count").to_dict(orient="records")
        if not df.empty
        else []
    )
    conflict_counts = df["conflict_type"].value_counts().to_dict() if not df.empty else {}
    agreement_proxy = float((df["agreement_pass"] == 1).mean()) if not df.empty else 0.0
    likely_fp_count = int(df["likely_false_positive"].sum()) if not df.empty else 0
    confusion_hotspots = []
    if not df.empty:
        hotspot_df = (
            df[df["conflict_type"] == "possible_missed_label"]
            .groupby(["predicted_species", "existing_labels_in_image"])
            .size()
            .reset_index(name="count")
            .sort_values(["count"], ascending=False)
            .head(30)
        )
        confusion_hotspots = hotspot_df.to_dict(orient="records")

    spotcheck_path = output_dir / f"{_resolve_output_stem('spotcheck_queue', output_suffix)}.csv"
    spotcheck_precision_estimate = None
    if not high_df.empty and spotcheck_size > 0:
        n = min(spotcheck_size, len(high_df))
        spot = high_df.sample(n=n, random_state=spotcheck_seed).copy()
        spot["review_decision"] = ""  # expected values: accept | reject | unsure
        spot["review_notes"] = ""
        spot.to_csv(spotcheck_path, index=False)
        # Automated precision proxy for immediate reporting before human review.
        spotcheck_precision_estimate = float((spot["likely_false_positive"] == 0).mean()) if n > 0 else None

    summary = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ"),
        "images_scanned": int(len(image_paths)),
        "source_mode": source_mode,
        "detector_mode": detector_mode,
        "allowed_source_datasets": sorted(allowed_source_datasets),
        "excluded_source_datasets": sorted(excluded_source_datasets),
        "shard": {"shard_index": shard_index, "shard_count": shard_count},
        "total_candidates": int(len(df)),
        "high_priority_candidates": int(len(high_df)),
        "per_species_counts": species_counts,
        "per_species_dataset_counts": ds_counts,
        "conflict_type_counts": conflict_counts,
        "filtered_low_agreement_count": int(filtered_low_agreement),
        "likely_false_positive_count": likely_fp_count,
        "confusion_hotspots_top": confusion_hotspots,
        "precision_proxy_non_ambiguous_ratio": agreement_proxy,
        "spotcheck_queue_csv": str(spotcheck_path.resolve()) if spotcheck_path.exists() else "",
        "spotcheck_precision_estimate_proxy": spotcheck_precision_estimate,
        "agreement_check": {
            "enabled": agreement_enabled,
            "required": agreement_required,
            "min_iou": agreement_iou_min,
            "conf_threshold_scale": agreement_conf_scale,
            "conf_threshold_min": agreement_conf_min,
        },
        "high_priority_conflict_types": sorted(high_priority_conflict_types),
        "visual_samples_dir": str(visual_dir.resolve()),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "all_candidates_csv": str(all_csv.resolve()),
        "high_priority_csv": str(high_csv.resolve()),
        "summary_json": str(summary_json.resolve()),
        "visual_samples_dir": str(visual_dir.resolve()),
    }


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    result = run_unlabeled_mining(cfg, logger)
    logger.info("Unlabeled mining completed: %s", result)


if __name__ == "__main__":
    main()
