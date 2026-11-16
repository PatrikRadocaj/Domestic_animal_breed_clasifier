"""Hybrid species-aware data preparation and custom Ultralytics trainer wiring."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import json
import os
from typing import Any

import cv2
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER

from animal_breed.data.label_mapping import CrossDatasetLabelMap
from animal_breed.data.taxonomy import load_taxonomy
from animal_breed.detection.hybrid_species_dataset import (
    HybridDatasetConfig,
    HybridSpeciesYoloDataset,
    SamplePlan,
    SourceImageRecord,
    SourceObject,
    build_sample_metadata_rows,
    build_hybrid_sample_plans,
    read_jsonl,
    write_jsonl,
)
from animal_breed.detection.yolo_source_index import discover_yolo_sources, resolve_image_for_label
from animal_breed.utils.logging import setup_logging


def _normalize_registry_species(x: str) -> str:
    return str(x).strip().lower().replace("-", "_")


def _load_registry_species(registry_path: str | Path) -> dict[str, str]:
    p = Path(registry_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out: dict[str, str] = {}
    if "dataset_id" not in df.columns or "species_value" not in df.columns:
        return out
    for row in df.to_dict(orient="records"):
        ds = str(row.get("dataset_id", "")).strip()
        if ds:
            out[ds] = _normalize_registry_species(str(row.get("species_value", "")))
    return out


def _resolve_species_ids(cfg: DictConfig, label_map: CrossDatasetLabelMap) -> list[str]:
    taxonomy = load_taxonomy(cfg.data.taxonomy_path)
    species_ids = sorted(
        set([row["species_id"] for row in taxonomy.species_records()]) | set(label_map.species_canonical.keys())
    )
    wanted = list(getattr(cfg.data.species_aware.hybrid, "target_species", []) or [])
    if wanted:
        keep = {str(x).strip().lower() for x in wanted if str(x).strip()}
        species_ids = [species_id for species_id in species_ids if species_id in keep]
    return species_ids


def _infer_subspecies(species_id: str, source_class_name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(source_class_name))
    slug = slug.strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    if not slug or slug == species_id or slug.isdigit():
        return "unknown"
    return slug


def _parse_dataset_class_maps(
    datasets,
    label_map: CrossDatasetLabelMap,
    target_species_ids: list[str],
) -> tuple[dict[str, dict[int, tuple[str, str]]], dict[str, int], dict[str, int]]:
    target = set(target_species_ids)
    ds_idx_to_species: dict[str, dict[int, tuple[str, str]]] = {}
    ds_manual_count: dict[str, int] = {}
    ds_unmapped_count: dict[str, int] = {}

    for ds in datasets:
        idx_map: dict[int, tuple[str, str]] = {}
        manual = 0
        unmapped = 0
        for idx, cls_name in enumerate(ds.names):
            res = label_map.resolve_species(cls_name, ds.dataset_id)
            if res.status == "resolved" and res.canonical_id is not None and res.canonical_id in target:
                idx_map[idx] = (res.canonical_id, str(cls_name))
            elif res.status == "manual_review":
                manual += 1
            else:
                unmapped += 1
        ds_idx_to_species[ds.dataset_id] = idx_map
        ds_manual_count[ds.dataset_id] = manual
        ds_unmapped_count[ds.dataset_id] = unmapped
    return ds_idx_to_species, ds_manual_count, ds_unmapped_count


def _collect_source_records(
    datasets,
    ds_idx_to_species: dict[str, dict[int, tuple[str, str]]],
    target_species_ids: list[str],
) -> dict[str, list[SourceImageRecord]]:
    class_to_idx = {species_id: idx for idx, species_id in enumerate(target_species_ids)}
    records_by_split: dict[str, list[SourceImageRecord]] = defaultdict(list)

    for ds in datasets:
        idx_map = ds_idx_to_species.get(ds.dataset_id, {})
        if not idx_map:
            continue
        for split, label_dir in ds.split_label_dirs.items():
            image_dir = ds.split_image_dirs.get(split)
            if image_dir is None or not label_dir.exists():
                continue
            split_key = "val" if split == "valid" else split
            for label_path in label_dir.rglob("*.txt"):
                image_path = resolve_image_for_label(label_path, image_dir)
                if image_path is None:
                    continue
                objects: list[SourceObject] = []
                raw_species_present: set[str] = set()
                had_unmapped_lines = False
                for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cls_idx = int(float(parts[0]))
                    except ValueError:
                        had_unmapped_lines = True
                        continue
                    mapping = idx_map.get(cls_idx)
                    if mapping is None:
                        had_unmapped_lines = True
                        continue
                    species_id, source_class_name = mapping
                    raw_species_present.add(species_id)
                    objects.append(
                        SourceObject(
                            class_id=int(class_to_idx[species_id]),
                            species_id=species_id,
                            subspecies=_infer_subspecies(species_id, source_class_name),
                            source_class_name=source_class_name,
                            bbox_xywhn=[float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                        )
                    )
                if not objects or had_unmapped_lines:
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                height, width = image.shape[:2]
                pure_species_id = next(iter(raw_species_present)) if len(raw_species_present) == 1 else None
                record = SourceImageRecord(
                    record_id=f"{ds.dataset_id}:{label_path.stem}",
                    dataset_id=ds.dataset_id,
                    split=split_key,
                    image_path=str(image_path.resolve()),
                    width=int(width),
                    height=int(height),
                    source_species_ids=sorted(raw_species_present),
                    objects=objects,
                    pure_species_id=pure_species_id,
                )
                records_by_split[split_key].append(record)
    return records_by_split


def _write_virtual_index(
    plans: list[SamplePlan],
    records_by_id: dict[str, SourceImageRecord],
    output_path: str | Path,
) -> None:
    rows = []
    for plan in plans:
        if plan.sample_kind == "real" and plan.record_id is not None:
            rows.append(records_by_id[plan.record_id].image_path)
        else:
            rows.append(f"hybrid://{plan.split}/{plan.sample_id}.jpg")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def build_hybrid_species_spec(cfg: DictConfig, logger) -> dict[str, Any]:
    hybrid_cfg = cfg.data.species_aware.hybrid
    train_cfg = hybrid_cfg.train
    os.environ["HYBRID_YOLO_WORKERS"] = str(int(getattr(train_cfg, "workers", 0)))
    output_dir = Path(str(hybrid_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = CrossDatasetLabelMap.from_file(str(cfg.data.cross_dataset_label_map_path))
    species_ids = _resolve_species_ids(cfg, label_map)
    datasets = discover_yolo_sources(Path(str(cfg.data.yolo_sources_root)))
    logger.info("Hybrid pipeline discovered %d YOLO source datasets", len(datasets))

    ds_idx_to_species, _, _ = _parse_dataset_class_maps(datasets, label_map, species_ids)
    records_by_split = _collect_source_records(datasets, ds_idx_to_species, species_ids)
    train_records = records_by_split.get("train", [])
    val_records = records_by_split.get("val", [])
    test_records = records_by_split.get("test", [])

    pure_train_by_species: dict[str, list[SourceImageRecord]] = {species_id: [] for species_id in species_ids}
    for record in train_records:
        if record.pure_species_id in pure_train_by_species:
            pure_train_by_species[record.pure_species_id].append(record)

    active_species_ids = [species_id for species_id in species_ids if pure_train_by_species.get(species_id)]
    dropped_species = [species_id for species_id in species_ids if species_id not in active_species_ids]
    if dropped_species:
        logger.warning(
            "Hybrid dataset excludes species without pure train images: %s",
            ", ".join(dropped_species),
        )
    active_idx = {species_id: idx for idx, species_id in enumerate(active_species_ids)}

    def _filter_records(records: list[SourceImageRecord]) -> list[SourceImageRecord]:
        filtered: list[SourceImageRecord] = []
        for record in records:
            kept_objects = []
            kept_species = set()
            for obj in record.objects:
                if obj.species_id not in active_idx:
                    continue
                kept_objects.append(
                    SourceObject(
                        class_id=int(active_idx[obj.species_id]),
                        species_id=obj.species_id,
                        subspecies=obj.subspecies,
                        source_class_name=obj.source_class_name,
                        bbox_xywhn=list(obj.bbox_xywhn),
                    )
                )
                kept_species.add(obj.species_id)
            if not kept_objects:
                continue
            pure_species_id = record.pure_species_id if record.pure_species_id in active_idx else None
            filtered.append(
                SourceImageRecord(
                    record_id=record.record_id,
                    dataset_id=record.dataset_id,
                    split=record.split,
                    image_path=record.image_path,
                    width=record.width,
                    height=record.height,
                    source_species_ids=sorted(kept_species),
                    objects=kept_objects,
                    pure_species_id=pure_species_id,
                )
            )
        return filtered

    train_records = _filter_records(train_records)
    val_records = _filter_records(val_records)
    test_records = _filter_records(test_records)
    pure_train_by_species = {species_id: [] for species_id in active_species_ids}
    for record in train_records:
        if record.pure_species_id in pure_train_by_species:
            pure_train_by_species[record.pure_species_id].append(record)

    ds_cfg = HybridDatasetConfig.from_mapping(OmegaConf.to_container(hybrid_cfg, resolve=True))
    train_plans = build_hybrid_sample_plans(active_species_ids, pure_train_by_species, train_records, "train", ds_cfg)
    val_plans = build_hybrid_sample_plans(active_species_ids, {}, val_records, "val", ds_cfg)
    test_plans = build_hybrid_sample_plans(active_species_ids, {}, test_records, "test", ds_cfg)

    source_rows = []
    all_records = train_records + val_records + test_records
    for record in all_records:
        source_rows.append(
            {
                "record_id": record.record_id,
                "dataset_id": record.dataset_id,
                "split": record.split,
                "image_path": record.image_path,
                "width": record.width,
                "height": record.height,
                "source_species_ids": record.source_species_ids,
                "pure_species_id": record.pure_species_id,
                "objects": [asdict(obj) for obj in record.objects],
            }
        )

    train_plan_rows = [asdict(plan) for plan in train_plans]
    val_plan_rows = [asdict(plan) for plan in val_plans]
    test_plan_rows = [asdict(plan) for plan in test_plans]

    source_records_path = output_dir / "source_records.jsonl"
    train_plans_path = output_dir / "train_plans.jsonl"
    val_plans_path = output_dir / "val_plans.jsonl"
    test_plans_path = output_dir / "test_plans.jsonl"
    train_metadata_index_path = output_dir / "train_metadata_index.jsonl"
    val_metadata_index_path = output_dir / "val_metadata_index.jsonl"
    test_metadata_index_path = output_dir / "test_metadata_index.jsonl"
    write_jsonl(source_records_path, source_rows)
    write_jsonl(train_plans_path, train_plan_rows)
    write_jsonl(val_plans_path, val_plan_rows)
    write_jsonl(test_plans_path, test_plan_rows)
    if bool(ds_cfg.train_metadata_write):
        write_jsonl(train_metadata_index_path, build_sample_metadata_rows(train_plans))
        write_jsonl(val_metadata_index_path, build_sample_metadata_rows(val_plans))
        write_jsonl(test_metadata_index_path, build_sample_metadata_rows(test_plans))

    records_by_id = {record.record_id: record for record in all_records}
    _write_virtual_index(train_plans, records_by_id, output_dir / "train.txt")
    _write_virtual_index(val_plans, records_by_id, output_dir / "val.txt")
    _write_virtual_index(test_plans, records_by_id, output_dir / "test.txt")

    train_mix_counts = pd.Series([plan.sample_kind for plan in train_plans]).value_counts().sort_index().to_dict()
    train_mix_counts = {str(k): int(v) for k, v in train_mix_counts.items()}

    metadata = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ"),
        "output_dir": str(output_dir.resolve()),
        "source_records_jsonl": str(source_records_path.resolve()),
        "train_plans_jsonl": str(train_plans_path.resolve()),
        "val_plans_jsonl": str(val_plans_path.resolve()),
        "test_plans_jsonl": str(test_plans_path.resolve()),
        "train_metadata_index_jsonl": str(train_metadata_index_path.resolve()) if ds_cfg.train_metadata_write else "",
        "val_metadata_index_jsonl": str(val_metadata_index_path.resolve()) if ds_cfg.train_metadata_write else "",
        "test_metadata_index_jsonl": str(test_metadata_index_path.resolve()) if ds_cfg.train_metadata_write else "",
        "class_names": active_species_ids,
        "nc": len(active_species_ids),
        "config": asdict(ds_cfg),
        "counts": {
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records": len(test_records),
            "train_plans": len(train_plans),
            "val_plans": len(val_plans),
            "test_plans": len(test_plans),
            "train_plan_mix": train_mix_counts,
            "pure_train_records_by_species": {species_id: len(records) for species_id, records in pure_train_by_species.items()},
            "excluded_species_without_pure_train_images": dropped_species,
        },
    }
    spec_path = output_dir / "hybrid_dataset_spec.json"
    spec_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Hybrid dataset spec written to %s", spec_path)
    return metadata


class HybridDetectionTrainer(DetectionTrainer):
    """Ultralytics detection trainer backed by an on-the-fly hybrid dataset."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        self.hybrid_spec_path = str(overrides.pop("hybrid_spec_path"))
        super().__init__(cfg=cfg or DEFAULT_CFG, overrides=overrides, _callbacks=_callbacks)
        self.add_callback("on_train_epoch_start", self._set_dataset_epoch)
        self._fallback_batch_count = 0

    def _set_dataset_epoch(self, trainer) -> None:
        for loader_name in ("train_loader", "test_loader"):
            loader = getattr(trainer, loader_name, None)
            if loader is not None and hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(getattr(trainer, "epoch", 0))

    def get_dataset(self):
        spec_path = Path(self.hybrid_spec_path)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.hybrid_spec = spec
        names = {idx: name for idx, name in enumerate(spec["class_names"])}
        return {
            "train": "hybrid://train",
            "val": "hybrid://val",
            "test": "hybrid://test",
            "names": names,
            "nc": len(names),
            "channels": 3,
        }

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        split = "train" if mode == "train" else "val"
        spec = self.hybrid_spec
        cfg = HybridDatasetConfig.from_mapping(spec["config"])
        records = [
            SourceImageRecord(
                record_id=row["record_id"],
                dataset_id=row["dataset_id"],
                split=row["split"],
                image_path=row["image_path"],
                width=int(row["width"]),
                height=int(row["height"]),
                source_species_ids=list(row["source_species_ids"]),
                pure_species_id=row.get("pure_species_id"),
                objects=[SourceObject(**obj) for obj in row["objects"]],
            )
            for row in read_jsonl(spec["source_records_jsonl"])
            if row["split"] == split or (split == "val" and row["split"] == "test")
        ]
        plan_path = spec["train_plans_jsonl"] if mode == "train" else spec["val_plans_jsonl"]
        plans = [SamplePlan(**row) for row in read_jsonl(plan_path)]
        return HybridSpeciesYoloDataset(
            class_names=list(spec["class_names"]),
            records=records,
            plans=plans,
            split=split,
            imgsz=int(self.args.imgsz),
            cfg=cfg,
            augment=mode == "train",
        )

    def _batch_has_valid_targets(self, batch: dict) -> bool:
        cls = batch.get("cls")
        bboxes = batch.get("bboxes")
        batch_idx = batch.get("batch_idx")
        if not isinstance(cls, torch.Tensor) or not isinstance(bboxes, torch.Tensor) or not isinstance(batch_idx, torch.Tensor):
            return False
        if cls.numel() == 0 or bboxes.numel() == 0 or batch_idx.numel() == 0:
            return False
        if bboxes.ndim != 2 or bboxes.shape[1] != 4 or bboxes.shape[0] != cls.numel() or batch_idx.numel() != cls.numel():
            return False
        if torch.isnan(cls).any() or torch.isnan(bboxes).any() or torch.isnan(batch_idx).any():
            return False
        nc = int(getattr(self.model, "nc", len(getattr(self, "names", [])) or len(self.data.get("names", {}))))
        if int(cls.min().item()) < 0 or int(cls.max().item()) >= nc:
            return False
        if int(batch_idx.min().item()) < 0 or int(batch_idx.max().item()) >= int(batch["img"].shape[0]):
            return False
        return True

    def _make_fallback_batch(self, original_batch: dict) -> dict:
        dataset = getattr(getattr(self, "train_loader", None), "dataset", None)
        if dataset is None or not hasattr(dataset, "_render_fallback_real"):
            raise RuntimeError("Hybrid trainer received an invalid target batch and no fallback dataset is available.")
        batch_size = int(original_batch["img"].shape[0]) if isinstance(original_batch.get("img"), torch.Tensor) else int(self.batch_size)
        samples = [dataset._render_fallback_real(i, "trainer_invalid_target_batch") for i in range(batch_size)]
        return dataset.collate_fn(samples)

    def preprocess_batch(self, batch: dict) -> dict:
        if not self._batch_has_valid_targets(batch):
            self._fallback_batch_count += 1
            LOGGER.warning(
                "Hybrid trainer replaced invalid target batch with fallback real samples "
                "(count=%d, img_shape=%s, cls_shape=%s, bbox_shape=%s, batch_idx_shape=%s)",
                self._fallback_batch_count,
                tuple(batch["img"].shape) if isinstance(batch.get("img"), torch.Tensor) else None,
                tuple(batch["cls"].shape) if isinstance(batch.get("cls"), torch.Tensor) else None,
                tuple(batch["bboxes"].shape) if isinstance(batch.get("bboxes"), torch.Tensor) else None,
                tuple(batch["batch_idx"].shape) if isinstance(batch.get("batch_idx"), torch.Tensor) else None,
            )
            batch = self._make_fallback_batch(batch)
        if isinstance(batch.get("bboxes"), torch.Tensor):
            batch["bboxes"] = batch["bboxes"].clamp_(0.0, 1.0)
        return super().preprocess_batch(batch)

    def plot_training_labels(self) -> None:
        """Skip Ultralytics' default label plot for the custom hybrid dataset.

        Their implementation assumes `self.train_loader.dataset.labels` exists,
        which is true for the stock YOLO dataset but not for our on-the-fly
        hybrid dataset wrapper.
        """
        return

    def final_eval(self):
        return


def train_hybrid_species_detector(cfg: DictConfig, logger, spec: dict[str, Any]) -> dict[str, Any]:
    train_cfg = cfg.data.species_aware.hybrid.train
    model = YOLO(str(train_cfg.base_model))
    device = getattr(train_cfg, "device", None)
    requested_workers = int(train_cfg.workers)
    worker_count = max(0, requested_workers)
    os.environ["HYBRID_YOLO_WORKERS"] = str(worker_count)
    cache_value = getattr(train_cfg, "cache", False)
    if isinstance(cache_value, str):
        cache_value = cache_value.strip()
        if not cache_value:
            cache_value = False
    logger.info("hybrid stage1 using dataloader workers=%d", worker_count)
    results = model.train(
        trainer=HybridDetectionTrainer,
        data="hybrid://species",
        hybrid_spec_path=str(Path(spec["output_dir"]) / "hybrid_dataset_spec.json"),
        model=str(train_cfg.base_model),
        imgsz=int(train_cfg.imgsz),
        epochs=int(train_cfg.epochs),
        batch=int(train_cfg.batch),
        workers=worker_count,
        amp=bool(train_cfg.amp),
        cache=cache_value,
        device=str(device) if device is not None and str(device).strip() else None,
        seed=int(getattr(cfg, "seed", 42)),
        project=str(train_cfg.project),
        name=str(train_cfg.name),
        exist_ok=True,
        deterministic=False,
        plots=False,
    )
    return {
        "spec_path": str((Path(spec["output_dir"]) / "hybrid_dataset_spec.json").resolve()),
        "save_dir": str(Path(str(getattr(results, "save_dir", train_cfg.project))).resolve()),
        "results": str(results),
    }


def build_and_optionally_train_hybrid(cfg: DictConfig, logger) -> dict[str, Any]:
    spec = build_hybrid_species_spec(cfg, logger)
    debug_report: dict[str, Any] = {}
    if int(cfg.data.species_aware.hybrid.debug_subset_count) > 0:
        train_records = [
            SourceImageRecord(
                record_id=row["record_id"],
                dataset_id=row["dataset_id"],
                split=row["split"],
                image_path=row["image_path"],
                width=int(row["width"]),
                height=int(row["height"]),
                source_species_ids=list(row["source_species_ids"]),
                pure_species_id=row.get("pure_species_id"),
                objects=[SourceObject(**obj) for obj in row["objects"]],
            )
            for row in read_jsonl(spec["source_records_jsonl"])
            if row["split"] == "train"
        ]
        plans = [SamplePlan(**row) for row in read_jsonl(spec["train_plans_jsonl"])]
        ds = HybridSpeciesYoloDataset(
            class_names=list(spec["class_names"]),
            records=train_records,
            plans=plans,
            split="train",
            imgsz=int(cfg.data.species_aware.hybrid.train.imgsz),
            cfg=HybridDatasetConfig.from_mapping(spec["config"]),
            augment=True,
        )
        debug_report = ds.export_debug_subset(
            Path(spec["output_dir"]) / "debug_subset",
            count=int(cfg.data.species_aware.hybrid.debug_subset_count),
        )
        logger.info("Hybrid debug subset written to %s", debug_report.get("output_dir", ""))

    train_report: dict[str, Any] = {}
    if bool(cfg.data.species_aware.hybrid.train.enabled):
        train_report = train_hybrid_species_detector(cfg, logger, spec)
        logger.info("Hybrid training completed: %s", train_report)

    return {
        "spec": spec,
        "debug_subset": debug_report,
        "training": train_report,
    }


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    out = build_and_optionally_train_hybrid(cfg, logger)
    logger.info("Hybrid species-aware pipeline completed: %s", out)


if __name__ == "__main__":
    main()
