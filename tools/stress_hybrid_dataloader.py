"""Stress-test the hybrid Stage-1 DataLoader before launching a long YOLO run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from animal_breed.detection.hybrid_species_dataset import (
    HybridDatasetConfig,
    HybridSpeciesYoloDataset,
    SamplePlan,
    SourceImageRecord,
    SourceObject,
    read_jsonl,
)


def _load_dataset(spec_path: Path, imgsz: int, workers: int) -> HybridSpeciesYoloDataset:
    os.environ["HYBRID_YOLO_WORKERS"] = str(int(workers))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
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
        if row["split"] == "train"
    ]
    plans = [SamplePlan(**row) for row in read_jsonl(spec["train_plans_jsonl"])]
    return HybridSpeciesYoloDataset(
        class_names=list(spec["class_names"]),
        records=records,
        plans=plans,
        split="train",
        imgsz=int(imgsz),
        cfg=HybridDatasetConfig.from_mapping(spec["config"]),
        augment=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-spec", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--min-batches", type=int, default=240)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    spec_path = Path(args.hybrid_spec).resolve()
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    dataset = _load_dataset(spec_path, int(args.imgsz), int(args.workers))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=dataset.collate_fn,
        pin_memory=False,
        persistent_workers=bool(args.workers > 0),
        drop_last=False,
    )
    target_batches = min(int(args.min_batches), len(loader))
    completed = 0
    sample_kinds: dict[str, int] = {}
    max_batch_objects = 0
    max_objects_per_image = 0
    max_cls_seen = -1.0
    max_box_seen = 0.0
    try:
        for batch in loader:
            completed += 1
            cls = batch.get("cls")
            bboxes = batch.get("bboxes")
            batch_idx = batch.get("batch_idx")
            if not isinstance(cls, torch.Tensor) or cls.numel() == 0:
                raise RuntimeError(f"empty cls tensor at dataloader batch {completed}")
            if not isinstance(batch_idx, torch.Tensor) or batch_idx.numel() != cls.numel():
                raise RuntimeError(f"invalid batch_idx tensor at dataloader batch {completed}")
            if not isinstance(bboxes, torch.Tensor) or bboxes.shape[0] != cls.numel() or bboxes.shape[-1] != 4:
                raise RuntimeError(f"invalid bbox tensor at dataloader batch {completed}")
            cls_min = float(cls.min().item())
            cls_max = float(cls.max().item())
            box_min = float(bboxes.min().item())
            box_max = float(bboxes.max().item())
            batch_idx_min = float(batch_idx.min().item())
            batch_idx_max = float(batch_idx.max().item())
            if cls_min < 0 or cls_max >= len(dataset.class_names):
                raise RuntimeError(
                    f"invalid class ids at batch {completed}: {cls_min}..{cls_max} "
                    f"for {len(dataset.class_names)} classes"
                )
            if batch_idx_min < 0 or batch_idx_max >= int(args.batch):
                raise RuntimeError(f"invalid batch_idx at batch {completed}: {batch_idx_min}..{batch_idx_max}")
            if box_min < -1e-5 or box_max > 1.00001:
                raise RuntimeError(f"invalid normalized boxes at batch {completed}: {box_min}..{box_max}")
            counts = torch.bincount(batch_idx.to(torch.long), minlength=int(args.batch))
            batch_objects = int(cls.numel())
            max_batch_objects = max(max_batch_objects, batch_objects)
            max_objects_per_image = max(max_objects_per_image, int(counts.max().item()))
            max_cls_seen = max(max_cls_seen, cls_max)
            max_box_seen = max(max_box_seen, box_max)
            kinds = batch.get("sample_kind", []) if isinstance(batch, dict) else []
            if isinstance(kinds, list):
                for kind in kinds:
                    sample_kinds[str(kind)] = sample_kinds.get(str(kind), 0) + 1
            if completed == 1 or completed % 25 == 0:
                elapsed = time.time() - started
                print(
                    f"[hybrid-stress] batch={completed}/{target_batches} "
                    f"elapsed={elapsed:.1f}s sample_kinds={sample_kinds}",
                    flush=True,
                )
            if completed >= target_batches:
                break
    finally:
        if hasattr(loader, "_iterator") and loader._iterator is not None:  # noqa: SLF001
            loader._iterator._shutdown_workers()  # noqa: SLF001

    report = {
        "hybrid_spec": str(spec_path),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "imgsz": int(args.imgsz),
        "dataset_len": len(dataset),
        "loader_batches": len(loader),
        "target_batches": target_batches,
        "completed_batches": completed,
        "sample_kinds": sample_kinds,
        "max_batch_objects": max_batch_objects,
        "max_objects_per_image": max_objects_per_image,
        "max_cls_seen": max_cls_seen,
        "max_box_seen": max_box_seen,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    if completed < target_batches:
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
