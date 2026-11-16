"""Species-aware training data filtering and optional detector training."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig
import yaml
from ultralytics import YOLO

from animal_breed.data.label_mapping import CrossDatasetLabelMap
from animal_breed.data.taxonomy import load_taxonomy
from animal_breed.detection.yolo_source_index import discover_yolo_sources, resolve_image_for_label
from animal_breed.utils.logging import setup_logging


@dataclass(frozen=True)
class SpeciesEligibilityRow:
    species_id: str
    dataset_id: str
    status: str  # included | excluded
    reason: str
    mapped_class_names: str
    mapped_class_indices: str


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _slugify_filename_component(value: str) -> str:
    text = str(value).strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _make_output_stem(dataset_id: str, label_path: Path) -> str:
    dataset_slug = _slugify_filename_component(dataset_id)[:24] or "dataset"
    label_slug = _slugify_filename_component(label_path.stem)[:40] or "sample"
    digest = hashlib.sha1(str(label_path).encode("utf-8")).hexdigest()[:12]
    return f"{dataset_slug}__{label_slug}__{digest}"


def _split_counts(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for split in ("train", "val", "test"):
        labels = root / "labels" / split
        out[split] = sum(1 for _ in labels.rglob("*.txt")) if labels.exists() else 0
    return out


def _extract_yolo_train_metrics(weights_run_dir: Path) -> dict[str, float | int]:
    csv_path = weights_run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    if df.empty:
        return {}
    last = df.iloc[-1].to_dict()
    metrics: dict[str, float | int] = {}
    wanted = {
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    }
    for key, value in last.items():
        if key not in wanted:
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    if "epoch" in metrics:
        try:
            metrics["epoch"] = int(metrics["epoch"])
        except (TypeError, ValueError):
            pass
    return metrics


def _copy_weights_to_canonical(run_dir: Path, canonical_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    src_best = run_dir / "weights" / "best.pt"
    src_last = run_dir / "weights" / "last.pt"
    dst_weights = canonical_root / "weights"
    dst_weights.mkdir(parents=True, exist_ok=True)
    if src_best.exists():
        dst_best = dst_weights / "best.pt"
        shutil.copy2(src_best, dst_best)
        out["best"] = str(dst_best.resolve())
    if src_last.exists():
        dst_last = dst_weights / "last.pt"
        shutil.copy2(src_last, dst_last)
        out["last"] = str(dst_last.resolve())
    return out


def _build_shared_species_dataset(
    datasets,
    ds_species_map: dict[str, dict[str, list[int]]],
    species_ids: list[str],
    output_dir: Path,
    max_images_per_species_per_split: int,
) -> dict[str, Any]:
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    species_to_class_idx = {species_id: idx for idx, species_id in enumerate(species_ids)}
    species_split_counter: dict[str, Counter[str]] = {s: Counter() for s in species_ids}
    total_split_counter: Counter[str] = Counter()

    for ds in datasets:
        per_split_species_written: dict[str, Counter[str]] = defaultdict(Counter)
        dataset_species_map = ds_species_map.get(ds.dataset_id, {})
        if not dataset_species_map:
            continue
        class_idx_to_species = {
            int(class_idx): species_id
            for species_id, class_indices in dataset_species_map.items()
            if species_id in species_to_class_idx
            for class_idx in class_indices
        }
        if not class_idx_to_species:
            continue
        for split, label_dir in ds.split_label_dirs.items():
            image_dir = ds.split_image_dirs.get(split)
            if image_dir is None or not label_dir.exists():
                continue
            for label_path in label_dir.rglob("*.txt"):
                image_path = resolve_image_for_label(label_path, image_dir)
                if image_path is None:
                    continue
                raw_kept: list[tuple[str, str]] = []
                for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cls_idx = int(float(parts[0]))
                    except ValueError:
                        continue
                    species_id = class_idx_to_species.get(cls_idx)
                    if species_id is None:
                        continue
                    if max_images_per_species_per_split > 0 and (
                        species_split_counter[species_id][split] >= max_images_per_species_per_split
                    ):
                        continue
                    raw_kept.append((species_id, " ".join(parts[1:])))
                if not raw_kept:
                    continue

                kept_lines: list[str] = []
                line_species: set[str] = set()
                for species_id, coords in raw_kept:
                    kept_lines.append(f"{species_to_class_idx[species_id]} {coords}")
                    line_species.add(species_id)
                if not kept_lines:
                    continue

                out_name = _make_output_stem(ds.dataset_id, label_path)
                out_image = output_dir / "images" / split / f"{out_name}{image_path.suffix.lower()}"
                out_label = output_dir / "labels" / split / f"{out_name}.txt"
                _safe_link_or_copy(image_path, out_image)
                out_label.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
                total_split_counter[split] += 1
                for species_id in line_species:
                    species_split_counter[species_id][split] += 1
                    per_split_species_written[split][species_id] += 1

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": species_ids,
        "nc": len(species_ids),
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return {
        "output_dir": str(output_dir.resolve()),
        "species_to_class_idx": species_to_class_idx,
        "image_counts_by_split": dict(total_split_counter),
        "species_image_counts_by_split": {species_id: dict(counter) for species_id, counter in species_split_counter.items()},
        "total_images": int(sum(total_split_counter.values())),
    }


def _train_shared_species_detector(cfg: DictConfig, dataset_dir: Path) -> dict[str, Any]:
    shared_cfg = cfg.data.species_aware.shared_train
    weights_root = Path(str(shared_cfg.weights_root))
    weights_root.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(shared_cfg.base_model))
    epochs = int(shared_cfg.epochs)
    results = model.train(
        data=str(dataset_dir / "data.yaml"),
        imgsz=int(shared_cfg.imgsz),
        epochs=epochs,
        batch=int(shared_cfg.batch),
        workers=int(shared_cfg.workers),
        project=str(weights_root.parent),
        name=weights_root.name,
        exist_ok=True,
        amp=bool(shared_cfg.amp),
        cache=bool(shared_cfg.cache),
    )
    run_dir = Path(str(getattr(results, "save_dir", ""))).resolve()
    if not run_dir.exists():
        run_dir = weights_root.resolve()
    copied_weights = _copy_weights_to_canonical(run_dir, weights_root.resolve())
    metrics = _extract_yolo_train_metrics(run_dir)
    result_summary = {}
    if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
        for k, v in results.results_dict.items():
            try:
                result_summary[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return {
        "weights_dir": str(weights_root.resolve()),
        "run_dir": str(run_dir),
        "result_summary": result_summary,
        "epochs": epochs,
        "metrics": metrics,
        "weights_files": copied_weights,
    }


def _normalize_registry_species(x: str) -> str:
    return str(x).strip().lower().replace("-", "_")


def _load_registry_species(registry_path: str | Path) -> dict[str, str]:
    p = Path(registry_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out: dict[str, str] = {}
    if "dataset_id" not in df.columns:
        return out
    if "species_value" not in df.columns:
        return out
    for row in df.to_dict(orient="records"):
        ds = str(row.get("dataset_id", "")).strip()
        if not ds:
            continue
        out[ds] = _normalize_registry_species(str(row.get("species_value", "")))
    return out


def build_species_aware_training(cfg: DictConfig, logger) -> dict[str, Any]:
    label_map = CrossDatasetLabelMap.from_file(str(cfg.data.cross_dataset_label_map_path))
    taxonomy = load_taxonomy(cfg.data.taxonomy_path)
    species_ids = sorted(
        set([row["species_id"] for row in taxonomy.species_records()]) | set(label_map.species_canonical.keys())
    )
    target_species_cfg = list(getattr(cfg.data.species_aware, "target_species", []) or [])
    if target_species_cfg:
        wanted = {str(x).strip().lower() for x in target_species_cfg if str(x).strip()}
        species_ids = [s for s in species_ids if s in wanted]

    yolo_root = Path(str(cfg.data.yolo_sources_root))
    processed_root = Path(str(cfg.data.species_aware.output_root))
    processed_root.mkdir(parents=True, exist_ok=True)
    reports_root = Path(str(cfg.data.species_aware.report_root))
    reports_root.mkdir(parents=True, exist_ok=True)
    registry_species = _load_registry_species(cfg.data.species_aware.registry_metadata_path)

    datasets = discover_yolo_sources(yolo_root)
    logger.info("Discovered %d YOLO source datasets in %s", len(datasets), yolo_root)

    # Dataset -> mapped species class indices
    ds_species_map: dict[str, dict[str, list[int]]] = {}
    ds_manual_count: dict[str, int] = {}
    ds_unmapped_count: dict[str, int] = {}
    for ds in datasets:
        mapped: dict[str, list[int]] = defaultdict(list)
        manual = 0
        unmapped = 0
        for idx, cls_name in enumerate(ds.names):
            res = label_map.resolve_species(cls_name, ds.dataset_id)
            if res.status == "resolved" and res.canonical_id is not None:
                mapped[res.canonical_id].append(idx)
            elif res.status == "manual_review":
                manual += 1
            else:
                unmapped += 1
        ds_species_map[ds.dataset_id] = dict(mapped)
        ds_manual_count[ds.dataset_id] = manual
        ds_unmapped_count[ds.dataset_id] = unmapped

    eligibility_rows: list[SpeciesEligibilityRow] = []
    species_split_counter: dict[str, Counter[str]] = {s: Counter() for s in species_ids}
    species_included: dict[str, list[str]] = {s: [] for s in species_ids}
    species_excluded: dict[str, list[dict[str, str]]] = {s: [] for s in species_ids}

    for species_id in species_ids:
        for ds in datasets:
            mapped_indices = ds_species_map.get(ds.dataset_id, {}).get(species_id, [])
            if mapped_indices:
                status = "included"
                reason = "explicit_species_label_in_dataset"
                if ds_manual_count.get(ds.dataset_id, 0) > 0:
                    reason = "explicit_species_label_in_dataset_with_other_manual_review_classes"
                species_included[species_id].append(ds.dataset_id)
            else:
                status = "excluded"
                reg_sp = registry_species.get(ds.dataset_id, "")
                if reg_sp and reg_sp not in {species_id, "multi"}:
                    reason = f"registry_species_is_{reg_sp}_not_{species_id}"
                elif ds_unmapped_count.get(ds.dataset_id, 0) > 0 and reg_sp in {species_id, "multi"}:
                    reason = "species_not_explicitly_mapped_or_unmapped_labels_present"
                else:
                    reason = "no_explicit_species_label"
                species_excluded[species_id].append({"dataset_id": ds.dataset_id, "reason": reason})
            mapped_names = [ds.names[i] for i in mapped_indices]
            eligibility_rows.append(
                SpeciesEligibilityRow(
                    species_id=species_id,
                    dataset_id=ds.dataset_id,
                    status=status,
                    reason=reason,
                    mapped_class_names=";".join(mapped_names),
                    mapped_class_indices=";".join(str(i) for i in mapped_indices),
                )
            )

    # Build per-species YOLO datasets
    max_images_per_split = int(getattr(cfg.data.species_aware, "max_images_per_split", 0))
    for species_id in species_ids:
        species_dir = processed_root / species_id
        for split in ("train", "val", "test"):
            (species_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (species_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        per_split_written: Counter[str] = Counter()
        for ds in datasets:
            cls_indices = set(ds_species_map.get(ds.dataset_id, {}).get(species_id, []))
            if not cls_indices:
                continue
            for split, label_dir in ds.split_label_dirs.items():
                if max_images_per_split > 0 and per_split_written[split] >= max_images_per_split:
                    continue
                image_dir = ds.split_image_dirs.get(split)
                if image_dir is None or not label_dir.exists():
                    continue
                for label_path in label_dir.rglob("*.txt"):
                    if max_images_per_split > 0 and per_split_written[split] >= max_images_per_split:
                        break
                    image_path = resolve_image_for_label(label_path, image_dir)
                    if image_path is None:
                        continue
                    kept_lines: list[str] = []
                    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        try:
                            cls_idx = int(float(parts[0]))
                        except ValueError:
                            continue
                        if cls_idx not in cls_indices:
                            continue
                        kept_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                    if not kept_lines:
                        continue
                    out_name = _make_output_stem(ds.dataset_id, label_path)
                    out_image = species_dir / "images" / split / f"{out_name}{image_path.suffix.lower()}"
                    out_label = species_dir / "labels" / split / f"{out_name}.txt"
                    _safe_link_or_copy(image_path, out_image)
                    out_label.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
                    species_split_counter[species_id][split] += 1
                    per_split_written[split] += 1

        data_yaml = {
            "path": str(species_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": [species_id],
            "nc": 1,
        }
        (species_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    # Reporting
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    eligibility_csv = reports_root / f"species_dataset_eligibility_{now}.csv"
    pd.DataFrame([asdict(x) for x in eligibility_rows]).to_csv(eligibility_csv, index=False)

    train_report = {
        "timestamp_utc": now,
        "yolo_sources_root": str(yolo_root.resolve()),
        "processed_root": str(processed_root.resolve()),
        "species": {},
    }
    for s in species_ids:
        train_report["species"][s] = {
            "included_dataset_ids": sorted(set(species_included[s])),
            "excluded_dataset_ids": species_excluded[s],
            "image_counts_by_split": dict(species_split_counter[s]),
            "total_images": int(sum(species_split_counter[s].values())),
        }
    train_report_path = reports_root / f"species_aware_training_report_{now}.json"
    train_report_path.write_text(json.dumps(train_report, indent=2), encoding="utf-8")

    shared_cfg = getattr(cfg.data.species_aware, "shared_train", None)
    shared_enabled = bool(getattr(shared_cfg, "enabled", False)) if shared_cfg is not None else False
    shared_dataset_report: dict[str, Any] = {}
    shared_training_report_path: Path | None = None
    shared_species_ids = [s for s in species_ids if species_included[s]]
    if shared_enabled:
        shared_output_dir = Path(str(shared_cfg.output_dir))
        shared_output_dir.mkdir(parents=True, exist_ok=True)
        shared_dataset_report = _build_shared_species_dataset(
            datasets=datasets,
            ds_species_map=ds_species_map,
            species_ids=shared_species_ids,
            output_dir=shared_output_dir,
            max_images_per_species_per_split=int(
                getattr(shared_cfg, "max_images_per_species_per_split", 0)
            ),
        )
        shared_training_report_path = reports_root / f"shared_species_training_report_{now}.json"
        shared_training_report_path.write_text(json.dumps(shared_dataset_report, indent=2), encoding="utf-8")

    # Optional training
    trained: dict[str, Any] = {}
    shared_trained: dict[str, Any] = {}
    if bool(cfg.data.species_aware.train.enabled):
        weights_root = Path(str(cfg.data.species_aware.train.weights_root))
        weights_root.mkdir(parents=True, exist_ok=True)
        base_model = str(cfg.data.species_aware.train.base_model)
        for s in species_ids:
            total = int(sum(species_split_counter[s].values()))
            if total <= 0:
                continue
            train_n = int(species_split_counter[s].get("train", 0))
            val_n = int(species_split_counter[s].get("val", 0))
            if train_n == 0 or val_n == 0:
                logger.warning("Skipping training for %s due to missing train/val samples", s)
                continue

            model = YOLO(base_model)
            species_dir = processed_root / s
            epochs = int(cfg.data.species_aware.train.epochs)
            results = model.train(
                data=str(species_dir / "data.yaml"),
                imgsz=int(cfg.data.species_aware.train.imgsz),
                epochs=epochs,
                batch=int(cfg.data.species_aware.train.batch),
                workers=int(cfg.data.species_aware.train.workers),
                project=str(weights_root),
                name=s,
                exist_ok=True,
                amp=bool(cfg.data.species_aware.train.amp),
                cache=bool(cfg.data.species_aware.train.cache),
            )
            run_dir = Path(str(getattr(results, "save_dir", ""))).resolve()
            if not run_dir.exists():
                run_dir = (weights_root / s).resolve()
            weights_dir = (weights_root / s).resolve()
            copied_weights = _copy_weights_to_canonical(run_dir, weights_dir)
            metrics = _extract_yolo_train_metrics(run_dir)
            result_summary = {}
            if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
                for k, v in results.results_dict.items():
                    try:
                        result_summary[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
            trained[s] = {
                "weights_dir": str(weights_dir),
                "run_dir": str(run_dir),
                "result_summary": result_summary,
                "epochs": epochs,
                "metrics": metrics,
                "weights_files": copied_weights,
            }
        trained_path = reports_root / f"species_aware_training_metrics_{now}.json"
        trained_path.write_text(json.dumps(trained, indent=2), encoding="utf-8")

    if shared_enabled and shared_dataset_report.get("total_images", 0) > 0:
        shared_trained = _train_shared_species_detector(
            cfg=cfg,
            dataset_dir=Path(str(shared_cfg.output_dir)),
        )
        shared_metrics_path = reports_root / f"shared_species_training_metrics_{now}.json"
        shared_metrics_path.write_text(json.dumps(shared_trained, indent=2), encoding="utf-8")

    return {
        "eligibility_csv": str(eligibility_csv.resolve()),
        "training_report_json": str(train_report_path.resolve()),
        "shared_training_report_json": str(shared_training_report_path.resolve()) if shared_training_report_path else "",
        "species_count": len(species_ids),
        "trained_species": sorted(trained),
        "shared_detector_trained": bool(shared_trained),
    }


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    out = build_species_aware_training(cfg, logger)
    logger.info("Species-aware pipeline completed: %s", out)


if __name__ == "__main__":
    main()
