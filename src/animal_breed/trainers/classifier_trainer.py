"""Shared classifier training core used by single-stage and hierarchical pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from animal_breed.data.datasets import BreedClassificationDataset
from animal_breed.data.transforms import classification_eval_transforms, classification_train_transforms
from animal_breed.utils.artifacts import (
    ArtifactBundle,
    build_snapshot_hashes,
    build_run_id,
    create_artifact_bundle,
    write_json,
    write_run_metadata,
)


@dataclass(frozen=True)
class ClassifierTrainResult:
    run_id: str
    best_top1: float
    best_checkpoint: str
    artifact_root: str


def _resolve_training_device(requested_device: Any) -> torch.device:
    requested = str(requested_device).strip().lower()
    if requested == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GPU device was requested ({requested_device}), but torch.cuda.is_available() is False."
        )

    if requested in {"", "auto", "cuda", "gpu"}:
        return torch.device("cuda:0")

    try:
        index = int(requested)
        return torch.device(f"cuda:{index}")
    except ValueError as exc:
        raise RuntimeError(f"Unsupported train.device value: {requested_device}") from exc


def _write_progress(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    max_k = min(max(topk), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    metrics: dict[str, float] = {}
    for k in topk:
        effective_k = min(k, logits.shape[1])
        correct_k = correct[:effective_k].reshape(-1).float().sum(0)
        metrics[f"top{effective_k}"] = float(correct_k.item() / targets.numel())
    return metrics


def _build_loaders(cfg, manifest: pd.DataFrame, label_to_index: dict[str, int], label_column: str, crop_from_bbox: bool):
    model_cfg = cfg.model.classifier
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Split manifest must contain non-empty train and val splits.")

    train_dataset = BreedClassificationDataset(
        manifest=train_df,
        label_to_index=label_to_index,
        label_column=label_column,
        transform=classification_train_transforms(int(model_cfg.img_size)),
        crop_from_bbox=crop_from_bbox,
    )
    val_dataset = BreedClassificationDataset(
        manifest=val_df,
        label_to_index=label_to_index,
        label_column=label_column,
        transform=classification_eval_transforms(int(model_cfg.img_size)),
        crop_from_bbox=crop_from_bbox,
    )

    persistent_workers = bool(getattr(cfg.train.loader, "persistent_workers", True))
    prefetch_factor = int(getattr(cfg.train.loader, "prefetch_factor", 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(model_cfg.train.batch_size),
        shuffle=True,
        num_workers=int(model_cfg.train.num_workers),
        pin_memory=bool(getattr(cfg.train.loader, "pin_memory", True)),
        persistent_workers=persistent_workers and int(model_cfg.train.num_workers) > 0,
        prefetch_factor=prefetch_factor if int(model_cfg.train.num_workers) > 0 else None,
        drop_last=bool(getattr(cfg.train.loader, "drop_last", False)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=int(cfg.eval.num_workers),
        pin_memory=bool(getattr(cfg.train.loader, "pin_memory", True)),
        persistent_workers=persistent_workers and int(cfg.eval.num_workers) > 0,
        prefetch_factor=prefetch_factor if int(cfg.eval.num_workers) > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader


def _filter_unreadable_rows(
    manifest: pd.DataFrame,
    *,
    species_id: str,
    logger,
    report_path: Path,
) -> pd.DataFrame:
    kept_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for row in manifest.to_dict(orient="records"):
        path = Path(str(row.get("path", "")))
        reason = ""
        try:
            BreedClassificationDataset.load_rgb_image(path)
        except Exception as exc:
            reason = str(exc)

        if reason:
            skipped_rows.append(
                {
                    "species_id": species_id,
                    "split": str(row.get("split", "")),
                    "path": str(path),
                    "breed_id": str(row.get("breed_id", "")),
                    "reason": reason,
                }
            )
            continue
        kept_rows.append(row)

    if skipped_rows:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped_rows).to_csv(report_path, index=False)
        logger.warning(
            "species=%s skipped_unreadable_images=%d report=%s",
            species_id,
            len(skipped_rows),
            report_path,
        )
        preview = skipped_rows[:5]
        for item in preview:
            logger.warning(
                "species=%s skipped_image split=%s path=%s reason=%s",
                species_id,
                item["split"],
                item["path"],
                item["reason"],
            )

    filtered = pd.DataFrame(kept_rows)
    if filtered.empty:
        return filtered
    return filtered.reset_index(drop=True)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger,
    species_id: str,
    epoch: int,
    total_epochs: int,
    phase: str,
    progress_path: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = True,
    log_every: int = 25,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_seen = 0
    total_top1 = 0.0
    total_top5 = 0.0
    last_loss = 0.0
    first_batch_logged = False
    loop_start = time.perf_counter()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        total_batches = len(loader)
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if not first_batch_logged:
                logger.info(
                    "species=%s epoch=%d/%d phase=%s first_batch images_device=%s targets_device=%s",
                    species_id,
                    epoch,
                    total_epochs,
                    phase,
                    images.device,
                    targets.device,
                )
                first_batch_logged = True
            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                assert optimizer is not None
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = targets.numel()
            acc = _accuracy(logits.detach(), targets, topk=(1, 5))
            batch_loss = float(loss.item())
            last_loss = batch_loss
            total_loss += batch_loss * batch_size
            total_top1 += acc["top1"] * batch_size
            total_top5 += acc[f"top{min(5, logits.shape[1])}"] * batch_size
            total_seen += batch_size

            if batch_index == 1 or batch_index % max(1, log_every) == 0 or batch_index == total_batches:
                payload = {
                    "species_id": species_id,
                    "phase": phase,
                    "epoch_current": int(epoch),
                    "epoch_total": int(total_epochs),
                    "batch_current": int(batch_index),
                    "batch_total": int(total_batches),
                    "train_loss_latest": float(last_loss) if phase == "train" else None,
                    "latest_loss": float(last_loss),
                    "device": str(device),
                    "last_update_time": datetime.now().isoformat(timespec="seconds"),
                }
                _write_progress(progress_path, payload)
                logger.info(
                    "species=%s epoch=%d/%d phase=%s batch=%d/%d loss=%.4f elapsed=%.1fs",
                    species_id,
                    epoch,
                    total_epochs,
                    phase,
                    batch_index,
                    total_batches,
                    last_loss,
                    time.perf_counter() - loop_start,
                )

    return {
        "loss": total_loss / max(1, total_seen),
        "top1": total_top1 / max(1, total_seen),
        "top5": total_top5 / max(1, total_seen),
    }


def _save_checkpoint(bundle: ArtifactBundle, filename: str, payload: dict[str, Any]) -> str:
    out = bundle.checkpoints_dir / filename
    torch.save(payload, out)
    return str(out.resolve())


def _load_compatible_checkpoint(model: nn.Module, checkpoint_path: Path, logger, species_id: str) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model", payload)
    current_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in state_dict.items():
        if key not in current_state:
            skipped.append(key)
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible[key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    logger.info(
        "species=%s warm_start checkpoint=%s loaded_tensors=%d skipped_tensors=%d missing_after_load=%d unexpected_after_load=%d",
        species_id,
        checkpoint_path,
        len(compatible),
        len(skipped),
        len(missing),
        len(unexpected),
    )


def train_classifier(
    cfg,
    manifest: pd.DataFrame,
    label_to_index: dict[str, int],
    label_column: str,
    run_prefix: str,
    logger,
    initial_checkpoint: str | None = None,
) -> ClassifierTrainResult:
    run_id = build_run_id(run_prefix, cfg)
    bundle = create_artifact_bundle(run_id, artifacts_root=str(cfg.paths.artifacts_dir))
    snapshot_hashes = build_snapshot_hashes(
        cfg,
        manifest_path=cfg.data.split_manifest_path if "split" in manifest.columns else cfg.data.manifest_path,
        taxonomy_path=cfg.data.taxonomy_path,
        source_registry_path=getattr(cfg.data, "source_registry_path", None),
    )
    write_run_metadata(bundle, cfg, snapshot_hashes)

    crop_from_bbox = str(cfg.model.classifier.crop_mode) in {"bbox", "detector_crop"}
    species_id = run_prefix.removeprefix("two-stage-breed-")
    manifest = _filter_unreadable_rows(
        manifest,
        species_id=species_id,
        logger=logger,
        report_path=bundle.qa_dir / "skipped_unreadable_images.csv",
    )
    if manifest.empty:
        raise ValueError(f"No readable rows remain for species={species_id} after filtering unreadable images.")
    train_loader, val_loader = _build_loaders(cfg, manifest, label_to_index, label_column, crop_from_bbox)

    device = _resolve_training_device(cfg.train.device)
    logger.info(
        "classifier device request=%s resolved=%s torch.cuda.is_available=%s torch.version.hip=%s",
        cfg.train.device,
        device,
        torch.cuda.is_available(),
        getattr(torch.version, "hip", None),
    )
    if device.type == "cuda":
        logger.info(
            "classifier device index=%s name=%s",
            device.index if device.index is not None else 0,
            torch.cuda.get_device_name(device.index if device.index is not None else 0),
        )
    model = timm.create_model(
        str(cfg.model.classifier.architecture),
        pretrained=bool(cfg.model.classifier.pretrained),
        num_classes=len(label_to_index),
    ).to(device)
    first_param = next(model.parameters(), None)
    logger.info("classifier model first_param_device=%s", first_param.device if first_param is not None else "missing")
    if str(first_param.device).split(":")[0] != device.type:
        raise RuntimeError(f"Model parameter device mismatch: expected {device}, got {first_param.device}")

    if initial_checkpoint:
        init_path = Path(initial_checkpoint)
        if init_path.exists():
            _load_compatible_checkpoint(model, init_path, logger, species_id)
        else:
            logger.warning("species=%s warm_start checkpoint missing: %s", species_id, init_path)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.model.classifier.train.label_smoothing))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.model.classifier.train.lr),
        weight_decay=float(cfg.model.classifier.train.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.model.classifier.train.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.model.classifier.train.amp))

    last_checkpoint = bundle.checkpoints_dir / "last.pt"
    if bool(cfg.train.resume) and last_checkpoint.exists():
        payload = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        logger.info("Resumed classifier from %s", last_checkpoint)

    best_top1 = -1.0
    best_checkpoint_path = ""
    epochs_without_improvement = 0
    early_stopping_patience = int(getattr(cfg.model.classifier.train, "early_stopping_patience", 0) or 0)
    early_stopping_min_delta = float(getattr(cfg.model.classifier.train, "early_stopping_min_delta", 0.0) or 0.0)
    history: list[dict[str, Any]] = []
    progress_path = str((bundle.root / "progress" / "status.json").resolve())
    logger.info(
        "species training starting species=%s epochs=%d train_batches=%d val_batches=%d progress_path=%s",
        species_id,
        int(cfg.model.classifier.train.epochs),
        len(train_loader),
        len(val_loader),
        progress_path,
    )

    for epoch in range(1, int(cfg.model.classifier.train.epochs) + 1):
        start = time.perf_counter()
        logger.info("species=%s epoch=%d/%d phase=train start", species_id, epoch, int(cfg.model.classifier.train.epochs))
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            logger,
            species_id,
            epoch,
            int(cfg.model.classifier.train.epochs),
            "train",
            progress_path=progress_path,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(cfg.model.classifier.train.amp),
            log_every=25,
        )
        logger.info("species=%s epoch=%d/%d phase=val start", species_id, epoch, int(cfg.model.classifier.train.epochs))
        val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            logger,
            species_id,
            epoch,
            int(cfg.model.classifier.train.epochs),
            "val",
            progress_path=progress_path,
            amp=bool(cfg.model.classifier.train.amp),
            log_every=25,
        )
        scheduler.step()
        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "seconds": elapsed,
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.4f train_top1=%.4f val_loss=%.4f val_top1=%.4f val_top5=%.4f time=%.1fs",
            epoch,
            row["train_loss"],
            row["train_top1"],
            row["val_loss"],
            row["val_top1"],
            row["val_top5"],
            row["seconds"],
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "label_to_index": label_to_index,
            "val_metrics": val_metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        _save_checkpoint(bundle, "last.pt", checkpoint)
        if val_metrics["top1"] > best_top1 + early_stopping_min_delta:
            best_top1 = val_metrics["top1"]
            best_checkpoint_path = _save_checkpoint(bundle, "best.pt", checkpoint)
            epochs_without_improvement = 0
            logger.info("species=%s epoch=%d new_best_val_top1=%.4f", species_id, epoch, best_top1)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "early stopping triggered at epoch=%d best_val_top1=%.4f patience=%d min_delta=%.4f",
                    epoch,
                    best_top1,
                    early_stopping_patience,
                    early_stopping_min_delta,
                )
                break

    _write_progress(
        progress_path,
        {
            "species_id": run_prefix.removeprefix("two-stage-breed-"),
            "phase": "complete",
            "epoch_current": int(history[-1]["epoch"]) if history else 0,
            "epoch_total": int(cfg.model.classifier.train.epochs),
            "batch_current": None,
            "batch_total": None,
            "train_loss_latest": float(history[-1]["train_loss"]) if history else None,
            "device": str(device),
            "last_update_time": datetime.now().isoformat(timespec="seconds"),
            "best_top1": float(best_top1),
        },
    )

    write_json(bundle.metrics_dir / "classifier_epoch_history.json", {"history": history})
    pd.DataFrame(history).to_csv(bundle.metrics_dir / "classifier_epoch_history.csv", index=False)

    return ClassifierTrainResult(
        run_id=run_id,
        best_top1=float(best_top1),
        best_checkpoint=best_checkpoint_path,
        artifact_root=str(bundle.root.resolve()),
    )
