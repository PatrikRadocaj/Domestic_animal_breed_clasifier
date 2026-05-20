"""Train a timm breed classifier from the split manifest."""

from __future__ import annotations

from pathlib import Path
import time

import hydra
import pandas as pd
import timm
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from animal_breed.data.datasets import BreedClassificationDataset
from animal_breed.data.manifest_schema import read_manifest
from animal_breed.data.taxonomy import load_taxonomy
from animal_breed.data.transforms import classification_eval_transforms, classification_train_transforms
from animal_breed.utils.logging import setup_logging
from animal_breed.utils.reproducibility import run_metadata, seed_everything


def _accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    max_k = min(max(topk), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    metrics = {}
    for k in topk:
        effective_k = min(k, logits.shape[1])
        correct_k = correct[:effective_k].reshape(-1).float().sum(0)
        metrics[f"top{effective_k}"] = float(correct_k.item() / targets.numel())
    return metrics


def _build_loaders(cfg: DictConfig, manifest: pd.DataFrame, label_to_index: dict[str, int]):
    model_cfg = cfg.model.classifier
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Split manifest must contain non-empty train and val splits.")

    crop_from_bbox = str(model_cfg.crop_mode) in {"bbox", "detector_crop"}
    train_dataset = BreedClassificationDataset(
        manifest=train_df,
        label_to_index=label_to_index,
        transform=classification_train_transforms(int(model_cfg.img_size)),
        crop_from_bbox=crop_from_bbox,
    )
    val_dataset = BreedClassificationDataset(
        manifest=val_df,
        label_to_index=label_to_index,
        transform=classification_eval_transforms(int(model_cfg.img_size)),
        crop_from_bbox=crop_from_bbox,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(model_cfg.train.batch_size),
        shuffle=True,
        num_workers=int(model_cfg.train.num_workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=int(cfg.eval.num_workers),
        pin_memory=True,
    )
    return train_loader, val_loader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = True,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_seen = 0
    total_top1 = 0.0
    total_top5 = 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = targets.numel()
            acc = _accuracy(logits.detach(), targets, topk=(1, 5))
            total_loss += float(loss.item()) * batch_size
            total_top1 += acc["top1"] * batch_size
            total_top5 += acc[f"top{min(5, logits.shape[1])}"] * batch_size
            total_seen += batch_size

    return {
        "loss": total_loss / max(1, total_seen),
        "top1": total_top1 / max(1, total_seen),
        "top5": total_top5 / max(1, total_seen),
    }


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
    logger.info("Training classifier with config:\n%s", OmegaConf.to_yaml(cfg.model.classifier))
    logger.info("Run metadata: %s", run_metadata(cfg, repo_root="."))

    manifest = read_manifest(cfg.data.split_manifest_path)
    taxonomy = load_taxonomy(cfg.data.taxonomy_path)
    breed_ids = [record["breed_id"] for record in taxonomy.breed_records()]
    active_breeds = sorted(set(manifest["breed_id"]) & set(breed_ids))
    label_to_index = {breed_id: idx for idx, breed_id in enumerate(active_breeds)}
    if not label_to_index:
        raise ValueError("No active breeds from manifest match taxonomy.")

    train_loader, val_loader = _build_loaders(cfg, manifest, label_to_index)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device != "cpu" else "cpu")

    model = timm.create_model(
        str(cfg.model.classifier.architecture),
        pretrained=bool(cfg.model.classifier.pretrained),
        num_classes=len(label_to_index),
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.model.classifier.train.label_smoothing))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.model.classifier.train.lr),
        weight_decay=float(cfg.model.classifier.train.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.model.classifier.train.epochs),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.model.classifier.train.amp))

    output_dir = Path(cfg.train.checkpoint_dir) / "classifier" / str(cfg.model.classifier.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_top1 = -1.0

    for epoch in range(1, int(cfg.model.classifier.train.epochs) + 1):
        start = time.perf_counter()
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, bool(cfg.model.classifier.train.amp)
        )
        val_metrics = _run_epoch(model, val_loader, criterion, device, amp=bool(cfg.model.classifier.train.amp))
        scheduler.step()
        elapsed = time.perf_counter() - start

        logger.info(
            "epoch=%d train_loss=%.4f train_top1=%.4f val_loss=%.4f val_top1=%.4f val_top5=%.4f time=%.1fs",
            epoch,
            train_metrics["loss"],
            train_metrics["top1"],
            val_metrics["loss"],
            val_metrics["top1"],
            val_metrics["top5"],
            elapsed,
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "label_to_index": label_to_index,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
