"""Export per-species and aggregated Stage-2 breed confusion matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje")
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from animal_breed.data.datasets import BreedClassificationDataset
from animal_breed.data.manifest_schema import read_manifest
from animal_breed.data.transforms import classification_eval_transforms
from animal_breed.data.taxonomy import load_taxonomy


def render_matrix(df: pd.DataFrame, out_path: Path, title: str) -> None:
    labels = list(df.index)
    size = max(10, min(48, len(labels) * 0.35))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(df.to_numpy(), interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if len(labels) <= 80:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def resolve_device(device_value: str) -> torch.device:
    requested = str(device_value).strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU requested for stage2 confusion export, but torch.cuda.is_available() is False.")
    if requested in {"", "auto", "cuda", "gpu"}:
        return torch.device("cuda:0")
    return torch.device(f"cuda:{int(requested)}")


def evaluate_species_checkpoint(
    checkpoint_path: Path,
    species_manifest: pd.DataFrame,
    label_to_index: dict[str, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = payload.get("config", {})
    model_cfg = ckpt_cfg.get("model", {}).get("classifier", {})
    train_cfg = ckpt_cfg.get("eval", {})
    img_size = int(model_cfg.get("img_size", 320))
    crop_mode = str(model_cfg.get("crop_mode", "detector_crop"))
    architecture = str(model_cfg.get("architecture"))

    val_df = species_manifest[species_manifest["split"] == "val"].reset_index(drop=True)
    dataset = BreedClassificationDataset(
        manifest=val_df,
        label_to_index=label_to_index,
        label_column="breed_id",
        transform=classification_eval_transforms(img_size),
        crop_from_bbox=crop_mode in {"bbox", "detector_crop"},
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size or int(train_cfg.get("batch_size", 64) or 64),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = timm.create_model(architecture, pretrained=False, num_classes=len(label_to_index))
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device)
    model.eval()

    index_to_label = {idx: label for label, idx in label_to_index.items()}
    pairs: list[dict[str, Any]] = []
    matrix = np.zeros((len(label_to_index), len(label_to_index)), dtype=np.int64)

    row_offset = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for idx in range(targets.shape[0]):
                actual_idx = int(targets[idx].item())
                pred_idx = int(preds[idx].item())
                matrix[actual_idx, pred_idx] += 1
                source_row = val_df.iloc[row_offset + idx]
                pairs.append(
                    {
                        "path": str(source_row["path"]),
                        "species_id": str(source_row["species_id"]),
                        "actual_breed_id": index_to_label[actual_idx],
                        "predicted_breed_id": index_to_label[pred_idx],
                    }
                )
            row_offset += targets.shape[0]

    labels = [index_to_label[idx] for idx in range(len(index_to_label))]
    counts_df = pd.DataFrame(matrix, index=labels, columns=labels)
    return counts_df, pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--taxonomy-yaml", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest_csv)
    taxonomy = load_taxonomy(args.taxonomy_yaml)
    summary_df = pd.read_csv(args.summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    global_pairs: list[dict[str, Any]] = []
    per_species_summary: list[dict[str, Any]] = []

    for row in summary_df.to_dict(orient="records"):
        species_id = str(row["species_id"])
        checkpoint_path = Path(str(row["best_checkpoint"]))
        species_manifest = manifest[manifest["species_id"] == species_id].reset_index(drop=True)
        breed_ids = sorted(
            breed_id for breed_id in set(species_manifest["breed_id"].dropna().astype(str))
            if breed_id.startswith(f"{species_id}:")
        )
        label_to_index = {breed_id: idx for idx, breed_id in enumerate(breed_ids)}
        counts_df, pairs = evaluate_species_checkpoint(
            checkpoint_path=checkpoint_path,
            species_manifest=species_manifest,
            label_to_index=label_to_index,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )
        species_out = output_dir / "per_species" / species_id
        species_out.mkdir(parents=True, exist_ok=True)
        norm_df = counts_df.div(counts_df.sum(axis=1).replace(0, 1), axis=0)
        counts_df.to_csv(species_out / "confusion_counts.csv")
        norm_df.to_csv(species_out / "confusion_normalized.csv")
        pd.DataFrame(pairs).to_csv(species_out / "predictions.csv", index=False)
        render_matrix(counts_df, species_out / "confusion_counts.png", f"{species_id} Stage-2 Confusion Counts")
        render_matrix(norm_df, species_out / "confusion_normalized.png", f"{species_id} Stage-2 Confusion Normalized")
        per_species_summary.append(
            {
                "species_id": species_id,
                "val_rows": len(pairs),
                "counts_csv": str((species_out / "confusion_counts.csv").resolve()),
                "normalized_csv": str((species_out / "confusion_normalized.csv").resolve()),
            }
        )
        global_pairs.extend(pairs)

    all_breeds = [
        row["breed_id"]
        for row in taxonomy.breed_records()
        if any(str(row["breed_id"]).startswith(f"{species_id}:") for species_id in set(summary_df["species_id"].astype(str)))
    ]
    if not all_breeds:
        all_breeds = sorted(set(item["actual_breed_id"] for item in global_pairs) | set(item["predicted_breed_id"] for item in global_pairs))
    global_matrix = pd.DataFrame(0, index=all_breeds, columns=all_breeds, dtype=np.int64)
    for pair in global_pairs:
        actual = str(pair["actual_breed_id"])
        predicted = str(pair["predicted_breed_id"])
        if actual not in global_matrix.index:
            global_matrix.loc[actual] = 0
            global_matrix[actual] = 0
        if predicted not in global_matrix.columns:
            global_matrix[predicted] = 0
        if predicted not in global_matrix.index:
            global_matrix.loc[predicted] = 0
        global_matrix.loc[actual, predicted] += 1
    global_matrix = global_matrix.sort_index().reindex(sorted(global_matrix.columns), axis=1, fill_value=0)
    global_norm = global_matrix.div(global_matrix.sum(axis=1).replace(0, 1), axis=0)
    global_matrix.to_csv(output_dir / "global_confusion_counts.csv")
    global_norm.to_csv(output_dir / "global_confusion_normalized.csv")
    pd.DataFrame(global_pairs).to_csv(output_dir / "global_predictions.csv", index=False)
    render_matrix(global_matrix, output_dir / "global_confusion_counts.png", "Stage-2 Global Breed Confusion Counts")
    render_matrix(global_norm, output_dir / "global_confusion_normalized.png", "Stage-2 Global Breed Confusion Normalized")

    summary = {
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "species_models": per_species_summary,
        "global_pairs": len(global_pairs),
        "global_counts_csv": str((output_dir / "global_confusion_counts.csv").resolve()),
        "global_normalized_csv": str((output_dir / "global_confusion_normalized.csv").resolve()),
    }
    (output_dir / "stage2_confusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
