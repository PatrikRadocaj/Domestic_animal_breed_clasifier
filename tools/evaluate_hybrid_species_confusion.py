"""Post-train evaluation for the hybrid species detector.

Builds an object-level confusion matrix on the hybrid validation split from
the saved source records/spec and a trained YOLO checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


BACKGROUND_LABEL = "__background__"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def xywhn_to_xyxy(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    xc, yc, wn, hn = [float(v) for v in box]
    bw = wn * width
    bh = hn * height
    xmin = max(0.0, (xc * width) - bw / 2.0)
    ymin = max(0.0, (yc * height) - bh / 2.0)
    xmax = min(float(width), xmin + bw)
    ymax = min(float(height), ymin + bh)
    return xmin, ymin, xmax, ymax


def compute_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def render_matrix(df: pd.DataFrame, out_path: Path, title: str) -> None:
    labels = list(df.index)
    size = max(10, min(40, len(labels) * 0.35))
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


def build_confusion_rows(
    *,
    model: YOLO,
    spec: dict[str, Any],
    output_dir: Path,
    conf: float,
    iou_match: float,
) -> dict[str, Any]:
    class_names = list(spec["class_names"])
    labels = class_names + [BACKGROUND_LABEL]
    name_to_index = {name: idx for idx, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    rows: list[dict[str, Any]] = []

    plans = load_jsonl(Path(spec["val_plans_jsonl"]))
    wanted_ids = {str(row["record_id"]) for row in plans if row.get("record_id")}
    records = {
        str(row["record_id"]): row
        for row in load_jsonl(Path(spec["source_records_jsonl"]))
        if str(row.get("record_id")) in wanted_ids and row.get("split") == "val"
    }

    for record in records.values():
        image_path = record["image_path"]
        width = int(record["width"])
        height = int(record["height"])
        gt_objects = record.get("objects", [])
        pred_result = model.predict(
            source=image_path,
            conf=conf,
            iou=0.5,
            imgsz=1280,
            device=0,
            verbose=False,
            max_det=50,
        )[0]
        pred_boxes = []
        if pred_result.boxes is not None:
            xyxy = pred_result.boxes.xyxy.cpu().numpy()
            cls = pred_result.boxes.cls.cpu().numpy()
            scores = pred_result.boxes.conf.cpu().numpy()
            for idx in range(len(xyxy)):
                pred_boxes.append(
                    {
                        "bbox": tuple(float(v) for v in xyxy[idx].tolist()),
                        "class_name": class_names[int(cls[idx])],
                        "confidence": float(scores[idx]),
                        "used": False,
                    }
                )

        for gt in gt_objects:
            actual_name = str(gt["species_id"])
            gt_bbox = xywhn_to_xyxy(list(gt["bbox_xywhn"]), width, height)
            best_idx = None
            best_iou = 0.0
            for idx, pred in enumerate(pred_boxes):
                if pred["used"]:
                    continue
                current_iou = compute_iou(gt_bbox, pred["bbox"])
                if current_iou >= iou_match and current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = idx
            if best_idx is None:
                predicted_name = BACKGROUND_LABEL
            else:
                pred_boxes[best_idx]["used"] = True
                predicted_name = str(pred_boxes[best_idx]["class_name"])
            matrix[name_to_index[actual_name], name_to_index[predicted_name]] += 1
            rows.append(
                {
                    "image_path": image_path,
                    "actual": actual_name,
                    "predicted": predicted_name,
                    "iou": best_iou,
                }
            )

        for pred in pred_boxes:
            if pred["used"]:
                continue
            predicted_name = str(pred["class_name"])
            matrix[name_to_index[BACKGROUND_LABEL], name_to_index[predicted_name]] += 1
            rows.append(
                {
                    "image_path": image_path,
                    "actual": BACKGROUND_LABEL,
                    "predicted": predicted_name,
                    "iou": 0.0,
                }
            )

    counts_df = pd.DataFrame(matrix, index=labels, columns=labels)
    norm_df = counts_df.div(counts_df.sum(axis=1).replace(0, 1), axis=0)
    counts_csv = output_dir / "species_confusion_matrix_counts.csv"
    norm_csv = output_dir / "species_confusion_matrix_normalized.csv"
    rows_csv = output_dir / "species_confusion_pairs.csv"
    counts_df.to_csv(counts_csv)
    norm_df.to_csv(norm_csv)
    pd.DataFrame(rows).to_csv(rows_csv, index=False)
    render_matrix(counts_df, output_dir / "species_confusion_matrix_counts.png", "Stage-1 Species Confusion Counts")
    render_matrix(norm_df, output_dir / "species_confusion_matrix_normalized.png", "Stage-1 Species Confusion Normalized")

    tp = sum(int(counts_df.loc[name, name]) for name in class_names)
    fp = int(counts_df.loc[BACKGROUND_LABEL, class_names].sum()) if BACKGROUND_LABEL in counts_df.index else 0
    fn = int(counts_df.loc[class_names, BACKGROUND_LABEL].sum()) if BACKGROUND_LABEL in counts_df.columns else 0
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)

    summary = {
        "output_dir": str(output_dir.resolve()),
        "weights": str(model.ckpt_path) if hasattr(model, "ckpt_path") else "",
        "val_records": len(records),
        "confusion_labels": labels,
        "object_precision_proxy": precision,
        "object_recall_proxy": recall,
        "counts_csv": str(counts_csv.resolve()),
        "normalized_csv": str(norm_csv.resolve()),
        "pairs_csv": str(rows_csv.resolve()),
    }
    (output_dir / "species_confusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--hybrid-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou-match", type=float, default=0.5)
    args = parser.parse_args()

    spec = json.loads(Path(args.hybrid_spec).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(Path(args.weights)))
    summary = build_confusion_rows(
        model=model,
        spec=spec,
        output_dir=output_dir,
        conf=float(args.conf),
        iou_match=float(args.iou_match),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
