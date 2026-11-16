"""Repair finalized temporary breed manifests before training.

This fixes path/image-dimension mismatches caused by EXIF orientation and
clips detector-generated boxes back into valid image bounds. The tool is
run-scoped and rewrites only the selected manifest outputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import DecompressionBombError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from animal_breed.data.quality.qa import run_quality_checks


def _safe_exif_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return image.size
    except (UnidentifiedImageError, OSError, DecompressionBombError):
        return None


def _repair_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    repaired = df.copy()
    bbox_cols = ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
    complete_mask = ~repaired[bbox_cols].isna().any(axis=1)

    path_to_size: dict[str, tuple[int, int]] = {}
    unreadable_paths: set[str] = set()
    width_updates = 0
    invalid_after_clip = 0
    clipped_boxes = 0

    affected_paths = sorted(set(repaired.loc[complete_mask, "path"].astype(str)))
    for path_str in affected_paths:
        size = _safe_exif_size(Path(path_str))
        if size is None:
            unreadable_paths.add(path_str)
            continue
        path_to_size[path_str] = size

    for path_str, (width, height) in path_to_size.items():
        path_mask = repaired["path"].astype(str) == path_str
        if not path_mask.any():
            continue
        if ((repaired.loc[path_mask, "width"] != width) | (repaired.loc[path_mask, "height"] != height)).any():
            width_updates += int(path_mask.sum())
        repaired.loc[path_mask, "width"] = int(width)
        repaired.loc[path_mask, "height"] = int(height)

    drop_indices: list[int] = []
    for idx, row in repaired.loc[complete_mask].iterrows():
        width = float(row["width"])
        height = float(row["height"])
        xmin = max(0.0, min(float(row["bbox_xmin"]), width))
        ymin = max(0.0, min(float(row["bbox_ymin"]), height))
        xmax = max(0.0, min(float(row["bbox_xmax"]), width))
        ymax = max(0.0, min(float(row["bbox_ymax"]), height))

        if (
            xmin != float(row["bbox_xmin"])
            or ymin != float(row["bbox_ymin"])
            or xmax != float(row["bbox_xmax"])
            or ymax != float(row["bbox_ymax"])
        ):
            clipped_boxes += 1

        if xmax <= xmin or ymax <= ymin:
            drop_indices.append(idx)
            invalid_after_clip += 1
            continue

        repaired.at[idx, "bbox_xmin"] = xmin
        repaired.at[idx, "bbox_ymin"] = ymin
        repaired.at[idx, "bbox_xmax"] = xmax
        repaired.at[idx, "bbox_ymax"] = ymax

    if drop_indices:
        repaired = repaired.drop(index=drop_indices).reset_index(drop=True)

    report = {
        "rows_in": int(len(df)),
        "rows_out": int(len(repaired)),
        "rows_dropped": int(len(df) - len(repaired)),
        "paths_checked": int(len(affected_paths)),
        "paths_unreadable": int(len(unreadable_paths)),
        "rows_with_dimension_updates": int(width_updates),
        "boxes_clipped": int(clipped_boxes),
        "boxes_dropped_after_clip": int(invalid_after_clip),
        "unreadable_path_examples": sorted(unreadable_paths)[:20],
    }
    return repaired, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--manifest-parquet", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--backup-suffix", default=".pre_repair.bak")
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv)
    manifest_parquet = Path(args.manifest_parquet)
    taxonomy_path = Path(args.taxonomy)
    report_json = Path(args.report_json)

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest csv: {manifest_csv}")
    if not manifest_parquet.exists():
        raise FileNotFoundError(f"Missing manifest parquet: {manifest_parquet}")

    csv_backup = manifest_csv.with_name(manifest_csv.name + args.backup_suffix)
    parquet_backup = manifest_parquet.with_name(manifest_parquet.name + args.backup_suffix)
    if not csv_backup.exists():
        shutil.copy2(manifest_csv, csv_backup)
    if not parquet_backup.exists():
        shutil.copy2(manifest_parquet, parquet_backup)

    df = pd.read_csv(manifest_csv)
    repaired, repair_report = _repair_dataframe(df)
    _, qa_report = run_quality_checks(repaired, taxonomy_path, {"strictness": "benchmark"})

    repaired.to_csv(manifest_csv, index=False)
    repaired.to_parquet(manifest_parquet, index=False)

    payload = {
        "repair": repair_report,
        "qa": qa_report,
        "manifest_csv": str(manifest_csv.resolve()),
        "manifest_parquet": str(manifest_parquet.resolve()),
        "taxonomy": str(taxonomy_path.resolve()),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not qa_report["valid"]:
        raise SystemExit("Repair completed, but quality gate still fails. See report_json for details.")


if __name__ == "__main__":
    main()
