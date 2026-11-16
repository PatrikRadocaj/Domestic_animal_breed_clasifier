"""Dataset audit report generation."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from animal_breed.data.manifest_schema import read_manifest, validate_manifest
from animal_breed.utils.logging import setup_logging


def _counts_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        df.groupby(columns)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )


def audit_manifest(df: pd.DataFrame) -> dict[str, object]:
    breed_counts = df["breed_id"].value_counts()
    species_counts = df["species_id"].value_counts()
    has_boxes = df[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].notna().all(axis=1)

    source_urls = df["source_url"] if "source_url" in df.columns else pd.Series(dtype=str)
    return {
        "num_images": int(len(df)),
        "num_species": int(df["species_id"].nunique()),
        "num_breeds": int(df["breed_id"].nunique()),
        "num_sources": int(df["source"].nunique()),
        "num_source_urls": int(source_urls.replace("", pd.NA).dropna().nunique()) if not source_urls.empty else 0,
        "num_images_with_boxes": int(has_boxes.sum()),
        "num_images_without_boxes": int((~has_boxes).sum()),
        "duplicate_path_count": int(df["path"].duplicated().sum()),
        "species_counts": species_counts.to_dict(),
        "breed_count_summary": {
            "min": int(breed_counts.min()) if not breed_counts.empty else 0,
            "median": float(breed_counts.median()) if not breed_counts.empty else 0.0,
            "mean": float(breed_counts.mean()) if not breed_counts.empty else 0.0,
            "max": int(breed_counts.max()) if not breed_counts.empty else 0,
            "tail_lt_10": int((breed_counts < 10).sum()),
            "tail_lt_50": int((breed_counts < 50).sum()),
            "head_gte_500": int((breed_counts >= 500).sum()),
        },
    }


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    df = read_manifest(cfg.data.manifest_path)

    validation = validate_manifest(
        df=df,
        taxonomy_path=cfg.data.taxonomy_path,
        require_existing_files=cfg.data.validation.require_existing_files,
        require_taxonomy_match=cfg.data.validation.require_taxonomy_match,
    )
    for warning in validation.warnings:
        logger.warning(warning)
    for error in validation.errors:
        logger.error(error)

    report = audit_manifest(df)
    output_dir = Path("results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "dataset_audit_v0.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _counts_table(df, ["species_id"]).to_csv(output_dir / "species_counts_v0.csv", index=False)
    _counts_table(df, ["species_id", "breed_id"]).to_csv(
        output_dir / "breed_counts_v0.csv",
        index=False,
    )
    logger.info("Wrote audit reports to %s", output_dir)


if __name__ == "__main__":
    main()
