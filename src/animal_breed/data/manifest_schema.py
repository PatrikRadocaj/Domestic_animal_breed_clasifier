"""Canonical manifest schema and validation CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hydra
import pandas as pd
from omegaconf import DictConfig

from animal_breed.data.taxonomy import load_taxonomy
from animal_breed.utils.logging import setup_logging


REQUIRED_COLUMNS: tuple[str, ...] = (
    "image_id",
    "path",
    "source",
    "license",
    "species_id",
    "breed_id",
    "width",
    "height",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "is_crowd",
    "split_group",
    "duplicate_cluster",
    "quality_tags",
)

OPTIONAL_SOURCE_COLUMNS: tuple[str, ...] = (
    "source_url",
    "source_type",
)


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_manifest(df: pd.DataFrame, csv_path: str | Path, parquet_path: str | Path | None = None) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)


def validate_manifest(
    df: pd.DataFrame,
    taxonomy_path: str | Path,
    require_existing_files: bool = True,
    require_taxonomy_match: bool = True,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return ValidationReport(False, tuple(errors), tuple(warnings))

    if df["image_id"].duplicated().any():
        errors.append("image_id must be unique.")

    if df["path"].isna().any():
        errors.append("path contains null values.")

    if require_existing_files:
        missing_paths = [p for p in df["path"].head(10000) if not Path(str(p)).exists()]
        if missing_paths:
            errors.append(f"Found missing files, first examples: {missing_paths[:5]}")

    invalid_dims = df[(df["width"].fillna(0) <= 0) | (df["height"].fillna(0) <= 0)]
    if not invalid_dims.empty:
        errors.append(f"{len(invalid_dims)} rows have invalid image dimensions.")

    bbox_cols = ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
    has_partial_bbox = df[bbox_cols].isna().any(axis=1) & ~df[bbox_cols].isna().all(axis=1)
    if has_partial_bbox.any():
        errors.append(f"{int(has_partial_bbox.sum())} rows have partial bounding boxes.")

    if require_taxonomy_match:
        taxonomy = load_taxonomy(taxonomy_path)
        unknown_species = sorted(set(df["species_id"].dropna()) - set(taxonomy.species))
        unknown_breeds = sorted(set(df["breed_id"].dropna()) - set(taxonomy.breeds))
        if unknown_species:
            errors.append(f"Unknown species_id values: {unknown_species[:20]}")
        if unknown_breeds:
            errors.append(f"Unknown breed_id values: {unknown_breeds[:20]}")

        bad_hierarchy = []
        for row in df[["species_id", "breed_id"]].dropna().itertuples(index=False):
            expected_species = taxonomy.breed_species(row.breed_id)
            if expected_species is not None and expected_species != row.species_id:
                bad_hierarchy.append((row.species_id, row.breed_id, expected_species))
        if bad_hierarchy:
            errors.append(f"Species/breed hierarchy mismatch examples: {bad_hierarchy[:20]}")

    duplicate_paths = int(df["path"].duplicated().sum())
    if duplicate_paths:
        warnings.append(f"{duplicate_paths} duplicate file paths found.")

    if "source_url" in df.columns:
        missing_source_urls = int(df["source_url"].fillna("").astype(str).str.strip().eq("").sum())
        if missing_source_urls:
            warnings.append(
                f"{missing_source_urls} rows have empty source_url. "
                "Set data.source_registry_path to preserve provenance."
            )

    return ValidationReport(not errors, tuple(errors), tuple(warnings))


def assert_valid(report: ValidationReport) -> None:
    if not report.valid:
        raise ValueError("\n".join(report.errors))


def _print_messages(messages: Iterable[str], prefix: str) -> None:
    for message in messages:
        print(f"{prefix}: {message}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    df = read_manifest(cfg.data.manifest_path)
    report = validate_manifest(
        df=df,
        taxonomy_path=cfg.data.taxonomy_path,
        require_existing_files=cfg.data.validation.require_existing_files,
        require_taxonomy_match=cfg.data.validation.require_taxonomy_match,
    )
    _print_messages(report.warnings, "WARNING")
    _print_messages(report.errors, "ERROR")
    logger.info("Manifest validation valid=%s rows=%d", report.valid, len(df))
    assert_valid(report)


if __name__ == "__main__":
    main()
