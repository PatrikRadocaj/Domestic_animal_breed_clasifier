"""Build canonical dataset manifests from raw image folders."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Iterable

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from animal_breed.data.manifest_schema import validate_manifest, write_manifest
from animal_breed.data.label_mapping import CrossDatasetLabelMap
from animal_breed.data.taxonomy import Taxonomy, load_taxonomy
from animal_breed.utils.logging import setup_logging
from animal_breed.utils.reproducibility import run_metadata


def _stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    return sha1(joined.encode("utf-8")).hexdigest()[:20]


def _iter_images(root: Path, extensions: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def _image_size(path: Path) -> tuple[int, int] | tuple[None, None]:
    try:
        with Image.open(path) as image:
            return image.size
    except (UnidentifiedImageError, OSError):
        return None, None


def _labels_from_path(
    image_path: Path,
    root: Path,
    taxonomy: Taxonomy,
    source_dataset: str,
    label_map: CrossDatasetLabelMap | None,
    species_level: int,
    breed_level: int,
) -> tuple[str | None, str | None, dict[str, str] | None]:
    rel_parts = image_path.relative_to(root).parts
    if len(rel_parts) <= max(species_level, breed_level):
        return None, None, {"reason": "path_too_shallow", "label_type": "path"}

    species_raw = str(rel_parts[species_level])
    breed_raw = str(rel_parts[breed_level])

    species = None
    if label_map is not None:
        species_resolution = label_map.resolve_species(species_raw, source_dataset)
        if species_resolution.status == "manual_review":
            return None, None, {
                "reason": species_resolution.reason,
                "label_type": "species",
                "raw_label": species_raw,
                "normalized_label": species_resolution.normalized_label,
            }
        if species_resolution.status == "resolved" and species_resolution.canonical_id is not None:
            species = taxonomy.resolve_species(species_resolution.canonical_id)
        if species is None:
            species = taxonomy.resolve_species(species_raw)
        if species is None:
            return None, None, {
                "reason": "unmapped_species",
                "label_type": "species",
                "raw_label": species_raw,
                "normalized_label": species_resolution.normalized_label,
            }
    else:
        species = taxonomy.resolve_species(species_raw)
    if species is None:
        return None, None, {
            "reason": "unresolved_species",
            "label_type": "species",
            "raw_label": species_raw,
            "normalized_label": species_raw.strip().lower(),
        }

    if label_map is not None:
        breed_resolution = label_map.resolve_breed(species, breed_raw, source_dataset)
        if breed_resolution.status == "manual_review":
            return None, None, {
                "reason": breed_resolution.reason,
                "label_type": "breed",
                "raw_label": breed_raw,
                "normalized_label": breed_resolution.normalized_label,
                "species_id": species,
            }
        breed = None
        if breed_resolution.status == "resolved" and breed_resolution.canonical_id is not None:
            breed = taxonomy.resolve_breed(species, breed_resolution.canonical_id.split(":", 1)[-1])
            if breed is None and breed_resolution.canonical_id in taxonomy.breeds:
                breed = breed_resolution.canonical_id
        if breed is None:
            breed = taxonomy.resolve_breed(species, breed_raw)
        if breed is None:
            return None, None, {
                "reason": "unmapped_breed",
                "label_type": "breed",
                "raw_label": breed_raw,
                "normalized_label": breed_resolution.normalized_label,
                "species_id": species,
            }
    else:
        breed = taxonomy.resolve_breed(species, breed_raw)
    if breed is None:
        return None, None, {
            "reason": "unresolved_breed",
            "label_type": "breed",
            "raw_label": breed_raw,
            "normalized_label": breed_raw.strip().lower(),
            "species_id": species,
        }
    return species, breed, None


def _load_source_registry(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    csv_path = Path(str(path))
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if "dataset_id" not in df.columns:
        return {}
    rows: dict[str, dict[str, str]] = {}
    for row in df.to_dict(orient="records"):
        dataset_id = str(row.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        rows[dataset_id] = {k: "" if pd.isna(v) else str(v) for k, v in row.items()}
    return rows


def build_manifest_from_tree(cfg: DictConfig) -> pd.DataFrame:
    taxonomy = load_taxonomy(cfg.data.taxonomy_path)
    label_map = None
    map_path = getattr(cfg.data, "cross_dataset_label_map_path", None)
    if map_path:
        label_map = CrossDatasetLabelMap.from_file(map_path)
    extensions = {ext.lower() for ext in cfg.data.image_extensions}
    source_registry = _load_source_registry(getattr(cfg.data, "source_registry_path", None))
    rows: list[dict[str, object]] = []
    skipped: list[tuple[str, str]] = []
    unmapped_rows: list[dict[str, str]] = []
    manual_review_rows: list[dict[str, str]] = []

    source_level_cfg = getattr(cfg.data.directory_layout, "source_level", None)
    source_level = int(source_level_cfg) if source_level_cfg is not None else None
    species_level = int(cfg.data.directory_layout.species_level)
    breed_level = int(cfg.data.directory_layout.breed_level)

    for raw_root in cfg.data.raw_image_roots:
        root = Path(raw_root)
        if not root.exists():
            continue

        images = list(_iter_images(root, extensions))
        for image_path in tqdm(images, desc=f"Scanning {root}"):
            rel_parts = image_path.relative_to(root).parts
            source_dataset = str(cfg.data.source_name)
            if source_level is not None and len(rel_parts) > source_level:
                source_dataset = str(rel_parts[source_level]).strip()

            species_id, breed_id, issue = _labels_from_path(
                image_path=image_path,
                root=root,
                taxonomy=taxonomy,
                source_dataset=source_dataset,
                label_map=label_map,
                species_level=species_level,
                breed_level=breed_level,
            )
            if issue is not None:
                report_row = {
                    "path": str(image_path),
                    "source_dataset": source_dataset,
                    "label_type": str(issue.get("label_type", "")),
                    "raw_label": str(issue.get("raw_label", "")),
                    "normalized_label": str(issue.get("normalized_label", "")),
                    "species_id": str(issue.get("species_id", "")),
                    "reason": str(issue.get("reason", "")),
                }
                if "manual_review" in report_row["reason"] or "ambiguous" in report_row["reason"]:
                    manual_review_rows.append(report_row)
                    skipped.append((str(image_path), "manual_review_label"))
                else:
                    unmapped_rows.append(report_row)
                    skipped.append((str(image_path), "unresolved_taxonomy_label"))
                continue

            width, height = _image_size(image_path)
            if width is None or height is None:
                skipped.append((str(image_path), "unreadable_image"))
                continue

            source_url = ""
            source_type = ""
            source_license = str(cfg.data.default_license)
            source_meta = source_registry.get(source_dataset, {})
            if source_meta:
                source_url = str(source_meta.get("source_url", "")).strip()
                source_type = str(source_meta.get("source_type", "")).strip()
                if source_license == "unknown":
                    source_license = str(source_meta.get("license", source_license)).strip() or source_license

            path_str = str(image_path)
            image_id = _stable_id(source_dataset, path_str)
            rows.append(
                {
                    "image_id": image_id,
                    "path": path_str,
                    "source": source_dataset,
                    "source_url": source_url,
                    "source_type": source_type,
                    "license": source_license,
                    "species_id": species_id,
                    "breed_id": breed_id,
                    "width": width,
                    "height": height,
                    "bbox_xmin": pd.NA,
                    "bbox_ymin": pd.NA,
                    "bbox_xmax": pd.NA,
                    "bbox_ymax": pd.NA,
                    "is_crowd": False,
                    "split_group": image_id,
                    "duplicate_cluster": image_id,
                    "quality_tags": "",
                }
            )

    df = pd.DataFrame(rows)
    if skipped:
        skipped_path = Path("results/reports/manifest_skipped_v0.csv")
        skipped_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped, columns=["path", "reason"]).to_csv(skipped_path, index=False)
    if unmapped_rows:
        unmapped_path = Path(str(getattr(cfg.data, "unmapped_labels_report_path", "results/reports/unmapped_labels_report.csv")))
        unmapped_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(unmapped_rows).to_csv(unmapped_path, index=False)
    if manual_review_rows:
        manual_path = Path(str(getattr(cfg.data, "manual_review_report_path", "results/reports/manual_review_labels.csv")))
        manual_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manual_review_rows).to_csv(manual_path, index=False)
    return df


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    logger.info("Building manifest with config:\n%s", OmegaConf.to_yaml(cfg.data))
    logger.info("Run metadata: %s", run_metadata(cfg, repo_root="."))

    df = build_manifest_from_tree(cfg)
    if df.empty:
        logger.warning("No manifest rows were created. Check data.raw_image_roots and taxonomy aliases.")

    report = validate_manifest(
        df=df,
        taxonomy_path=cfg.data.taxonomy_path,
        require_existing_files=cfg.data.validation.require_existing_files,
        require_taxonomy_match=cfg.data.validation.require_taxonomy_match,
    )
    for warning in report.warnings:
        logger.warning(warning)
    for error in report.errors:
        logger.error(error)
    if not report.valid:
        raise ValueError("Manifest validation failed.")

    write_manifest(df, cfg.data.manifest_path, cfg.data.manifest_parquet_path)
    logger.info("Wrote %d rows to %s", len(df), cfg.data.manifest_path)


if __name__ == "__main__":
    main()
