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
    species_level: int,
    breed_level: int,
) -> tuple[str | None, str | None]:
    rel_parts = image_path.relative_to(root).parts
    if len(rel_parts) <= max(species_level, breed_level):
        return None, None

    species = taxonomy.resolve_species(rel_parts[species_level])
    if species is None:
        return None, None

    breed = taxonomy.resolve_breed(species, rel_parts[breed_level])
    return species, breed


def build_manifest_from_tree(cfg: DictConfig) -> pd.DataFrame:
    taxonomy = load_taxonomy(cfg.data.taxonomy_path)
    extensions = {ext.lower() for ext in cfg.data.image_extensions}
    rows: list[dict[str, object]] = []
    skipped: list[tuple[str, str]] = []

    species_level = int(cfg.data.directory_layout.species_level)
    breed_level = int(cfg.data.directory_layout.breed_level)

    for raw_root in cfg.data.raw_image_roots:
        root = Path(raw_root)
        if not root.exists():
            continue

        images = list(_iter_images(root, extensions))
        for image_path in tqdm(images, desc=f"Scanning {root}"):
            species_id, breed_id = _labels_from_path(
                image_path=image_path,
                root=root,
                taxonomy=taxonomy,
                species_level=species_level,
                breed_level=breed_level,
            )
            if species_id is None or breed_id is None:
                skipped.append((str(image_path), "unresolved_taxonomy_label"))
                continue

            width, height = _image_size(image_path)
            if width is None or height is None:
                skipped.append((str(image_path), "unreadable_image"))
                continue

            path_str = str(image_path)
            image_id = _stable_id(cfg.data.source_name, path_str)
            rows.append(
                {
                    "image_id": image_id,
                    "path": path_str,
                    "source": str(cfg.data.source_name),
                    "license": str(cfg.data.default_license),
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
