"""Train species-conditioned breed classifiers for the YOLO -> breed two-stage pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from animal_breed.data.manifest_schema import read_manifest
from animal_breed.data.quality.qa import run_quality_checks
from animal_breed.data.taxonomy import load_taxonomy
from animal_breed.trainers.classifier_trainer import train_classifier
from animal_breed.utils.artifacts import (
    build_run_id,
    build_snapshot_hashes,
    create_artifact_bundle,
    write_json,
    write_run_metadata,
)
from animal_breed.utils.logging import setup_logging
from animal_breed.utils.reproducibility import run_metadata, seed_everything


def _copy_checkpoint(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.resolve())


def _find_prior_species_checkpoint(artifacts_root: Path, species_id: str, current_run_id: str) -> str | None:
    candidates: list[Path] = []
    candidates.extend(artifacts_root.glob(f"two-stage-breed-models-*/checkpoints/breed_models/{species_id}/best.pt"))
    candidates.extend(artifacts_root.glob(f"two-stage-breed-{species_id}-*/checkpoints/best.pt"))
    filtered = [
        path for path in candidates
        if current_run_id not in str(path) and path.exists()
    ]
    if not filtered:
        return None
    best = sorted(filtered, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return str(best.resolve())


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
    logger.info("Training two-stage breed models with config:\n%s", OmegaConf.to_yaml(cfg.model.classifier))
    logger.info("Run metadata: %s", run_metadata(cfg, repo_root="."))

    manifest = read_manifest(cfg.data.split_manifest_path)
    _, qa_report = run_quality_checks(manifest, cfg.data.taxonomy_path, dict(cfg.data.quality))
    if not qa_report["valid"]:
        run_id = build_run_id("two-stage-breed-models", cfg)
        bundle = create_artifact_bundle(run_id, artifacts_root=str(cfg.paths.artifacts_dir))
        snapshot_hashes = build_snapshot_hashes(
            cfg,
            manifest_path=cfg.data.split_manifest_path,
            taxonomy_path=cfg.data.taxonomy_path,
            source_registry_path=getattr(cfg.data, "source_registry_path", None),
        )
        write_run_metadata(bundle, cfg, snapshot_hashes)
        write_json(bundle.qa_dir / "data_quality_report.json", qa_report)
        raise ValueError("Data quality gate failed before two-stage breed training")

    taxonomy = load_taxonomy(cfg.data.taxonomy_path)
    run_id = build_run_id("two-stage-breed-models", cfg)
    bundle = create_artifact_bundle(run_id, artifacts_root=str(cfg.paths.artifacts_dir))
    snapshot_hashes = build_snapshot_hashes(
        cfg,
        manifest_path=cfg.data.split_manifest_path,
        taxonomy_path=cfg.data.taxonomy_path,
        source_registry_path=getattr(cfg.data, "source_registry_path", None),
    )
    write_run_metadata(bundle, cfg, snapshot_hashes)

    models_root = bundle.checkpoints_dir / "breed_models"
    summaries: list[dict[str, object]] = []
    start_species = getattr(cfg, "start_species", None)
    started = start_species is None
    if start_species:
        logger.info("Two-stage breed restart requested from species=%s", start_species)
    warm_start_enabled = bool(getattr(cfg, "stage2_warm_start_from_old", True))

    for species in taxonomy.species_records():
        species_id = species["species_id"]
        if not started:
            if species_id == str(start_species):
                started = True
            else:
                logger.info("Skipping already-completed species before restart point: %s", species_id)
                continue
        species_manifest = manifest[manifest["species_id"] == species_id].reset_index(drop=True)
        if species_manifest.empty:
            continue

        train_df = species_manifest[species_manifest["split"] == "train"]
        val_df = species_manifest[species_manifest["split"] == "val"]
        if train_df.empty or val_df.empty:
            continue

        active_breeds = sorted(set(species_manifest["breed_id"].dropna().astype(str)))
        active_breeds = [breed_id for breed_id in active_breeds if breed_id.startswith(f"{species_id}:")]
        if len(active_breeds) < 2:
            continue

        label_to_index_breed = {breed_id: i for i, breed_id in enumerate(active_breeds)}
        logger.info(
            "Training species-conditioned breed model for %s with %d train / %d val samples across %d breeds",
            species_id,
            len(train_df),
            len(val_df),
            len(active_breeds),
        )
        initial_checkpoint = None
        if warm_start_enabled:
            initial_checkpoint = _find_prior_species_checkpoint(Path(str(cfg.paths.artifacts_dir)), species_id, run_id)
            if initial_checkpoint:
                logger.info("species=%s warm-starting from old checkpoint=%s", species_id, initial_checkpoint)
        result = train_classifier(
            cfg=cfg,
            manifest=species_manifest,
            label_to_index=label_to_index_breed,
            label_column="breed_id",
            run_prefix=f"two-stage-breed-{species_id}",
            logger=logger,
            initial_checkpoint=initial_checkpoint,
        )
        artifact_root = Path(result.artifact_root)
        best_src = Path(result.best_checkpoint)
        last_src = artifact_root / "checkpoints" / "last.pt"
        metrics_csv = artifact_root / "metrics" / "classifier_epoch_history.csv"

        best_dst = models_root / species_id / "best.pt"
        last_dst = models_root / species_id / "last.pt"
        _copy_checkpoint(best_src, best_dst)
        if last_src.exists():
            _copy_checkpoint(last_src, last_dst)

        summaries.append(
            {
                "species_id": species_id,
                "train_samples": int(len(train_df)),
                "val_samples": int(len(val_df)),
                "num_breeds": int(len(active_breeds)),
                "run_id": result.run_id,
                "artifact_root": str(artifact_root.resolve()),
                "best_checkpoint": str(best_dst.resolve()),
                "last_checkpoint": str(last_dst.resolve()) if last_dst.exists() else "",
                "best_top1": float(result.best_top1),
                "metrics_csv": str(metrics_csv.resolve()) if metrics_csv.exists() else "",
            }
        )

    if start_species and not started:
        raise ValueError(f"Requested start_species={start_species} was not found in the taxonomy order.")

    summary_df = pd.DataFrame(summaries)
    if summary_df.empty:
        raise ValueError(
            "No species-conditioned breed models were trained. "
            "Check that the temporary breed manifest contains train/val rows and at least two breeds per species."
        )
    if not summary_df.empty:
        summary_df.to_csv(bundle.metrics_dir / "two_stage_breed_models_summary.csv", index=False)
    write_json(bundle.metrics_dir / "two_stage_breed_models_summary.json", {"models": summaries})
    write_json(bundle.qa_dir / "data_quality_report.json", qa_report)
    logger.info("Two-stage breed training complete. trained_species_models=%d models_root=%s", len(summaries), models_root)


if __name__ == "__main__":
    main()
