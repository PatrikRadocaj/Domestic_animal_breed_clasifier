"""Train hierarchical EfficientNetV2 pipeline: species model + per-species breed models."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from animal_breed.data.manifest_schema import read_manifest
from animal_breed.data.quality.qa import run_quality_checks
from animal_breed.data.taxonomy import load_taxonomy
from animal_breed.trainers.classifier_trainer import train_classifier
from animal_breed.utils.artifacts import write_json
from animal_breed.utils.logging import setup_logging
from animal_breed.utils.reproducibility import run_metadata, seed_everything


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
    logger.info("Training hierarchical classifier with config:\n%s", OmegaConf.to_yaml(cfg.model.classifier))
    logger.info("Run metadata: %s", run_metadata(cfg, repo_root="."))

    manifest = read_manifest(cfg.data.split_manifest_path)
    _, qa_report = run_quality_checks(manifest, cfg.data.taxonomy_path, dict(cfg.data.quality))
    if not qa_report["valid"]:
        raise ValueError("Data quality gate failed before hierarchical training")

    taxonomy = load_taxonomy(cfg.data.taxonomy_path)

    # Stage 1: species model
    species_ids = sorted(set(manifest["species_id"].dropna().astype(str)))
    label_to_index_species = {species_id: i for i, species_id in enumerate(species_ids)}
    if len(label_to_index_species) < 2:
        raise ValueError("Need at least 2 species classes for stage-1 training.")

    species_result = train_classifier(
        cfg=cfg,
        manifest=manifest,
        label_to_index=label_to_index_species,
        label_column="species_id",
        run_prefix="hierarchical-species",
        logger=logger,
    )
    write_json(f"{species_result.artifact_root}/qa/data_quality_report.json", qa_report)

    # Stage 2: per-species breed models
    stage2_runs: list[dict[str, str]] = []
    for species in taxonomy.species_records():
        species_id = species["species_id"]
        species_manifest = manifest[manifest["species_id"] == species_id].reset_index(drop=True)
        if species_manifest.empty:
            continue

        train_df = species_manifest[species_manifest["split"] == "train"]
        val_df = species_manifest[species_manifest["split"] == "val"]
        if train_df.empty or val_df.empty:
            continue

        active_breeds = sorted(set(species_manifest["breed_id"].dropna().astype(str)))
        active_breeds = [b for b in active_breeds if b.startswith(f"{species_id}:")]
        if len(active_breeds) < 2:
            continue

        label_to_index_breed = {breed_id: i for i, breed_id in enumerate(active_breeds)}
        run = train_classifier(
            cfg=cfg,
            manifest=species_manifest,
            label_to_index=label_to_index_breed,
            label_column="breed_id",
            run_prefix=f"hierarchical-breed-{species_id}",
            logger=logger,
        )
        write_json(f"{run.artifact_root}/qa/data_quality_report.json", qa_report)
        stage2_runs.append({"species_id": species_id, "run_id": run.run_id, "artifact_root": run.artifact_root})

    logger.info("Stage1 run_id=%s, stage2_models=%d", species_result.run_id, len(stage2_runs))


if __name__ == "__main__":
    main()
