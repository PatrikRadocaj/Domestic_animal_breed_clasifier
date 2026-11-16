"""Train a timm breed classifier from the split manifest via shared trainer core."""

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
    logger.info("Training classifier with config:\n%s", OmegaConf.to_yaml(cfg.model.classifier))
    logger.info("Run metadata: %s", run_metadata(cfg, repo_root="."))

    manifest = read_manifest(cfg.data.split_manifest_path)

    _, qa_report = run_quality_checks(manifest, cfg.data.taxonomy_path, dict(cfg.data.quality))
    if not qa_report["valid"]:
        raise ValueError("Data quality gate failed before classifier training")

    taxonomy = load_taxonomy(cfg.data.taxonomy_path)
    breed_ids = [record["breed_id"] for record in taxonomy.breed_records()]
    active_breeds = sorted(set(manifest["breed_id"]) & set(breed_ids))
    label_to_index = {breed_id: idx for idx, breed_id in enumerate(active_breeds)}
    if not label_to_index:
        raise ValueError("No active breeds from manifest match taxonomy.")

    result = train_classifier(
        cfg=cfg,
        manifest=manifest,
        label_to_index=label_to_index,
        label_column="breed_id",
        run_prefix="classifier",
        logger=logger,
    )
    write_json(
        f"{result.artifact_root}/qa/data_quality_report.json",
        qa_report,
    )
    logger.info("Training complete. run_id=%s best_top1=%.4f", result.run_id, result.best_top1)


if __name__ == "__main__":
    main()
