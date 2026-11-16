"""CLI for dataset quality gate."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from animal_breed.data.manifest_schema import read_manifest
from animal_breed.data.quality.qa import run_quality_checks
from animal_breed.utils.artifacts import build_run_id, create_artifact_bundle, write_json
from animal_breed.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    manifest = read_manifest(cfg.data.manifest_path)
    run_id = build_run_id("dataqa", cfg)
    bundle = create_artifact_bundle(run_id, artifacts_root=str(cfg.paths.artifacts_dir))

    _, report = run_quality_checks(
        manifest,
        taxonomy_path=cfg.data.taxonomy_path,
        config=dict(cfg.data.quality),
    )
    out_path = bundle.qa_dir / "data_quality_report.json"
    write_json(out_path, report)
    logger.info("Quality report: %s", out_path)
    if not report["valid"]:
        raise ValueError("Data quality gate failed")


if __name__ == "__main__":
    main()
