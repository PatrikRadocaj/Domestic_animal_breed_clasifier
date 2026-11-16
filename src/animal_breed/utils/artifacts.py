"""Artifact contract and reproducibility snapshot helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from animal_breed.utils.reproducibility import git_commit, git_status_short


@dataclass(frozen=True)
class ArtifactBundle:
    run_id: str
    root: Path
    checkpoints_dir: Path
    metrics_dir: Path
    predictions_dir: Path
    provenance_dir: Path
    qa_dir: Path


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def stable_text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    digest = sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_hash(df: pd.DataFrame) -> str:
    csv_blob = df.sort_index(axis=1).to_csv(index=False)
    return stable_text_hash(csv_blob)


def build_run_id(prefix: str, cfg: DictConfig) -> str:
    cfg_hash = stable_text_hash(OmegaConf.to_yaml(cfg))[:8]
    return f"{prefix}-{_utc_stamp()}-{cfg_hash}"


def create_artifact_bundle(run_id: str, artifacts_root: str | Path = "artifacts") -> ArtifactBundle:
    root = Path(artifacts_root) / run_id
    checkpoints = root / "checkpoints"
    metrics = root / "metrics"
    preds = root / "predictions"
    prov = root / "provenance"
    qa = root / "qa"
    for path in (checkpoints, metrics, preds, prov, qa):
        path.mkdir(parents=True, exist_ok=True)
    return ArtifactBundle(
        run_id=run_id,
        root=root,
        checkpoints_dir=checkpoints,
        metrics_dir=metrics,
        predictions_dir=preds,
        provenance_dir=prov,
        qa_dir=qa,
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_cfg_yaml(cfg: DictConfig) -> str:
    try:
        return OmegaConf.to_yaml(cfg, resolve=True)
    except Exception:
        return OmegaConf.to_yaml(cfg, resolve=False)


def _safe_cfg_container(cfg: DictConfig) -> dict[str, Any]:
    try:
        raw = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        raw = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    return {"config": raw}


def build_snapshot_hashes(
    cfg: DictConfig,
    manifest_path: str | Path,
    taxonomy_path: str | Path,
    source_registry_path: str | Path | None,
) -> dict[str, str]:
    return {
        "manifest_hash": file_hash(manifest_path),
        "taxonomy_hash": file_hash(taxonomy_path),
        "source_registry_hash": file_hash(source_registry_path) if source_registry_path else "missing",
        "resolved_config_hash": stable_text_hash(_safe_cfg_yaml(cfg)),
    }


def write_run_metadata(
    bundle: ArtifactBundle,
    cfg: DictConfig,
    snapshot_hashes: dict[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": bundle.run_id,
        "timestamp_utc": _utc_stamp(),
        "seed": int(cfg.seed),
        "deterministic": bool(cfg.train.deterministic),
        "git_commit": git_commit("."),
        "git_status_short": git_status_short("."),
        "snapshot_hashes": snapshot_hashes,
        "config": _safe_cfg_container(cfg),
    }
    write_json(bundle.provenance_dir / "run_metadata.json", payload)
    write_json(bundle.provenance_dir / "snapshot_hashes.json", snapshot_hashes)
    return payload
