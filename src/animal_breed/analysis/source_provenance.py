"""Generate source provenance reports from the canonical manifest and optional source registry."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import hydra
import pandas as pd
from omegaconf import DictConfig
import os

from animal_breed.data.manifest_schema import read_manifest
from animal_breed.sync.datasource_sync import DatasourceSyncTransaction
from animal_breed.utils.artifacts import build_snapshot_hashes, write_json
from animal_breed.utils.logging import setup_logging


def _canonical_website(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _infer_group(source_type: str, website: str) -> str:
    normalized_type = source_type.strip().lower()
    if normalized_type in {"wikimedia", "inaturalist", "feathersite", "scraped"}:
        return "scraped"
    if website in {"commons.wikimedia.org", "inaturalist.org", "feathersite.com"}:
        return "scraped"
    return "normal_download"


def _write_current_sources(sources_df: pd.DataFrame, output_dir: Path) -> None:
    current_csv = output_dir / "sources_used_current_from_manifest.csv"
    current_txt = output_dir / "sources_used_current_from_manifest.txt"
    current = sources_df[["source", "source_url"]].drop_duplicates().sort_values(["source", "source_url"])
    current.to_csv(current_csv, index=False)

    lines = [f"Sources used in manifest: {len(current)}"]
    for row in current.itertuples(index=False):
        lines.append(f"- {row.source}: {row.source_url}")
    current_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_grouped_sources(sources_df: pd.DataFrame, output_dir: Path) -> None:
    grouped_df = sources_df.copy()
    grouped_df["website"] = grouped_df["source_url"].map(_canonical_website)
    grouped_df["group"] = [
        _infer_group(str(source_type), str(website))
        for source_type, website in zip(grouped_df["source_type"], grouped_df["website"])
    ]
    grouped_df = grouped_df[["group", "website", "source", "source_url", "manifest_count"]]
    grouped_df = grouped_df.sort_values(["group", "website", "source"]).reset_index(drop=True)

    grouped_csv = output_dir / "sources_websites_grouped.csv"
    grouped_df.to_csv(grouped_csv, index=False)

    lines: list[str] = ["Sources grouped by acquisition type (current manifest)", ""]
    for group_name in ["scraped", "normal_download"]:
        subset = grouped_df[grouped_df["group"] == group_name]
        if subset.empty:
            continue
        lines.append(f"[{group_name}]")
        websites = subset.groupby("website")
        for website, website_rows in websites:
            lines.append(f"- {website} ({len(website_rows)} datasets)")
            for row in website_rows.itertuples(index=False):
                lines.append(
                    f"  - {row.source}: {row.source_url} (manifest_count={int(row.manifest_count)})"
                )
        lines.append("")

    (output_dir / "sources_websites_grouped.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_pending_sources(
    source_registry: pd.DataFrame,
    sources_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if source_registry.empty:
        return

    used_source_ids = set(sources_df["source"].astype(str))
    registry = source_registry.copy()
    registry["dataset_id"] = registry["dataset_id"].astype(str)
    registry["status"] = registry["status"].fillna("").astype(str)

    pending = registry[~registry["dataset_id"].isin(used_source_ids)].copy()
    pending = pending.sort_values(["status", "dataset_id"]).reset_index(drop=True)
    if not pending.empty:
        pending[["dataset_id", "source_url", "status", "notes"]].to_csv(
            output_dir / "dataset_sources_pending_raw.csv", index=False
        )

    used = registry[registry["dataset_id"].isin(used_source_ids)].copy()
    used = used.sort_values(["dataset_id"]).reset_index(drop=True)
    used[["dataset_id", "source_url", "status", "notes"]].to_csv(
        output_dir / "dataset_sources_used_registry.csv", index=False
    )


def build_provenance_reports(cfg: DictConfig) -> dict[str, int]:
    manifest = read_manifest(cfg.data.manifest_path)
    output_dir = Path(str(getattr(cfg.data, "source_reports_output_dir", "results/reports/sources")))
    output_dir.mkdir(parents=True, exist_ok=True)

    if "source_url" not in manifest.columns:
        manifest["source_url"] = ""
    if "source_type" not in manifest.columns:
        manifest["source_type"] = ""

    sources = (
        manifest.groupby(["source", "source_url", "source_type"], dropna=False)
        .size()
        .reset_index(name="manifest_count")
        .sort_values("manifest_count", ascending=False)
        .reset_index(drop=True)
    )

    _write_current_sources(sources, output_dir)
    _write_grouped_sources(sources, output_dir)

    source_registry_path = Path(str(getattr(cfg.data, "source_registry_path", "")))
    source_registry = pd.DataFrame()
    if source_registry_path.exists():
        source_registry = pd.read_csv(source_registry_path)
        required_cols = ["dataset_id", "source_url", "status", "notes"]
        for col in required_cols:
            if col not in source_registry.columns:
                source_registry[col] = ""
    _write_pending_sources(source_registry, sources, output_dir)

    snapshot_hashes = build_snapshot_hashes(
        cfg,
        manifest_path=cfg.data.manifest_path,
        taxonomy_path=cfg.data.taxonomy_path,
        source_registry_path=getattr(cfg.data, "source_registry_path", None),
    )
    write_json(output_dir / "snapshot_hashes.json", snapshot_hashes)

    return {
        "manifest_rows": int(len(manifest)),
        "unique_sources": int(sources["source"].nunique()),
        "unique_source_urls": int(sources["source_url"].replace("", pd.NA).dropna().nunique()),
    }


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    origin_session_id = os.getenv("CODEX_THREAD_ID", "unknown_session")
    with DatasourceSyncTransaction(repo_root=".", origin_session_id=origin_session_id):
        stats = build_provenance_reports(cfg)
    logger.info("Source provenance reports written: %s", stats)


if __name__ == "__main__":
    main()
