"""Sanitize temporary breed manifests for trainable subclass labels.

This derives a cleaned run-scoped manifest/taxonomy from the temporary breed
run outputs, removing placeholder labels, species-self labels, and obvious
context/noise buckets while canonicalizing simple suffix variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from animal_breed.data.quality.qa import run_quality_checks


PLACEHOLDER_SLUGS = {
    "unknown",
    "unlabeled",
    "unidentified",
    "unidentified_mixed",
    "mixed",
    "local",
    "other",
    "na",
    "n_a",
    "none",
}

PLACEHOLDER_TOKENS = {
    "unknown",
    "unlabeled",
    "unidentified",
    "mixed",
    "local",
    "other",
    "hybrid",
}

SPECIES_SELF_ALIASES: dict[str, set[str]] = {
    "alpaca": {"alpaca", "vicugna_pacos"},
    "camel": {"camel", "camelus"},
    "dromedary": {"dromedary", "camelus_dromedarius"},
    "duck": {"duck"},
    "goose": {"goose"},
    "llama": {"llama", "lama", "lama_glama"},
    "peafowl": {"peafowl", "peafowls"},
    "turkey": {"turkey"},
}

CAT_NON_BREED_SLUGS = {
    "short_hair",
    "medium_hair",
    "long_hair",
    "tortoiseshell",
    "torbie",
    "dilute_tortoiseshell",
    "trimaps",
    "dilute_calico",
    "tabby",
    "tuxedo",
    "calico",
    "tiger",
    "extra_toes_hemingway_polydactyl",
    "local",
    "mixed",
}


def _normalize_slug(slug: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(slug).strip().lower()).strip("_")


def _context_stripped_slug(species_id: str, slug: str, species_slug_set: set[str]) -> str:
    out = slug
    numeric = re.match(r"(.+)_\d+$", out)
    if numeric and numeric.group(1) in species_slug_set:
        out = numeric.group(1)

    if "_with_other_species" in out:
        out = out.split("_with_other_species", 1)[0].strip("_")

    if "_in_" in out:
        base = out.split("_in_", 1)[0].strip("_")
        if base:
            out = base

    if out.startswith("at_") or out.startswith("in_"):
        return out

    if out.endswith("_in"):
        return out

    return out


def _exclude_reason(species_id: str, slug: str) -> str | None:
    if slug in PLACEHOLDER_SLUGS:
        return "placeholder_slug"

    tokens = [tok for tok in slug.split("_") if tok]
    if any(tok in PLACEHOLDER_TOKENS for tok in tokens):
        return "placeholder_token"

    if slug.endswith("_cross") or slug.endswith("cross") or "cross" in tokens:
        return "crossbreed_bucket"

    if species_id in SPECIES_SELF_ALIASES and slug in SPECIES_SELF_ALIASES[species_id]:
        return "species_self_label"

    if species_id == "cat" and slug in CAT_NON_BREED_SLUGS:
        return "cat_non_breed_label"

    if slug.startswith("at_") or slug.startswith("in_"):
        return "context_only_label"

    if slug.endswith("_in"):
        return "context_only_label"

    return None


def _write_taxonomy_from_manifest(df: pd.DataFrame, out_path: Path) -> None:
    species_ids = sorted(df["species_id"].dropna().astype(str).unique().tolist())
    payload: dict[str, Any] = {
        "version": "temporary_breed_run_v1_trainable",
        "description": "Sanitized run-scoped taxonomy generated for breed/subclass training.",
        "unknown_policy": {
            "species_id": "unknown_species",
            "breed_id": "unknown_breed",
            "allow_unknown_breed_for_training": False,
        },
        "species": [{"id": sid, "name": sid.replace("_", " ").title(), "aliases": [sid]} for sid in species_ids],
        "breeds": {sid: [] for sid in species_ids},
    }

    for (species_id, breed_id), group in df.groupby(["species_id", "breed_id"], dropna=False):
        if pd.isna(species_id) or pd.isna(breed_id):
            continue
        slug = str(breed_id).split(":", 1)[-1]
        aliases = sorted({_normalize_slug(x) for x in group["raw_label"].dropna().astype(str) if str(x).strip()} | {slug})
        payload["breeds"][str(species_id)].append(
            {
                "id": str(breed_id),
                "name": slug.replace("_", " ").title(),
                "aliases": aliases,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def sanitize_manifest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = df.copy()
    cleaned["species_id"] = cleaned["species_id"].astype(str)
    cleaned["breed_id"] = cleaned["breed_id"].astype(str)

    species_to_slugs: dict[str, set[str]] = {}
    for species_id, grp in cleaned.groupby("species_id"):
        species_to_slugs[str(species_id)] = {str(b).split(":", 1)[-1] for b in grp["breed_id"].dropna().astype(str)}

    canonical_map: dict[str, str | None] = {}
    exclusion_rows: list[dict[str, Any]] = []
    merge_rows: list[dict[str, Any]] = []

    for breed_id in sorted(cleaned["breed_id"].dropna().astype(str).unique()):
        species_id, slug = breed_id.split(":", 1)
        species_slugs = species_to_slugs.get(species_id, set())
        stripped = _context_stripped_slug(species_id, slug, species_slugs)
        reason = _exclude_reason(species_id, stripped)
        if reason is not None:
            canonical_map[breed_id] = None
            exclusion_rows.append({"breed_id": breed_id, "species_id": species_id, "reason": reason})
            continue
        canonical_breed_id = f"{species_id}:{stripped}"
        canonical_map[breed_id] = canonical_breed_id
        if canonical_breed_id != breed_id:
            merge_rows.append({"from": breed_id, "to": canonical_breed_id, "reason": "canonicalized_context_suffix"})

    cleaned["sanitized_breed_id"] = cleaned["breed_id"].map(canonical_map)
    excluded_mask = cleaned["sanitized_breed_id"].isna()
    excluded = cleaned.loc[excluded_mask].copy()
    kept = cleaned.loc[~excluded_mask].copy()
    kept["breed_id"] = kept["sanitized_breed_id"].astype(str)
    kept = kept.drop(columns=["sanitized_breed_id"])

    report = {
        "rows_in": int(len(df)),
        "rows_out": int(len(kept)),
        "rows_excluded": int(len(excluded)),
        "unique_breeds_in": int(df["breed_id"].nunique()),
        "unique_breeds_out": int(kept["breed_id"].nunique()),
        "species_in": int(df["species_id"].nunique()),
        "species_out": int(kept["species_id"].nunique()),
        "excluded_breeds": exclusion_rows,
        "merged_breeds": merge_rows,
        "excluded_counts_by_species": excluded.groupby("species_id").size().sort_values(ascending=False).to_dict(),
        "remaining_breed_counts_by_species": kept.groupby("species_id")["breed_id"].nunique().sort_values().to_dict(),
    }
    return kept.reset_index(drop=True), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--out-manifest-csv", required=True)
    parser.add_argument("--out-manifest-parquet", required=True)
    parser.add_argument("--out-taxonomy", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv)
    taxonomy_path = Path(args.taxonomy)
    out_manifest_csv = Path(args.out_manifest_csv)
    out_manifest_parquet = Path(args.out_manifest_parquet)
    out_taxonomy = Path(args.out_taxonomy)
    report_json = Path(args.report_json)

    df = pd.read_csv(manifest_csv)
    kept, report = sanitize_manifest(df)

    out_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_manifest_csv, index=False)
    kept.to_parquet(out_manifest_parquet, index=False)
    _write_taxonomy_from_manifest(kept, out_taxonomy)

    _, qa_report = run_quality_checks(kept, out_taxonomy, {"strictness": "benchmark"})
    payload = {
        "sanitize": report,
        "qa": qa_report,
        "source_manifest": str(manifest_csv.resolve()),
        "source_taxonomy": str(taxonomy_path.resolve()),
        "out_manifest_csv": str(out_manifest_csv.resolve()),
        "out_manifest_parquet": str(out_manifest_parquet.resolve()),
        "out_taxonomy": str(out_taxonomy.resolve()),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not qa_report["valid"]:
        raise SystemExit("Sanitized manifest still fails quality gate. See report_json.")


if __name__ == "__main__":
    main()
