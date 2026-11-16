"""Strict review/finalization pass for temporary breed runs.

Outputs:
1. Stage-2 trainable manifest/taxonomy with only high-confidence real breed classes.
2. Direct-YOLO manifest/taxonomy where valid breed species stay breed-level and
   species without valid breed splits collapse to species-level outputs.
3. Review reports with provenance for exclusions and manual-review cases.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any

import cv2
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from animal_breed.data.label_mapping import CrossDatasetLabelMap
from animal_breed.data.quality.qa import run_quality_checks


LABEL_MAP = CrossDatasetLabelMap.from_file(REPO_ROOT / "configs" / "mappings" / "cross_dataset_label_map.yaml")

try:
    cv2.setLogLevel(0)
except Exception:
    pass

logging.getLogger("PIL").setLevel(logging.ERROR)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

PLACEHOLDER_SLUGS = {
    "unknown",
    "unlabeled",
    "unidentified",
    "unidentified_mixed",
    "mixed",
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
    "hybrid",
    "other",
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

NOISE_TOKENS = {
    "animal",
    "animals",
    "breed",
    "breeds",
    "class",
    "classes",
    "dataset",
    "image",
    "images",
}


def _normalize_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _context_stripped_slug(slug: str) -> str:
    out = slug
    if "_with_other_species" in out:
        out = out.split("_with_other_species", 1)[0].strip("_")
    if "_in_" in out:
        base = out.split("_in_", 1)[0].strip("_")
        if base:
            out = base
    if out.endswith("_in"):
        out = out[: -len("_in")].strip("_")
    return out


def _simple_canonical_slug(raw_label: str) -> str:
    norm = _normalize_slug(raw_label)
    if not norm:
        return ""
    tokens = [tok for tok in norm.split("_") if tok and tok not in NOISE_TOKENS]
    cleaned = "_".join(tokens).strip("_")
    return _context_stripped_slug(cleaned or norm)


def _is_placeholder(slug: str) -> bool:
    if not slug or slug in PLACEHOLDER_SLUGS:
        return True
    tokens = [tok for tok in slug.split("_") if tok]
    return any(tok in PLACEHOLDER_TOKENS for tok in tokens)


def _readable_image_reason(path: Path) -> str | None:
    if not path.exists():
        return "missing_file"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return "unreadable_image"
    return None


def _write_taxonomy_from_groups(
    *,
    out_path: Path,
    species_ids: list[str],
    breed_alias_groups: dict[str, set[str]],
) -> None:
    payload: dict[str, Any] = {
        "version": "temporary_breed_run_v2_strict",
        "description": "Strict reviewed taxonomy for current run.",
        "unknown_policy": {
            "species_id": "unknown_species",
            "breed_id": "unknown_breed",
            "allow_unknown_breed_for_training": False,
        },
        "species": [{"id": sid, "name": sid.replace("_", " ").title(), "aliases": [sid]} for sid in species_ids],
        "breeds": {sid: [] for sid in species_ids},
    }

    for breed_id in sorted(breed_alias_groups):
        species_id, breed_slug = breed_id.split(":", 1)
        aliases = sorted({_normalize_slug(x) for x in breed_alias_groups[breed_id] if str(x).strip()} | {breed_slug})
        payload["breeds"].setdefault(species_id, []).append(
            {
                "id": breed_id,
                "name": breed_slug.replace("_", " ").title(),
                "aliases": aliases,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _review_manifest(df: pd.DataFrame) -> pd.DataFrame:
    review_rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        species_id = str(row["species_id"])
        source = str(row.get("source", "")).strip()
        raw_label = str(row.get("raw_label", "")).strip()
        current_breed_id = str(row.get("breed_id", "")).strip().lower()
        current_slug = current_breed_id.split(":", 1)[-1] if ":" in current_breed_id else _normalize_slug(current_breed_id)
        raw_norm = LABEL_MAP.normalize(raw_label)
        simple_slug = _simple_canonical_slug(raw_label)
        species_resolution = LABEL_MAP.resolve_species(raw_label, source) if raw_label else None
        breed_resolution = LABEL_MAP.resolve_breed(species_id, raw_label, source) if raw_label else None
        image_reason = _readable_image_reason(Path(str(row["path"])))

        review_status = "exclude"
        review_reason = ""
        review_confidence = "excluded"
        canonical_breed_id = ""
        use_for_stage2 = False
        keep_for_direct = False
        explicit_species_label = False

        if image_reason:
            review_reason = image_reason
        elif not raw_label:
            review_reason = "missing_raw_label"
        elif _is_placeholder(raw_norm) or _is_placeholder(simple_slug):
            review_reason = "placeholder_label"
        elif species_resolution and species_resolution.status == "resolved" and species_resolution.canonical_id == species_id:
            review_status = "species_level"
            review_reason = "species_level_label"
            review_confidence = "high"
            explicit_species_label = True
            keep_for_direct = True
        elif species_id == "cat" and simple_slug in CAT_NON_BREED_SLUGS:
            review_reason = "cat_non_breed_label"
        elif breed_resolution and breed_resolution.status == "resolved" and breed_resolution.canonical_id:
            canonical_breed_id = str(breed_resolution.canonical_id)
            review_status = "keep_breed"
            review_reason = breed_resolution.reason
            review_confidence = "high"
            use_for_stage2 = True
            keep_for_direct = True
        elif breed_resolution and breed_resolution.status == "manual_review":
            review_status = "manual_review"
            review_reason = breed_resolution.reason
            review_confidence = "review_needed"
        else:
            if simple_slug and not _is_placeholder(simple_slug):
                heuristic_breed_id = f"{species_id}:{simple_slug}"
                if simple_slug == current_slug or raw_norm == current_slug or _context_stripped_slug(raw_norm) == current_slug:
                    canonical_breed_id = heuristic_breed_id
                    review_status = "keep_breed"
                    review_reason = "normalized_spelling_variant"
                    review_confidence = "high"
                    use_for_stage2 = True
                    keep_for_direct = True
                else:
                    review_status = "manual_review"
                    review_reason = "unmapped_label_needs_review"
                    review_confidence = "review_needed"
                    canonical_breed_id = heuristic_breed_id
            else:
                review_reason = "empty_canonical_slug"

        reviewed = dict(row)
        reviewed["raw_label_normalized"] = raw_norm
        reviewed["canonical_breed_id"] = canonical_breed_id
        reviewed["review_status"] = review_status
        reviewed["review_reason"] = review_reason
        reviewed["review_confidence"] = review_confidence
        reviewed["use_for_stage2"] = bool(use_for_stage2)
        reviewed["keep_for_direct"] = bool(keep_for_direct)
        reviewed["explicit_species_label"] = bool(explicit_species_label)
        review_rows.append(reviewed)

    return pd.DataFrame(review_rows)


def finalize_manifest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    reviewed = _review_manifest(df)

    breed_candidates = reviewed.loc[reviewed["use_for_stage2"]].copy()
    breed_counts = (
        breed_candidates.groupby("species_id")["canonical_breed_id"].nunique().sort_values(ascending=False).to_dict()
    )
    valid_stage2_species = {species_id for species_id, count in breed_counts.items() if int(count) >= 2}

    stage2_df = breed_candidates.loc[breed_candidates["species_id"].isin(valid_stage2_species)].copy()
    stage2_df["breed_id"] = stage2_df["canonical_breed_id"]
    stage2_df = stage2_df.drop(
        columns=[
            "canonical_breed_id",
            "review_status",
            "review_reason",
            "review_confidence",
            "use_for_stage2",
            "keep_for_direct",
            "explicit_species_label",
            "raw_label_normalized",
        ],
        errors="ignore",
    )

    direct_df = reviewed.loc[reviewed["keep_for_direct"]].copy()
    direct_df["direct_label_id"] = ""
    breed_mask = direct_df["species_id"].isin(valid_stage2_species) & direct_df["review_status"].eq("keep_breed")
    direct_df.loc[breed_mask, "direct_label_id"] = direct_df.loc[breed_mask, "canonical_breed_id"]
    direct_df.loc[~breed_mask, "direct_label_id"] = direct_df.loc[~breed_mask, "species_id"]
    direct_df = direct_df.loc[direct_df["direct_label_id"].astype(str).str.strip().ne("")].copy()

    summary = {
        "rows_in": int(len(df)),
        "rows_reviewed": int(len(reviewed)),
        "rows_stage2": int(len(stage2_df)),
        "rows_direct": int(len(direct_df)),
        "stage2_species_count": int(stage2_df["species_id"].nunique()) if not stage2_df.empty else 0,
        "stage2_breed_count": int(stage2_df["breed_id"].nunique()) if not stage2_df.empty else 0,
        "direct_class_count": int(direct_df["direct_label_id"].nunique()) if not direct_df.empty else 0,
        "valid_stage2_species": sorted(valid_stage2_species),
        "species_with_no_stage2_split": sorted(set(reviewed["species_id"].astype(str)) - valid_stage2_species),
        "review_status_counts": reviewed["review_status"].value_counts(dropna=False).to_dict(),
        "review_reason_counts": reviewed["review_reason"].value_counts(dropna=False).to_dict(),
        "breed_counts_by_species_pre_filter": breed_counts,
    }
    return stage2_df.reset_index(drop=True), direct_df.reset_index(drop=True), summary, reviewed.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--taxonomy", required=False, default="")
    parser.add_argument("--out-manifest-csv", required=True)
    parser.add_argument("--out-manifest-parquet", required=True)
    parser.add_argument("--out-taxonomy", required=True)
    parser.add_argument("--out-direct-manifest-csv", required=True)
    parser.add_argument("--out-direct-manifest-parquet", required=True)
    parser.add_argument("--out-direct-taxonomy", required=True)
    parser.add_argument("--review-report-csv", required=True)
    parser.add_argument("--review-report-json", required=True)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv)
    out_manifest_csv = Path(args.out_manifest_csv)
    out_manifest_parquet = Path(args.out_manifest_parquet)
    out_taxonomy = Path(args.out_taxonomy)
    out_direct_manifest_csv = Path(args.out_direct_manifest_csv)
    out_direct_manifest_parquet = Path(args.out_direct_manifest_parquet)
    out_direct_taxonomy = Path(args.out_direct_taxonomy)
    review_report_csv = Path(args.review_report_csv)
    review_report_json = Path(args.review_report_json)

    source_df = pd.read_csv(manifest_csv)
    stage2_df, direct_df, summary, reviewed = finalize_manifest(source_df)

    if stage2_df.empty:
        raise SystemExit("Strict sanitation produced an empty stage-2 manifest.")
    if direct_df.empty:
        raise SystemExit("Strict sanitation produced an empty direct-YOLO manifest.")

    out_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    stage2_df.to_csv(out_manifest_csv, index=False)
    stage2_df.to_parquet(out_manifest_parquet, index=False)
    direct_df.to_csv(out_direct_manifest_csv, index=False)
    direct_df.to_parquet(out_direct_manifest_parquet, index=False)
    reviewed.to_csv(review_report_csv, index=False)

    stage2_aliases: dict[str, set[str]] = {}
    for row in stage2_df.to_dict(orient="records"):
        stage2_aliases.setdefault(str(row["breed_id"]), set()).add(str(row.get("raw_label", "")))
    _write_taxonomy_from_groups(
        out_path=out_taxonomy,
        species_ids=sorted(stage2_df["species_id"].dropna().astype(str).unique().tolist()),
        breed_alias_groups=stage2_aliases,
    )

    direct_breed_aliases: dict[str, set[str]] = {}
    breed_rows = direct_df.loc[direct_df["direct_label_id"].astype(str).str.contains(":", regex=False)].copy()
    for row in breed_rows.to_dict(orient="records"):
        direct_breed_aliases.setdefault(str(row["direct_label_id"]), set()).add(str(row.get("raw_label", "")))
    _write_taxonomy_from_groups(
        out_path=out_direct_taxonomy,
        species_ids=sorted(direct_df["species_id"].dropna().astype(str).unique().tolist()),
        breed_alias_groups=direct_breed_aliases,
    )

    _, qa_report = run_quality_checks(stage2_df, out_taxonomy, {"strictness": "benchmark"})
    if not qa_report["valid"]:
        raise SystemExit("Strict stage-2 manifest still fails quality checks.")

    payload = {
        "summary": summary,
        "qa": qa_report,
        "source_manifest": str(manifest_csv.resolve()),
        "stage2_manifest_csv": str(out_manifest_csv.resolve()),
        "direct_manifest_csv": str(out_direct_manifest_csv.resolve()),
        "review_report_csv": str(review_report_csv.resolve()),
    }
    review_report_json.parent.mkdir(parents=True, exist_ok=True)
    review_report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
