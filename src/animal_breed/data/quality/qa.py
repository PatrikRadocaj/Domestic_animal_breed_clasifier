"""Data quality gates for training and evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from animal_breed.data.taxonomy import load_taxonomy


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    count: int
    details: str


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _a_hash(path: Path) -> str | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(resized.mean())
    bits = (resized > mean).astype("uint8").reshape(-1)
    return "".join("1" if int(v) else "0" for v in bits)


def run_quality_checks(
    df: pd.DataFrame,
    taxonomy_path: str | Path,
    config: dict[str, Any],
) -> tuple[list[QualityIssue], dict[str, Any]]:
    issues: list[QualityIssue] = []

    bbox_rows = df[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "width", "height"]].copy()
    complete = ~bbox_rows[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].isna().any(axis=1)
    bad_bbox = 0
    bad_aspect = 0
    max_aspect = float(config.get("max_bbox_aspect", 12.0))
    for row in bbox_rows[complete].itertuples(index=False):
        xmin, ymin, xmax, ymax, w, h = map(float, row)
        if not (0 <= xmin < xmax <= w and 0 <= ymin < ymax <= h):
            bad_bbox += 1
            continue
        bw = max(1e-6, xmax - xmin)
        bh = max(1e-6, ymax - ymin)
        aspect = max(bw / bh, bh / bw)
        if aspect > max_aspect:
            bad_aspect += 1

    if bad_bbox:
        issues.append(QualityIssue("bbox_bounds", "error", bad_bbox, "Invalid bbox coordinates"))
    if bad_aspect:
        issues.append(QualityIssue("bbox_aspect", "warning", bad_aspect, f"Aspect>{max_aspect}"))

    taxonomy = load_taxonomy(taxonomy_path)
    hierarchy_bad = 0
    for row in df[["species_id", "breed_id"]].dropna().itertuples(index=False):
        expected = taxonomy.breed_species(str(row.breed_id))
        if expected is not None and expected != str(row.species_id):
            hierarchy_bad += 1
    if hierarchy_bad:
        issues.append(QualityIssue("taxonomy_hierarchy", "error", hierarchy_bad, "species_id/breed_id mismatch"))

    split_col = str(config.get("split_column", "split"))
    split_leaks = 0
    if split_col in df.columns:
        for col in ["duplicate_cluster", "split_group"]:
            if col in df.columns:
                grouped = df.groupby(col)[split_col].nunique(dropna=True)
                split_leaks += int((grouped > 1).sum())
    if split_leaks:
        issues.append(QualityIssue("split_leakage", "error", split_leaks, "Duplicate clusters cross splits"))

    source_col = str(config.get("source_column", "source"))
    source_leaks = 0
    if source_col in df.columns:
        source_leaks = int(df.groupby("path")[source_col].nunique(dropna=True).gt(1).sum())
    if source_leaks:
        issues.append(QualityIssue("source_leakage", "warning", source_leaks, "Same path mapped to multiple sources"))

    dedup_cfg = config.get("dedup", {})
    exact_dup = 0
    near_dup = 0
    exact_groups: dict[str, list[str]] = defaultdict(list)
    near_groups: dict[str, list[str]] = defaultdict(list)

    if bool(dedup_cfg.get("enabled", True)):
        max_images = int(dedup_cfg.get("max_images", 5000))
        sample_paths = [Path(str(p)) for p in df["path"].head(max_images)]
        hash_counter: Counter[str] = Counter()
        ahash_counter: Counter[str] = Counter()
        for p in sample_paths:
            if not p.exists() or not p.is_file():
                continue
            try:
                file_hash = _sha256_file(p)
                hash_counter[file_hash] += 1
                if hash_counter[file_hash] > 1:
                    exact_groups[file_hash].append(str(p))
                if bool(dedup_cfg.get("near_enabled", True)):
                    ahash = _a_hash(p)
                    if ahash is not None:
                        ahash_counter[ahash] += 1
                        if ahash_counter[ahash] > 1:
                            near_groups[ahash].append(str(p))
            except OSError:
                continue

        exact_dup = int(sum(c - 1 for c in hash_counter.values() if c > 1))
        near_dup = int(sum(c - 1 for c in ahash_counter.values() if c > 1))

    if exact_dup:
        issues.append(QualityIssue("exact_duplicates", "warning", exact_dup, "Exact SHA256 duplicates in sampled set"))
    if near_dup:
        issues.append(QualityIssue("near_duplicates", "warning", near_dup, "Perceptual hash duplicates in sampled set"))

    severity_rank = {"error": 3, "warning": 2, "info": 1}
    max_severity = max((severity_rank.get(x.severity, 0) for x in issues), default=0)
    strict_mode = str(config.get("strictness", "benchmark"))
    fail_on_warnings = bool(config.get("fail_on_warnings", False))
    valid = max_severity < 3 and (not fail_on_warnings or max_severity < 2)
    if strict_mode == "benchmark":
        valid = max_severity < 3

    report = {
        "valid": bool(valid),
        "strictness": strict_mode,
        "issue_count": len(issues),
        "issues": [x.__dict__ for x in issues],
        "dedup_stats": {
            "exact_duplicates_sampled": exact_dup,
            "near_duplicates_sampled": near_dup,
            "sampled_rows": int(min(len(df), int(config.get("dedup", {}).get("max_images", 5000)))),
        },
    }
    return issues, report
