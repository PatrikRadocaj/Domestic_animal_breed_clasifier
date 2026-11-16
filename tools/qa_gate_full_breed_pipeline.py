"""Pipeline-specific QA gates for the full breed workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha1
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import DecompressionBombError

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from animal_breed.data.manifest_schema import read_manifest, validate_manifest
from animal_breed.data.quality.qa import run_quality_checks
from animal_breed.runtime.amd_gpu_probe import build_probe_report

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
PLACEHOLDER_TOKENS = {"unknown", "unlabeled", "placeholder", "other", "misc", "none", "nan"}


@dataclass(frozen=True)
class GateIssue:
    severity: str
    code: str
    message: str


def _safe_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _add_issue(issues: list[GateIssue], severity: str, code: str, message: str) -> None:
    issues.append(GateIssue(severity=severity, code=code, message=message))


def _fingerprint_rows(rows: list[str]) -> str:
    digest = sha1()
    for row in rows:
        digest.update(row.encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


def _is_readable_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
        return True, ""
    except (UnidentifiedImageError, OSError, DecompressionBombError, SyntaxError, ValueError) as exc:
        return False, str(exc)


def _scan_yolo_source_root(root: Path) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    fingerprint_rows: list[str] = []
    total_images = 0
    total_labels = 0
    total_metadata = 0
    for child in sorted(x for x in root.iterdir() if x.is_dir()):
        images = sum(1 for p in child.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS)
        labels = sum(1 for p in child.rglob("*.txt") if p.is_file())
        metadata = sum(1 for p in child.rglob("*.json") if p.is_file())
        total_images += images
        total_labels += labels
        total_metadata += metadata
        datasets.append(
            {
                "dataset_id": child.name,
                "images": images,
                "labels": labels,
                "metadata": metadata,
            }
        )
        fingerprint_rows.append(f"{child.name}|{images}|{labels}|{metadata}")
    return {
        "dataset_count": len(datasets),
        "total_images": total_images,
        "total_labels": total_labels,
        "total_metadata": total_metadata,
        "datasets": datasets,
        "fingerprint": _fingerprint_rows(fingerprint_rows),
    }


def _check_split_balance(df: pd.DataFrame, issues: list[GateIssue], split_column: str = "split") -> dict[str, Any]:
    if split_column not in df.columns:
        _add_issue(issues, "error", "missing_split_column", f"Manifest is missing split column '{split_column}'.")
        return {}
    split_counts = df[split_column].value_counts(dropna=False).to_dict()
    for split_name in ("train", "val"):
        if int(split_counts.get(split_name, 0)) <= 0:
            _add_issue(issues, "error", "empty_split", f"Split '{split_name}' has no rows.")
    return {"split_counts": split_counts}


def _check_readability_sample(df: pd.DataFrame, issues: list[GateIssue], max_rows: int = 1000) -> dict[str, Any]:
    checked = 0
    unreadable: list[dict[str, str]] = []
    for raw_path in df["path"].head(max_rows):
        path = Path(str(raw_path))
        checked += 1
        ok, reason = _is_readable_image(path)
        if not ok:
            unreadable.append({"path": str(path), "reason": reason})
    unreadable_rate = float(len(unreadable) / max(1, checked))
    if unreadable_rate > 0.005:
        _add_issue(
            issues,
            "error",
            "unreadable_image_rate",
            f"Unreadable image sample rate {unreadable_rate:.2%} exceeds 0.5% over {checked} checked rows.",
        )
    elif unreadable:
        _add_issue(
            issues,
            "warning",
            "unreadable_image_sample",
            f"Found {len(unreadable)} unreadable images in readability sample of {checked}.",
        )
    return {"checked_rows": checked, "unreadable_count": len(unreadable), "examples": unreadable[:10]}


def gate_pre_stage1(args: argparse.Namespace) -> dict[str, Any]:
    issues: list[GateIssue] = []
    yolo_root = Path(args.yolo_root)
    non_yolo_root = Path(args.non_yolo_root)
    phr_yaml = Path(args.phr_yaml)

    if not yolo_root.exists():
        _add_issue(issues, "error", "missing_yolo_root", f"Missing yolo_sources root: {yolo_root}")
    if not non_yolo_root.exists():
        _add_issue(issues, "error", "missing_non_yolo_root", f"Missing non_yolo_sources root: {non_yolo_root}")
    if not phr_yaml.exists():
        _add_issue(issues, "error", "missing_phr_yaml", f"Missing phr data.yaml: {phr_yaml}")

    gpu_probe = build_probe_report()
    if args.expect_gpu and not bool(gpu_probe.get("cuda_available")):
        _add_issue(issues, "error", "gpu_unavailable", "AMD GPU runtime probe reports torch.cuda.is_available() == False.")
    else:
        device_name = str(gpu_probe.get("device_name", ""))
        if args.expect_gpu and "7800 XT" not in device_name.upper():
            _add_issue(
                issues,
                "warning",
                "unexpected_gpu_name",
                f"GPU runtime is available, but detected device is '{device_name}', not the expected RX 7800 XT string.",
            )

    yolo_scan = _scan_yolo_source_root(yolo_root) if yolo_root.exists() else {}
    if yolo_scan:
        if int(yolo_scan["dataset_count"]) <= 0:
            _add_issue(issues, "error", "empty_yolo_root", "No YOLO source datasets were found.")
        if int(yolo_scan["total_images"]) <= 0:
            _add_issue(issues, "error", "no_yolo_images", "No YOLO images were found under yolo_sources.")
        if int(yolo_scan["total_labels"]) <= 0:
            _add_issue(issues, "error", "no_yolo_labels", "No YOLO label files were found under yolo_sources.")

    phr_info: dict[str, Any] = {}
    if phr_yaml.exists():
        payload = yaml.safe_load(phr_yaml.read_text(encoding="utf-8")) or {}
        names = [str(x).strip() for x in payload.get("names", [])]
        phr_info = {
            "nc": int(payload.get("nc", len(names))),
            "names": names,
            "path": str(payload.get("path", "")),
        }
        if len(names) != len(set(names)):
            _add_issue(issues, "error", "duplicate_phr_names", "phr-labeled-images data.yaml has duplicate class names.")
        if "dromedary" not in {x.lower() for x in names}:
            _add_issue(issues, "warning", "missing_dromedary_class", "phr-labeled-images data.yaml does not list dromedary.")

    severity_rank = {"error": 3, "warning": 2, "info": 1}
    max_severity = max((severity_rank.get(x.severity, 0) for x in issues), default=0)
    valid = max_severity < 3
    return {
        "phase": "pre_stage1",
        "valid": valid,
        "issues": [asdict(x) for x in issues],
        "gpu_probe": gpu_probe,
        "yolo_sources": yolo_scan,
        "phr_dataset": phr_info,
    }


def gate_post_stage1_for_mining(args: argparse.Namespace) -> dict[str, Any]:
    issues: list[GateIssue] = []
    hybrid_spec = Path(args.hybrid_spec)
    best_weights = Path(args.stage1_best)
    results_csv = Path(args.stage1_results) if args.stage1_results else None

    if not hybrid_spec.exists():
        _add_issue(issues, "error", "missing_hybrid_spec", f"Missing Stage-1 hybrid spec: {hybrid_spec}")
        spec = {}
    else:
        spec = json.loads(hybrid_spec.read_text(encoding="utf-8"))
    if not best_weights.exists():
        _add_issue(issues, "error", "missing_stage1_best", f"Missing Stage-1 best checkpoint: {best_weights}")

    counts = dict(spec.get("counts") or {})
    if counts:
        if int(counts.get("train_plans", 0)) <= 0:
            _add_issue(issues, "error", "empty_stage1_train", "Stage-1 hybrid spec has zero train plans.")
        if int(counts.get("val_plans", 0)) <= 0:
            _add_issue(issues, "error", "empty_stage1_val", "Stage-1 hybrid spec has zero val plans.")
        if int(counts.get("test_plans", 0)) <= 0:
            _add_issue(issues, "warning", "empty_stage1_test", "Stage-1 hybrid spec has zero test plans.")
        excluded = list(counts.get("excluded_species_without_pure_train_images") or [])
        if excluded:
            _add_issue(
                issues,
                "warning",
                "excluded_species_without_pure_train_images",
                f"Stage-1 excluded species without pure train images: {excluded}",
            )

    results_summary: dict[str, Any] = {}
    if results_csv and results_csv.exists():
        results_df = pd.read_csv(results_csv)
        if results_df.empty:
            _add_issue(issues, "error", "empty_stage1_results", "Stage-1 results.csv is empty.")
        else:
            latest = results_df.iloc[-1].to_dict()
            best = results_df.sort_values("metrics/mAP50-95(B)", ascending=False).iloc[0].to_dict()
            results_summary = {"latest": latest, "best": best, "epochs_recorded": int(len(results_df))}
            best_map = float(best.get("metrics/mAP50-95(B)", 0.0))
            if best_map < float(args.min_stage1_map5095):
                _add_issue(
                    issues,
                    "warning",
                    "low_stage1_map5095",
                    f"Best Stage-1 mAP50-95 {best_map:.4f} is below advisory threshold {args.min_stage1_map5095:.4f}.",
                )

    severity_rank = {"error": 3, "warning": 2, "info": 1}
    valid = max((severity_rank.get(x.severity, 0) for x in issues), default=0) < 3
    return {
        "phase": "post_stage1_for_mining",
        "valid": valid,
        "issues": [asdict(x) for x in issues],
        "hybrid_spec": {
            "path": str(hybrid_spec),
            "class_names": list(spec.get("class_names") or []),
            "counts": counts,
        },
        "stage1_results": results_summary,
    }


def gate_post_breed_finalize_pre_stage2(args: argparse.Namespace) -> dict[str, Any]:
    issues: list[GateIssue] = []
    manifest_csv = Path(args.manifest_csv)
    taxonomy_yaml = Path(args.taxonomy_yaml)
    build_summary_json = Path(args.build_summary_json) if args.build_summary_json else None
    review_json = Path(args.review_json) if args.review_json else None
    review_csv = Path(args.review_csv) if args.review_csv else None

    if not manifest_csv.exists():
        _add_issue(issues, "error", "missing_stage2_manifest", f"Missing Stage-2 manifest: {manifest_csv}")
        df = pd.DataFrame()
    else:
        df = read_manifest(manifest_csv)
    if not taxonomy_yaml.exists():
        _add_issue(issues, "error", "missing_stage2_taxonomy", f"Missing Stage-2 taxonomy: {taxonomy_yaml}")

    manifest_report: dict[str, Any] = {}
    quality_report: dict[str, Any] = {}
    readability_report: dict[str, Any] = {}
    species_summary: list[dict[str, Any]] = []
    ratio_alerts: list[dict[str, Any]] = []

    if not df.empty and taxonomy_yaml.exists():
        validation = validate_manifest(df, taxonomy_yaml, require_existing_files=True, require_taxonomy_match=True)
        manifest_report = {
            "valid": bool(validation.valid),
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
            "rows": int(len(df)),
        }
        if not validation.valid:
            for msg in validation.errors:
                _add_issue(issues, "error", "manifest_validation_failed", msg)

        _, quality_report = run_quality_checks(
            df,
            taxonomy_yaml,
            config={
                "strictness": "benchmark",
                "fail_on_warnings": False,
                "split_column": "split",
                "source_column": "source",
                "max_bbox_aspect": 12.0,
                "dedup": {"enabled": True, "near_enabled": True, "max_images": 3000},
            },
        )
        if not bool(quality_report.get("valid", False)):
            for issue in quality_report.get("issues", []):
                if str(issue.get("severity", "")).lower() == "error":
                    _add_issue(
                        issues,
                        "error",
                        f"quality_{issue.get('code', 'unknown')}",
                        str(issue.get("details", issue.get("code", "quality error"))),
                    )

        readability_report = _check_readability_sample(df, issues, max_rows=int(args.readability_sample))
        manifest_report.update(_check_split_balance(df, issues))

        if "breed_id" in df.columns:
            placeholder_mask = df["breed_id"].fillna("").astype(str).str.strip().str.lower().isin(PLACEHOLDER_TOKENS)
            if bool(placeholder_mask.any()):
                _add_issue(
                    issues,
                    "error",
                    "placeholder_breed_ids_in_stage2",
                    f"Stage-2 manifest still contains {int(placeholder_mask.sum())} placeholder breed IDs.",
                )

        if {"species_id", "breed_id", "split"}.issubset(df.columns):
            for species_id, species_df in df.groupby("species_id"):
                breed_count = int(species_df["breed_id"].nunique())
                split_counts = species_df["split"].value_counts().to_dict()
                sample_count = int(len(species_df))
                species_row = {
                    "species_id": str(species_id),
                    "rows": sample_count,
                    "breed_classes": breed_count,
                    "split_counts": split_counts,
                }
                if "bbox_source" in species_df.columns:
                    bbox_source = species_df["bbox_source"].fillna("").astype(str)
                    pseudo_mask = bbox_source.str.contains("detector", case=False, na=False)
                    pseudo_count = int(pseudo_mask.sum())
                    human_count = int((~pseudo_mask).sum())
                    species_row["pseudo_rows"] = pseudo_count
                    species_row["human_rows"] = human_count
                    if human_count > 0 and pseudo_count > (human_count * float(args.max_pseudo_ratio)) and breed_count >= 2:
                        ratio_alerts.append(
                            {
                                "species_id": str(species_id),
                                "pseudo_rows": pseudo_count,
                                "human_rows": human_count,
                                "ratio": float(pseudo_count / max(1, human_count)),
                            }
                        )
                        _add_issue(
                            issues,
                            "warning",
                            "pseudo_ratio_high",
                            f"species={species_id} pseudo/human ratio {pseudo_count}/{human_count} exceeds advisory limit {args.max_pseudo_ratio:.2f}.",
                        )
                if breed_count < 2:
                    _add_issue(
                        issues,
                        "warning",
                        "species_not_stage2_eligible",
                        f"species={species_id} has only {breed_count} breed classes in Stage-2 manifest.",
                    )
                species_summary.append(species_row)

    build_summary = {}
    if build_summary_json and build_summary_json.exists():
        build_summary = json.loads(build_summary_json.read_text(encoding="utf-8"))
        detector_stats = dict(build_summary.get("detector_stats") or {})
        processed = int(detector_stats.get("processed", 0))
        matched = int(detector_stats.get("matched", 0))
        if processed > 0:
            mining_precision_proxy = float(matched / max(1, processed))
            if mining_precision_proxy < float(args.min_mining_match_ratio):
                _add_issue(
                    issues,
                    "warning",
                    "low_mining_match_ratio",
                    f"Mining matched/processed ratio {mining_precision_proxy:.4f} is below advisory threshold {args.min_mining_match_ratio:.4f}.",
                )

    review_summary = {}
    if review_json and review_json.exists():
        review_summary = json.loads(review_json.read_text(encoding="utf-8"))
        manual_review_count = int((review_summary.get("review_status_counts") or {}).get("manual_review", 0))
        if manual_review_count > 0:
            _add_issue(
                issues,
                "warning",
                "manual_review_rows_exist",
                f"{manual_review_count} rows were flagged manual_review during strict finalize.",
            )
    elif review_csv and review_csv.exists():
        review_df = pd.read_csv(review_csv)
        if "review_status" in review_df.columns:
            manual_review_count = int(review_df["review_status"].eq("manual_review").sum())
            if manual_review_count > 0:
                _add_issue(
                    issues,
                    "warning",
                    "manual_review_rows_exist",
                    f"{manual_review_count} rows were flagged manual_review during strict finalize.",
                )

    severity_rank = {"error": 3, "warning": 2, "info": 1}
    valid = max((severity_rank.get(x.severity, 0) for x in issues), default=0) < 3
    return {
        "phase": "post_breed_finalize_pre_stage2",
        "valid": valid,
        "issues": [asdict(x) for x in issues],
        "manifest_report": manifest_report,
        "quality_report": quality_report,
        "readability_report": readability_report,
        "build_summary": build_summary,
        "review_summary": review_summary,
        "species_summary": species_summary,
        "ratio_alerts": ratio_alerts,
    }


def gate_post_direct_prep_pre_direct_yolo(args: argparse.Namespace) -> dict[str, Any]:
    issues: list[GateIssue] = []
    yaml_path = Path(args.data_yaml)
    manifest_csv = Path(args.manifest_csv)

    if not yaml_path.exists():
        _add_issue(issues, "error", "missing_direct_yaml", f"Missing direct YOLO yaml: {yaml_path}")
        yaml_payload = {}
    else:
        yaml_payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not manifest_csv.exists():
        _add_issue(issues, "error", "missing_direct_manifest", f"Missing direct manifest: {manifest_csv}")
        df = pd.DataFrame()
    else:
        df = read_manifest(manifest_csv)

    summary: dict[str, Any] = {}
    if yaml_payload:
        base_path = Path(str(yaml_payload.get("path", "")))
        train_path = base_path / str(yaml_payload.get("train", ""))
        val_path = base_path / str(yaml_payload.get("val", ""))
        names = yaml_payload.get("names", [])
        summary = {
            "path": str(base_path),
            "train": str(train_path),
            "val": str(val_path),
            "nc": int(yaml_payload.get("nc", len(names))),
            "names_count": int(len(names)),
        }
        if not train_path.exists():
            _add_issue(issues, "error", "missing_direct_train_dir", f"Direct YOLO train dir missing: {train_path}")
        if not val_path.exists():
            _add_issue(issues, "error", "missing_direct_val_dir", f"Direct YOLO val dir missing: {val_path}")
        if len(names) != len(set(str(x) for x in names)):
            _add_issue(issues, "error", "duplicate_direct_names", "Direct YOLO yaml contains duplicate class names.")

    manifest_split_report: dict[str, Any] = {}
    if not df.empty:
        manifest_split_report = _check_split_balance(df, issues)
        direct_col = "direct_label_id"
        if direct_col in df.columns:
            blank_direct = int(df[direct_col].fillna("").astype(str).str.strip().eq("").sum())
            if blank_direct > 0:
                _add_issue(issues, "error", "blank_direct_labels", f"{blank_direct} rows have blank direct_label_id.")
        readability_report = _check_readability_sample(df, issues, max_rows=int(args.readability_sample))
    else:
        readability_report = {}

    severity_rank = {"error": 3, "warning": 2, "info": 1}
    valid = max((severity_rank.get(x.severity, 0) for x in issues), default=0) < 3
    return {
        "phase": "post_direct_prep_pre_direct_yolo",
        "valid": valid,
        "issues": [asdict(x) for x in issues],
        "yaml_summary": summary,
        "manifest_split_report": manifest_split_report,
        "readability_report": readability_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full breed pipeline QA gates.")
    sub = parser.add_subparsers(dest="phase", required=True)

    pre = sub.add_parser("pre_stage1")
    pre.add_argument("--yolo-root", default=str(REPO_ROOT / "data" / "raw" / "yolo_sources"))
    pre.add_argument("--non-yolo-root", default=str(REPO_ROOT / "data" / "raw" / "non_yolo_sources"))
    pre.add_argument("--phr-yaml", default=str(REPO_ROOT / "data" / "raw" / "yolo_sources" / "phr-labeled-images" / "data.yaml"))
    pre.add_argument("--expect-gpu", action="store_true")
    pre.add_argument("--report-json", required=True)

    post1 = sub.add_parser("post_stage1_for_mining")
    post1.add_argument("--hybrid-spec", required=True)
    post1.add_argument("--stage1-best", required=True)
    post1.add_argument("--stage1-results", default="")
    post1.add_argument("--min-stage1-map5095", type=float, default=0.15)
    post1.add_argument("--report-json", required=True)

    post2 = sub.add_parser("post_breed_finalize_pre_stage2")
    post2.add_argument("--manifest-csv", required=True)
    post2.add_argument("--taxonomy-yaml", required=True)
    post2.add_argument("--build-summary-json", default="")
    post2.add_argument("--review-json", default="")
    post2.add_argument("--review-csv", default="")
    post2.add_argument("--readability-sample", type=int, default=1000)
    post2.add_argument("--max-pseudo-ratio", type=float, default=2.0)
    post2.add_argument("--min-mining-match-ratio", type=float, default=0.40)
    post2.add_argument("--report-json", required=True)

    post3 = sub.add_parser("post_direct_prep_pre_direct_yolo")
    post3.add_argument("--data-yaml", required=True)
    post3.add_argument("--manifest-csv", required=True)
    post3.add_argument("--readability-sample", type=int, default=1000)
    post3.add_argument("--report-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "pre_stage1":
        report = gate_pre_stage1(args)
    elif args.phase == "post_stage1_for_mining":
        report = gate_post_stage1_for_mining(args)
    elif args.phase == "post_breed_finalize_pre_stage2":
        report = gate_post_breed_finalize_pre_stage2(args)
    elif args.phase == "post_direct_prep_pre_direct_yolo":
        report = gate_post_direct_prep_pre_direct_yolo(args)
    else:
        raise ValueError(f"Unsupported phase: {args.phase}")

    out_path = Path(args.report_json)
    _safe_json(out_path, report)
    print(json.dumps(report, indent=2))
    if not bool(report.get("valid", False)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
