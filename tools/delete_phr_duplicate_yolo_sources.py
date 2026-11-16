"""Verify and delete YOLO source folders already integrated into phr-labeled-images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
LABEL_EXTS = {".txt"}
DATA_EXTS = IMAGE_EXTS | LABEL_EXTS
DEFAULT_CANDIDATES = [
    "bangladeshi-cow-breeds-yd84s35d73-yolo-multispecies",
    "cattle-breeds-anandkumarsahu09-yolo",
    "horse-breeds-olgabelitskaya-yolo",
    "wikimedia-duck-breeds-yolo-multispecies",
    "wikimedia-goose-breeds-yolo-multispecies",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_phr_metadata(phr_root: Path) -> dict[str, list[dict]]:
    by_source: dict[str, list[dict]] = {}
    for meta_path in sorted((phr_root / "metadata").rglob("*.json")):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source_id = str(payload.get("source_dataset_id", "")).strip()
        if not source_id:
            continue
        payload["_metadata_path"] = str(meta_path)
        by_source.setdefault(source_id, []).append(payload)
    return by_source


def _source_pairs(source_dir: Path) -> tuple[set[str], int, int]:
    image_stems = {p.stem for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    label_stems = {p.stem for p in source_dir.rglob("*.txt") if p.is_file()}
    return image_stems & label_stems, len(image_stems), len(label_stems)


def _source_stem_variants(stem: str) -> set[str]:
    variants = {stem}
    if "__" in stem:
        variants.add(stem.split("__", 1)[1])
    return variants


def _archive_provenance(source_dir: Path, archive_dir: Path) -> int:
    copied = 0
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source_dir)
        if rel.parts and rel.parts[0].lower() == "metadata":
            # Per-image metadata is already represented in phr-labeled-images.
            # Copying it into the archive creates very long Windows paths and is
            # not needed for label-name recovery.
            continue
        if path.suffix.lower() in DATA_EXTS:
            continue
        dst = archive_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(path, dst)
            copied += 1
        except OSError:
            # Do not block deletion on non-trainable provenance copy failures.
            # The deletion report records how many files were archived.
            continue
    return copied


def _evaluate_candidate(source_root: Path, phr_root: Path, source_id: str, phr_by_source: dict[str, list[dict]]) -> dict:
    source_dir = source_root / source_id
    row = {
        "source_id": source_id,
        "exists": source_dir.exists(),
        "source_trainable_pairs": 0,
        "source_images": 0,
        "source_labels": 0,
        "phr_metadata": 0,
        "missing_stem_count": 0,
        "missing_path_count": 0,
        "decision": "retain",
        "reason": "",
        "archived_provenance_files": 0,
    }
    if not source_dir.exists():
        row["decision"] = "missing"
        row["reason"] = "source folder does not exist"
        return row

    source_stems, image_count, label_count = _source_pairs(source_dir)
    phr_rows = phr_by_source.get(source_id, [])
    phr_stems = {Path(str(payload.get("image_name", ""))).stem for payload in phr_rows if payload.get("image_name")}
    missing_stems = sorted(stem for stem in source_stems if _source_stem_variants(stem).isdisjoint(phr_stems))
    missing_paths = []
    for payload in phr_rows:
        image_path = Path(str(payload.get("image_path", "")))
        label_path = Path(str(payload.get("label_path", "")))
        if not image_path.exists() or not label_path.exists():
            missing_paths.append(str(payload.get("_metadata_path", "")))

    row.update(
        {
            "source_trainable_pairs": len(source_stems),
            "source_images": image_count,
            "source_labels": label_count,
            "phr_metadata": len(phr_rows),
            "missing_stem_count": len(missing_stems),
            "missing_path_count": len(missing_paths),
        }
    )
    if not source_stems:
        row["decision"] = "delete"
        row["reason"] = "no trainable image/label pairs"
    elif len(phr_rows) < len(source_stems):
        row["reason"] = "PHR metadata count is smaller than source trainable pairs"
    elif missing_stems:
        row["reason"] = "source stems missing from PHR metadata"
    elif missing_paths:
        row["reason"] = "PHR metadata has missing image/label paths"
    else:
        row["decision"] = "delete"
        row["reason"] = "all source trainable pairs covered by PHR metadata"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(_repo_root() / "data" / "raw" / "yolo_sources"))
    parser.add_argument("--phr-root", default=str(_repo_root() / "data" / "raw" / "yolo_sources" / "phr-labeled-images"))
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--candidate", action="append", default=[])
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    phr_root = Path(args.phr_root).resolve()
    report_csv = Path(args.report_csv)
    summary_json = Path(args.summary_json)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    candidates = args.candidate or DEFAULT_CANDIDATES

    phr_by_source = _load_phr_metadata(phr_root)
    rows = [_evaluate_candidate(source_root, phr_root, source_id, phr_by_source) for source_id in candidates]

    archive_root = phr_root / "integration_archives" / "deleted_duplicate_sources"
    deleted = []
    retained = []
    if args.delete:
        for row in rows:
            if row["decision"] != "delete":
                retained.append(row["source_id"])
                continue
            source_dir = source_root / row["source_id"]
            archive_dir = archive_root / row["source_id"]
            copied = _archive_provenance(source_dir, archive_dir)
            row["archived_provenance_files"] = copied
            shutil.rmtree(source_dir)
            deleted.append(row["source_id"])
    else:
        retained = [row["source_id"] for row in rows if row["decision"] != "delete"]

    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "source_id",
            "exists",
            "source_trainable_pairs",
            "source_images",
            "source_labels",
            "phr_metadata",
            "missing_stem_count",
            "missing_path_count",
            "decision",
            "reason",
            "archived_provenance_files",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_root": str(source_root),
        "phr_root": str(phr_root),
        "delete_requested": bool(args.delete),
        "deleted": deleted,
        "retained": retained,
        "rows": rows,
        "report_csv": str(report_csv.resolve()),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    blocking = [row for row in rows if row["exists"] and row["decision"] != "delete"]
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 1 if blocking and not args.delete else 0


if __name__ == "__main__":
    raise SystemExit(main())
