"""Normalize raw YOLO source images in place for stable multi-worker training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
JPEG_EXTS = {".jpg", ".jpeg"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_replace_strings(payload: Any, replacements: dict[str, str]) -> Any:
    if isinstance(payload, dict):
        return {k: _json_replace_strings(v, replacements) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_json_replace_strings(v, replacements) for v in payload]
    if isinstance(payload, str):
        out = payload
        for old, new in replacements.items():
            out = out.replace(old, new)
        return out
    return payload


def _update_metadata_paths(root: Path, old_path: Path, new_path: Path) -> int:
    phr_root = root / "phr-labeled-images"
    images_root = phr_root / "images"
    metadata_root = phr_root / "metadata"
    try:
        rel = old_path.resolve().relative_to(images_root.resolve())
    except ValueError:
        return 0
    parts = rel.parts
    if len(parts) < 2:
        return 0
    split = parts[0]
    meta_path = metadata_root / split / f"{old_path.stem}.json"
    if not meta_path.exists():
        return 0
    old_abs = str(old_path.resolve())
    new_abs = str(new_path.resolve())
    replacements = {
        old_abs: new_abs,
        old_abs.replace("\\", "/"): new_abs.replace("\\", "/"),
        old_path.name: new_path.name,
    }
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    updated = _json_replace_strings(payload, replacements)
    if updated == payload:
        return 0
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(updated, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(meta_path)
    return 1


def _read_rgb_image(path: Path) -> tuple[Image.Image | None, str | None]:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB"), None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return None, str(exc)


def _normalize_one(path: Path, root: Path, max_side: int, quality: int, dry_run: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "old_path": str(path),
        "new_path": "",
        "status": "",
        "old_width": "",
        "old_height": "",
        "new_width": "",
        "new_height": "",
        "old_mode": "",
        "metadata_updates": 0,
        "error": "",
    }
    image, error = _read_rgb_image(path)
    if image is None:
        row["status"] = "skipped_unreadable"
        row["error"] = error or "unreadable"
        return row

    old_width, old_height = image.size
    row["old_width"] = old_width
    row["old_height"] = old_height
    row["old_mode"] = image.mode
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    new_width, new_height = image.size
    row["new_width"] = new_width
    row["new_height"] = new_height

    target = path if path.suffix.lower() in JPEG_EXTS else path.with_suffix(".jpg")
    row["new_path"] = str(target)
    if target.exists() and target.resolve() != path.resolve():
        row["status"] = "skipped_target_exists"
        row["error"] = f"target exists: {target}"
        return row

    needs_write = (
        target != path
        or old_width != new_width
        or old_height != new_height
        or path.suffix.lower() not in JPEG_EXTS
    )
    if not needs_write:
        row["status"] = "already_ok"
        return row
    if dry_run:
        row["status"] = "would_normalize"
        return row

    tmp = target.with_name(target.name + ".tmp")
    image.save(tmp, format="JPEG", quality=quality)
    verify, verify_error = _read_rgb_image(tmp)
    if verify is None:
        tmp.unlink(missing_ok=True)
        row["status"] = "failed_verify"
        row["error"] = verify_error or "replacement failed verification"
        return row
    tmp.replace(target)
    if target.resolve() != path.resolve():
        path.unlink()
        row["metadata_updates"] = _update_metadata_paths(root, path, target)
    row["status"] = "normalized"
    return row


def _verify_tree(root: Path, max_side: int) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        image, error = _read_rgb_image(path)
        if image is None:
            issues.append({"path": str(path), "issue": "unreadable", "detail": error or ""})
            continue
        if path.suffix.lower() not in JPEG_EXTS:
            issues.append({"path": str(path), "issue": "non_jpeg_extension", "detail": path.suffix})
        if image.mode != "RGB":
            issues.append({"path": str(path), "issue": "not_rgb", "detail": image.mode})
        if max(image.size) > max_side:
            issues.append({"path": str(path), "issue": "oversized", "detail": f"{image.size[0]}x{image.size[1]}"})
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(_repo_root() / "data" / "raw" / "yolo_sources"))
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report_csv = Path(args.report_csv)
    summary_json = Path(args.summary_json)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        issues = _verify_tree(root, args.max_side)
        summary = {"root": str(root), "verify_only": True, "issues": len(issues), "issue_samples": issues[:50]}
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        with report_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "issue", "detail"])
            writer.writeheader()
            writer.writerows(issues)
        return 1 if issues else 0

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    image_paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    for idx, path in enumerate(image_paths, start=1):
        row = _normalize_one(path, root, int(args.max_side), int(args.quality), bool(args.dry_run))
        rows.append(row)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if idx % 1000 == 0:
            print(f"[normalize-yolo-images] processed={idx}/{len(image_paths)} counts={counts}", flush=True)

    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "old_path",
            "new_path",
            "status",
            "old_width",
            "old_height",
            "new_width",
            "new_height",
            "old_mode",
            "metadata_updates",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    verify_issues = [] if args.dry_run else _verify_tree(root, int(args.max_side))
    summary = {
        "root": str(root),
        "dry_run": bool(args.dry_run),
        "total_images_seen": len(image_paths),
        "counts": counts,
        "verify_issue_count": len(verify_issues),
        "verify_issue_samples": verify_issues[:50],
        "report_csv": str(report_csv.resolve()),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    if verify_issues:
        return 2
    if counts.get("skipped_unreadable", 0) or counts.get("skipped_target_exists", 0) or counts.get("failed_verify", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
