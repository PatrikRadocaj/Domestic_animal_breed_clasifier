from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "results" / "reports" / "source_label_hygiene_fixes_20260706.csv"


def record(rows: list[dict[str, Any]], action: str, path: Path, detail: str, status: str) -> None:
    rows.append(
        {
            "action": action,
            "path": str(path),
            "detail": detail,
            "status": status,
        }
    )


def safe_rename_dir(rows: list[dict[str, Any]], src: Path, new_name: str, reason: str) -> None:
    if not src.exists():
        record(rows, "rename_dir", src, f"{new_name} | {reason}", "missing_source")
        return
    dst = src.with_name(new_name)
    if dst.exists():
        record(rows, "rename_dir", src, f"{new_name} | {reason}", "target_exists_skipped")
        return
    src.rename(dst)
    record(rows, "rename_dir", src, f"{dst} | {reason}", "renamed")


def fix_yaml_names(rows: list[dict[str, Any]], path: Path, names: list[str], reason: str) -> None:
    if not path.exists():
        record(rows, "fix_yaml_names", path, reason, "missing")
        return
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    old = payload.get("names")
    if old == names:
        record(rows, "fix_yaml_names", path, reason, "already_ok")
        return
    backup = path.with_suffix(path.suffix + ".label_hygiene_bak")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    payload["names"] = names
    payload["nc"] = len(names)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    record(rows, "fix_yaml_names", path, f"{old!r} -> {names!r} | {reason}", "updated")


def fix_bactrian_metadata(rows: list[dict[str, Any]]) -> None:
    root = REPO_ROOT / "data" / "raw" / "yolo_sources" / "wikimedia-bactrian-camel-yolo-multispecies" / "metadata"
    if not root.exists():
        record(rows, "fix_metadata", root, "bactrian metadata root", "missing")
        return
    changed_files = 0
    changed_objects = 0
    for path in root.rglob("*.json"):
        try:
            original_text = path.read_text(encoding="utf-8")
            payload = json.loads(original_text)
        except Exception as exc:
            record(rows, "fix_metadata", path, f"json parse failed: {exc}", "skipped")
            continue
        changed = False
        for obj in payload.get("objects", []) or []:
            source_label = str(obj.get("source_label_name", "")).strip().lower()
            subspecies = str(obj.get("subspecies", "")).strip().lower()
            class_name = str(obj.get("class_name", "")).strip().lower()
            species_name = str(obj.get("species_name", "")).strip().lower()
            if source_label == "lama" or subspecies == "lama" or class_name == "lama" or species_name == "lama":
                obj["class_name"] = "llama"
                obj["species_name"] = "llama"
                obj["source_label_name"] = "llama"
                obj["subspecies"] = "unknown"
                changed = True
                changed_objects += 1
            if source_label in {"donkey-zebra", "zebra-donkey"} or subspecies in {"donkey_zebra", "zebra_donkey"}:
                obj["class_name"] = "donkey"
                obj["species_name"] = "donkey"
                obj["source_label_name"] = "donkey"
                obj["subspecies"] = "unknown"
                changed = True
                changed_objects += 1
        if changed:
            backup = path.with_suffix(path.suffix + ".label_hygiene_bak")
            if not backup.exists():
                backup.write_text(original_text, encoding="utf-8")
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            changed_files += 1
    record(rows, "fix_metadata", root, f"files={changed_files} objects={changed_objects}", "updated")


def main() -> int:
    rows: list[dict[str, Any]] = []

    # High-confidence non-YOLO source folder fixes.
    base = REPO_ROOT / "data" / "raw" / "non_yolo_sources"
    safe_rename_dir(
        rows,
        base / "cats-dogs-ng6tv57j5c" / "FeliBreed – A Multi-Breed Cat Image Dataset" / "FeliBreed – A Multi-Breed Cat Image Dataset" / "Bengal Cat",
        "Bengal",
        "remove redundant species word from cat breed label",
    )
    safe_rename_dir(
        rows,
        base / "cats-dogs-ng6tv57j5c" / "FeliBreed – A Multi-Breed Cat Image Dataset" / "FeliBreed – A Multi-Breed Cat Image Dataset" / "Tabby Cat",
        "Tabby",
        "remove redundant species word from cat morph label",
    )
    safe_rename_dir(
        rows,
        base / "cats-dogs-ng6tv57j5c" / "FeliBreed – A Multi-Breed Cat Image Dataset" / "FeliBreed – A Multi-Breed Cat Image Dataset" / "Local",
        "_excluded_local_unknown",
        "local is not a breed or visually consistent morph",
    )
    safe_rename_dir(
        rows,
        base / "cats-dogs-ng6tv57j5c" / "FeliBreed – A Multi-Breed Cat Image Dataset" / "FeliBreed – A Multi-Breed Cat Image Dataset" / "Mixed Breed",
        "_excluded_mixed_breed",
        "mixed breed placeholder is not a clean Stage-2 class",
    )

    safe_rename_dir(rows, base / "buffalo-breeds-vdgnxsm692" / "Data" / "Data" / "Khundi", "Kundi", "canonical spelling for Kundi buffalo")
    safe_rename_dir(rows, base / "buffalo-breeds-vdgnxsm692" / "Data" / "Data" / "Neli Ravi", "Nili_Ravi", "canonical spelling for Nili-Ravi buffalo")
    safe_rename_dir(rows, base / "buffalo-breeds-vdgnxsm692" / "Data" / "Data" / "Mix", "_excluded_mixed_breed", "mixed placeholder")
    safe_rename_dir(
        rows,
        base / "buffalo-breeds-vdgnxsm692" / "Un Labelled Buffalos Dataset for Clustering Problems" / "Un Labelled Buffalos Dataset for Clustering Problems",
        "_excluded_unlabeled_buffalo",
        "unlabeled clustering data has no breed label",
    )

    safe_rename_dir(
        rows,
        base / "cattle-side-back-h2s22wr5py" / "Cattle side view and back view dataset" / "Cattle side and back view images" / "back view",
        "_excluded_back_view",
        "camera/viewpoint label, not breed",
    )
    safe_rename_dir(
        rows,
        base / "cattle-side-back-h2s22wr5py" / "Cattle side view and back view dataset" / "Cattle side and back view images" / "side view",
        "_excluded_side_view",
        "camera/viewpoint label, not breed",
    )

    for split in ("train", "valid", "test"):
        safe_rename_dir(
            rows,
            base / "chicken-detection-tracking-nirmalsankalana" / split / "empty",
            "_excluded_empty",
            "negative/empty folder, not chicken breed",
        )

    cow_breed_root = base / "cow-breed-classification-zaidworks0508" / "Cow Breed Dataset"
    safe_rename_dir(rows, cow_breed_root / "fresian", "friesian_cattle", "fix spelling and avoid horse Friesian collision")
    safe_rename_dir(rows, cow_breed_root / "fresian cross", "friesian_cattle_cross", "fix spelling and preserve cross-breed marker")
    safe_rename_dir(rows, cow_breed_root / "brahman cross", "brahman_cross", "normalize folder spelling")
    safe_rename_dir(rows, cow_breed_root / "cholistani cross", "cholistani_cross", "normalize folder spelling")
    safe_rename_dir(rows, cow_breed_root / "sahiwal cross", "sahiwal_cross", "normalize folder spelling")
    safe_rename_dir(rows, cow_breed_root / "unidentified (mixed)", "_excluded_unidentified_mixed", "unknown/mixed placeholder")

    sheep_numeric_root = base / "sheep-breeds-4skwhnrscr" / "dataset_2" / "1selected"
    if sheep_numeric_root.exists():
        for folder in sorted([p for p in sheep_numeric_root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
            safe_rename_dir(rows, folder, f"_excluded_numeric_{folder.name}", "numeric folder has no source breed mapping")

    # YOLO source mapping/metadata fixes.
    fix_yaml_names(
        rows,
        REPO_ROOT / "data" / "raw" / "yolo_sources" / "detection-of-stray-dogs-v1-rf" / "data.yaml",
        ["dog"],
        "single-class stray dog detector exported class as numeric 0",
    )
    fix_bactrian_metadata(rows)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["action", "path", "detail", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"report": str(REPORT), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
