from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml


def _load_original_downloader(module_path: Path):
    spec = importlib.util.spec_from_file_location("original_download", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import downloader module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _target_dir_for_dataset(raw_dir: Path, dataset_id: str, entry: dict) -> Path:
    wikimedia = entry.get("wikimedia_commons")
    if isinstance(wikimedia, dict):
        output_subdir = str(wikimedia.get("output_subdir", "")).strip()
        if output_subdir:
            return raw_dir / output_subdir
    return raw_dir / dataset_id


def _is_effectively_downloaded(raw_dir: Path, dataset_id: str, entry: dict) -> bool:
    target = _target_dir_for_dataset(raw_dir, dataset_id, entry)
    if not target.exists():
        return False
    return any(p.is_file() for p in target.rglob("*"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all configured dataset sources.")
    parser.add_argument("--force", action="store_true", help="Re-run download for all datasets, even if present.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional dataset id filter; can be repeated.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_src = repo_root / "data" / "sources" / "original_project_configs" / "download_commands.yaml"
    if not config_src.exists():
        raise FileNotFoundError(f"Missing config: {config_src}")

    # The original downloader expects: <repo_root>/configs/download_commands.yaml.
    tmp_repo = repo_root / ".tmp_download_repo"
    tmp_cfg_dir = tmp_repo / "configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_src, tmp_cfg_dir / "download_commands.yaml")

    cfg = yaml.safe_load(config_src.read_text(encoding="utf-8")) or {}
    datasets = cfg.get("datasets") or {}
    dataset_ids = sorted(datasets.keys())
    selected = set(x.strip() for x in args.dataset if x.strip())
    if selected:
        unknown = sorted(selected - set(dataset_ids))
        if unknown:
            raise KeyError(f"Unknown dataset ids: {', '.join(unknown)}")
        dataset_ids = [d for d in dataset_ids if d in selected]

    downloader_path = Path(r"C:\Users\User\Desktop\New folder\scripts\download.py")
    if not downloader_path.exists():
        raise FileNotFoundError(f"Original downloader not found: {downloader_path}")

    downloader = _load_original_downloader(downloader_path)
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = repo_root / "data" / "sources" / "automation_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"download_all_sources_{run_ts}.json"
    out_csv = out_dir / f"download_all_sources_{run_ts}.csv"

    results: list[dict[str, str]] = []
    for dataset_id in dataset_ids:
        entry = datasets.get(dataset_id) or {}
        if not args.force and _is_effectively_downloaded(raw_dir, dataset_id, entry):
            now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            print(f"[skipped] {dataset_id} (already present)")
            results.append(
                {
                    "dataset_id": dataset_id,
                    "status": "skipped",
                    "started_utc": now,
                    "finished_utc": now,
                    "error": "",
                }
            )
            continue

        started = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        status = "ok"
        error = ""
        try:
            downloader.run_download(dataset_id=dataset_id, raw_dir=raw_dir, repo_root=tmp_repo)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)
        finished = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        print(f"[{status}] {dataset_id}")
        results.append(
            {
                "dataset_id": dataset_id,
                "status": status,
                "started_utc": started,
                "finished_utc": finished,
                "error": error,
            }
        )
        # Persist progress after each dataset for interruption-safe resuming.
        partial = {
            "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "results_so_far": results,
        }
        out_json.write_text(json.dumps(partial, indent=2), encoding="utf-8")
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["dataset_id", "status", "started_utc", "finished_utc", "error"]
            )
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "total": len(results),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "raw_dir": str(raw_dir.resolve()),
        "config": str(config_src.resolve()),
        "results": results,
    }

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "status", "started_utc", "finished_utc", "error"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print(f"Summary: ok={summary['ok']} failed={summary['failed']} total={summary['total']}")


if __name__ == "__main__":
    main()
