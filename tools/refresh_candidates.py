"""Periodic scouting artifact refresh with canonical URL dedup summary."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


def canonicalize(url: str) -> str:
    p = urlparse((url or "").strip())
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/")
    return f"https://{host}{path}" if host else ""


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    registry_path = repo / "data" / "sources" / "source_registry_v1.csv"
    candidates_path = repo / "data" / "sources" / "yolo_candidates_2026-05-27.csv"

    if not registry_path.exists() or not candidates_path.exists():
        raise FileNotFoundError("source registry or candidate file not found")

    reg = pd.read_csv(registry_path)
    cand = pd.read_csv(candidates_path)

    reg_urls = {canonicalize(str(u)) for u in reg.get("source_url", []).tolist()}
    cand["canonical_url"] = cand["source_url"].astype(str).map(canonicalize)
    cand["already_known"] = cand["canonical_url"].isin(reg_urls)

    summary = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "candidate_rows": int(len(cand)),
        "already_known": int(cand["already_known"].sum()),
        "new_candidates": int((~cand["already_known"]).sum()),
    }

    out_dir = repo / "data" / "sources" / "automation_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "weekly_scout_refresh.json"
    out_csv = out_dir / "weekly_scout_candidates.csv"

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cand.to_csv(out_csv, index=False)
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
