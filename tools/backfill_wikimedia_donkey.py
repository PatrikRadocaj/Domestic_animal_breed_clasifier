from __future__ import annotations

import importlib.util
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup


ROOT_CATEGORY_URL = "https://commons.wikimedia.org/wiki/Category:Donkey_breeds"
USER_AGENT = "CommonsDatasetDownloader/1.0 (contact: https://github.com/your-org)"


def _slug(value: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "category"


def _load_downloader_function():
    script_path = Path(r"C:\Users\User\Desktop\New folder\scripts\download_wikimedia_category.py")
    root_path = Path(r"C:\Users\User\Desktop\New folder")
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    import sys

    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    spec = importlib.util.spec_from_file_location("wikimedia_category_dl", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_wikimedia_category_download


def _discover_subcategories() -> list[tuple[str, int]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    try:
        resp = requests.get(ROOT_CATEGORY_URL, headers=headers, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        sub = soup.select_one("#mw-subcategories")
        if sub is None:
            return []

        rows: list[tuple[str, int]] = []
        for li in sub.select("li"):
            a = li.select_one('a[href^="/wiki/Category:"]')
            if a is None:
                continue
            href = a.get("href")
            if not href:
                continue
            url = "https://commons.wikimedia.org" + href
            text = li.get_text(" ", strip=True)
            m = re.search(r"\((?:[^)]*?)(\d+)\sF\)", text)
            file_count = int(m.group(1)) if m else 0
            rows.append((url, file_count))
        return rows
    except Exception:
        # Fallback: MediaWiki API categorymembers for robust discovery under stricter anti-bot pages.
        category = "Category:Donkey breeds"
        cmcontinue = ""
        rows = []
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmtype": "subcat",
                "cmlimit": "500",
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            resp = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
            members = payload.get("query", {}).get("categorymembers", [])
            for item in members:
                title = str(item.get("title", "")).strip()
                if not title.startswith("Category:"):
                    continue
                rows.append(
                    (
                        f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                        1,  # Unknown file count in API-only fallback; keep as eligible target.
                    )
                )
            cmcontinue = payload.get("continue", {}).get("cmcontinue", "")
            if not cmcontinue:
                break
        return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "data" / "raw" / "wikimedia" / "donkey"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = repo_root / "data" / "sources" / "automation_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    state_path = run_dir / "wikimedia_donkey_backfill_state.json"
    log_path = run_dir / f"wikimedia_donkey_backfill_{run_ts}.json"

    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {"completed_urls": []}
    else:
        state = {"completed_urls": []}
    completed = set(state.get("completed_urls", []))

    run_wikimedia_category_download = _load_downloader_function()
    subcategories = _discover_subcategories()

    # Include root category and only subcategories with visible file counts.
    targets = [(ROOT_CATEGORY_URL, 18)] + [(u, c) for (u, c) in subcategories if c > 0]

    records: list[dict[str, str | int]] = []
    for url, declared_files in targets:
        if url in completed:
            records.append(
                {
                    "url": url,
                    "status": "skipped",
                    "declared_files": declared_files,
                    "error": "",
                }
            )
            continue

        try:
            # Keep sleep low; downloader already handles 429 backoff internally.
            run_wikimedia_category_download(
                category_url=url,
                out_dir=out_dir,
                label=_slug(url.split("Category:", 1)[-1]),
                sleep_s=0.05,
                user_agent=USER_AGENT,
                max_files=0,
            )
            completed.add(url)
            state["completed_urls"] = sorted(completed)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            records.append(
                {
                    "url": url,
                    "status": "ok",
                    "declared_files": declared_files,
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "url": url,
                    "status": "failed",
                    "declared_files": declared_files,
                    "error": str(exc),
                }
            )

    summary = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "target_count": len(targets),
        "ok": sum(1 for r in records if r["status"] == "ok"),
        "skipped": sum(1 for r in records if r["status"] == "skipped"),
        "failed": sum(1 for r in records if r["status"] == "failed"),
        "output_dir": str(out_dir.resolve()),
        "state_file": str(state_path.resolve()),
        "records": records,
    }
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {log_path}")
    print(f"Summary: ok={summary['ok']} skipped={summary['skipped']} failed={summary['failed']}")


if __name__ == "__main__":
    main()
