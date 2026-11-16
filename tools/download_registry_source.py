from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import tarfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse

import requests


OLD_PROJECT_ROOT = Path(r"C:\Users\User\Desktop\New folder")
OLD_SCRIPT_DIR = OLD_PROJECT_ROOT / "scripts"
OLD_SPECIES_DIR = OLD_PROJECT_ROOT / "New folder" / "species"
RUNS_DIR_REL = Path("data") / "sources" / "automation_runs"
COMMONS = "https://commons.wikimedia.org"
RASTER_EXT_DEFAULT = {"jpg", "jpeg", "png", "webp", "gif", "tif", "tiff"}


WIKIMEDIA_LINK_FILE_BY_DATASET: dict[str, str] = {
    "alpaca-wikimedia": "camelidae_alpaca_photo_categories.txt",
    "bactrian-camel-wikimedia": "camelidae_bactrian_camel_photo_categories.txt",
    "cattle-breeds-wikimedia": "cattle_breeds_by_name_subcategory_links.txt",
    "chicken-breeds-wikimedia": "chicken_breeds_by_name_subcategory_links.txt",
    "dromedary-wikimedia": "camelidae_dromedary_photo_categories.txt",
    "goat-breeds-wikimedia": "goat_breeds_subcategory_links_from_unidentified.txt",
    "horse-breeds-wikimedia": "horse_breeds_by_name_subcategory_links.txt",
    "llama-wikimedia": "camelidae_llama_photo_categories.txt",
    "peafowl-wikimedia": "peafowl_photo_categories.txt",
    "pig-breeds-wikimedia": "pig_breeds_subcategory_links_from_unidentified.txt",
    "pigeon-breeds-wikimedia": "pigeon_breeds_subcategory_links_from_unidentified.txt",
    "rabbit-breeds-wikimedia": "rabbit_breeds_subcategory_links_from_unidentified.txt",
    "sheep-breeds-wikimedia": "sheep_breeds_by_name_subcategory_links.txt",
    "water-buffalo-wikimedia": "water_buffalo_photo_categories.txt",
}


WIKIMEDIA_FALLBACK_CATEGORIES: dict[str, list[str]] = {
    "turkey-breeds-wikimedia": [
        "https://commons.wikimedia.org/wiki/Category:Turkey_breeds",
        "https://commons.wikimedia.org/wiki/Category:Turkeys_by_breed",
    ],
    "goose-breeds-wikimedia": [
        "https://commons.wikimedia.org/wiki/Category:Goose_breeds",
        "https://commons.wikimedia.org/wiki/Category:Domestic_goose_breeds",
    ],
    "duck-breeds-wikimedia": [
        "https://commons.wikimedia.org/wiki/Category:Duck_breeds",
        "https://commons.wikimedia.org/wiki/Category:Domestic_duck_breeds",
    ],
}

WIKIMEDIA_RECURSIVE_DATASETS: dict[str, dict[str, Any]] = {
    "goose-breeds-wikimedia": {
        "seed_urls": [
            "https://commons.wikimedia.org/wiki/Category:Domestic_greylag_goose_breeds",
            "https://commons.wikimedia.org/wiki/Category:Domestic_swan_goose_breeds",
        ],
        "max_depth": 2,
        "max_total_files": 20,
        "thumb_width": 256,
        "skip_root_files": True,
        "subcategory_allow_fn": "_allow_goose_subcategory",
    },
    "duck-breeds-wikimedia": {
        "seed_urls": [
            "https://commons.wikimedia.org/wiki/Category:Bantam_ducks_(APA)",
            "https://commons.wikimedia.org/wiki/Category:Heavy_ducks_(APA)",
            "https://commons.wikimedia.org/wiki/Category:Light_ducks_(APA)",
            "https://commons.wikimedia.org/wiki/Category:Medium_ducks_(APA)",
        ],
        "max_depth": 2,
        "max_total_files": 20,
        "thumb_width": 256,
        "skip_root_files": True,
        "subcategory_allow_fn": "_allow_duck_subcategory",
    },
}


INAT_RULES: dict[str, dict[str, Any]] = {
    "inaturalist-camelus-ferus-cc0": {
        "taxon_id": 74138,
        "photo_license": "cc0",
        "max_images": 500,
        "target_subdir": "inaturalist-camelus-ferus-cc0",
    },
    "inaturalist-camelus-bactrianus-cc-by": {
        "taxon_id": 524409,
        "photo_license": "cc-by",
        "max_images": 500,
        "target_subdir": "inaturalist-camelus-bactrianus-cc-by",
    },
    "inaturalist-bactrian-camel-manual": {
        "taxon_id": 524409,
        "photo_license": "cc0",
        "max_images": 100,
        "target_subdir": "inaturalist-bactrian-camel-manual",
    },
}


def _load_registry(registry_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset_id = str(row.get("dataset_id", "")).strip()
            if dataset_id:
                out[dataset_id] = {k: "" if v is None else str(v) for k, v in row.items()}
    return out


def _load_old_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_old_project_imports() -> None:
    for p in (OLD_PROJECT_ROOT, OLD_SCRIPT_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _slug(value: str) -> str:
    x = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unknown"


def _dataset_species_slug(dataset_id: str) -> str:
    if dataset_id.endswith("-wikimedia"):
        return dataset_id.removesuffix("-wikimedia").replace("-breeds", "").replace("-", "_")
    return _slug(dataset_id)


def _wikimedia_search_categories(query: str, limit: int = 12) -> list[str]:
    api = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srnamespace": 14,  # Category namespace
        "srlimit": min(max(1, int(limit)), 50),
    }
    resp = requests.get(api, params=params, timeout=60, headers={"User-Agent": "AnimalBreedDataCollector/1.0"})
    resp.raise_for_status()
    payload = resp.json()
    out: list[str] = []
    for item in payload.get("query", {}).get("search", []):
        title = str(item.get("title", "")).strip()
        if title.startswith("Category:"):
            out.append(f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}")
    # preserve order, unique
    seen = set()
    deduped: list[str] = []
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        deduped.append(u)
    return deduped


def _extract_commons_category_urls(text: str) -> list[str]:
    pattern = r"https://commons\.wikimedia\.org/wiki/Category:[^\s\}\{]+"
    urls = re.findall(pattern, text)
    out: list[str] = []
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _species_query_terms(dataset_id: str) -> list[str]:
    species = _dataset_species_slug(dataset_id).replace("_", " ")
    if species == "water buffalo":
        return ["Category:Water buffalo breeds", "Category:Water buffalo", "Category:Bubalus bubalis"]
    if species == "dromedary":
        return ["Category:Dromedary breeds", "Category:Camelus dromedarius", "Category:Dromedaries"]
    if species == "goose":
        return ["Category:Goose breeds", "Category:Domestic goose breeds", "Category:Geese"]
    if species == "duck":
        return ["Category:Duck breeds", "Category:Domestic duck breeds", "Category:Ducks"]
    if species == "peafowl":
        return ["Category:Peafowl", "Category:Peafowl breeds", "Category:Pavo cristatus"]
    return [f"Category:{species} breeds", f"Category:{species}", species]


def _commons_category_title_from_url(category_url: str) -> str:
    parsed = urlparse(category_url)
    path = parsed.path or ""
    if "/wiki/" in path:
        token = path.split("/wiki/", 1)[1]
    else:
        token = path.rsplit("/", 1)[-1]
    token = unquote(token).strip()
    if not token.startswith("Category:"):
        token = f"Category:{token}"
    return token.replace(" ", "_")


def _commons_file_page_url(title: str) -> str:
    clean = title.replace(" ", "_").strip()
    return f"{COMMONS}/wiki/{quote(clean, safe=':')}"


def _allow_goose_subcategory(parent_title: str, child_title: str, depth: int) -> bool:
    return True


def _allow_duck_subcategory(parent_title: str, child_title: str, depth: int) -> bool:
    text = child_title.lower()
    if "pcgb" in text:
        return False
    return True


def _commons_api_json(session: requests.Session, params: dict[str, Any]) -> dict[str, Any]:
    api = "https://commons.wikimedia.org/w/api.php"
    request_params = dict(params)
    request_params.setdefault("format", "json")
    request_params.setdefault("formatversion", 2)
    backoff = 2.0
    attempts = 0
    while True:
        attempts += 1
        try:
            resp = session.get(api, params=request_params, timeout=60)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and str(retry_after).isdigit():
                    wait_s = float(int(retry_after))
                else:
                    wait_s = backoff
                    backoff = min(backoff * 2.0, 60.0)
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            return payload
        except (requests.RequestException, ValueError):
            if attempts >= 6:
                raise
            wait_s = min(60.0, backoff)
            backoff = min(backoff * 2.0, 60.0)
            time.sleep(wait_s)


def _download_commons_media_bounded(
    session: requests.Session,
    url: str,
    dest: Path,
    referer: str,
    timeout_s: int = 120,
    max_attempts: int = 3,
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    backoff = 4.0
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            with session.get(url, headers={"Referer": referer}, stream=True, timeout=timeout_s) as response:
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and str(retry_after).isdigit():
                        wait_s = float(int(retry_after))
                    else:
                        wait_s = backoff
                        backoff = min(backoff * 2.0, 30.0)
                    if attempt < max_attempts:
                        time.sleep(wait_s)
                        continue
                    return False

                response.raise_for_status()
                tmp_dest = dest.with_suffix(dest.suffix + ".part")
                with tmp_dest.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            handle.write(chunk)
            try:
                tmp_dest.replace(dest)
            except Exception:
                if dest.exists():
                    try:
                        dest.unlink()
                    except Exception:
                        pass
                tmp_dest.replace(dest)
            return True
        except requests.RequestException:
            if attempt >= max_attempts:
                return False
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
        except Exception:
            return False
    return False


def _get_commons_html_bounded(
    session: requests.Session,
    url: str,
    referer: str | None = None,
    timeout_s: int = 60,
    max_attempts: int = 6,
) -> str:
    backoff = 2.0
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            headers = {}
            if referer:
                headers["Referer"] = referer
            resp = session.get(url, headers=headers, timeout=timeout_s)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and str(retry_after).isdigit():
                    wait_s = float(int(retry_after))
                else:
                    wait_s = backoff
                    backoff = min(backoff * 2.0, 30.0)
                if attempt < max_attempts:
                    time.sleep(min(wait_s, 15.0))
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            if attempt >= max_attempts:
                raise
            time.sleep(min(backoff, 15.0))
            backoff = min(backoff * 2.0, 30.0)


def _iter_commons_category_members(
    session: requests.Session,
    category_title: str,
    member_types: str = "file|subcat",
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmtype": member_types,
        "cmlimit": "max",
    }
    members: list[dict[str, Any]] = []
    try:
        while True:
            payload = _commons_api_json(session, params)
            chunk = payload.get("query", {}).get("categorymembers", []) or []
            members.extend(chunk)
            cont = payload.get("continue")
            if not cont:
                break
            params.update({k: v for k, v in cont.items() if k != "continue"})
        return members
    except Exception:
        return _iter_commons_category_members_html(session=session, category_title=category_title, member_types=member_types)


def _iter_commons_category_members_html(
    session: requests.Session,
    category_title: str,
    member_types: str = "file|subcat",
) -> list[dict[str, Any]]:
    from bs4 import BeautifulSoup

    page_url = f"{COMMONS}/wiki/{quote(category_title.replace(' ', '_'), safe=':')}"
    results: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    next_url: str | None = page_url
    while next_url:
        html = _get_commons_html_bounded(session=session, url=next_url, referer=page_url)
        soup = BeautifulSoup(html, "html.parser")
        content = soup.select_one("#mw-content-text") or soup
        if "file" in member_types:
            for a in content.select('a[href^="/wiki/File:"]'):
                href = a.get("href")
                if not href:
                    continue
                title = unquote(href.split("/wiki/", 1)[1]).replace("_", " ")
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                results.append({"ns": 6, "title": title})
        if "subcat" in member_types:
            for a in content.select('a[href^="/wiki/Category:"]'):
                href = a.get("href")
                if not href:
                    continue
                title = unquote(href.split("/wiki/", 1)[1]).replace("_", " ")
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                results.append({"ns": 14, "title": title})
        next_anchor = soup.select_one("a.mw-nextlink") or soup.select_one('a[rel="next"]')
        if next_anchor and next_anchor.get("href"):
            next_url = requests.compat.urljoin(COMMONS, str(next_anchor["href"]))
        else:
            next_url = None
    return results


def _download_wikimedia_recursive(
    dataset_id: str,
    raw_dir: Path,
    seed_urls: list[str],
    max_depth: int,
    max_total_files: int,
    thumb_width: int,
    skip_root_files: bool,
    subcategory_allow_fn: Callable[[str, str, int], bool] | None,
) -> dict[str, Any]:
    _ensure_old_project_imports()
    category_module = _load_old_module(OLD_SCRIPT_DIR / "download_wikimedia_category.py", "old_wikimedia_category_dl")

    out_root = raw_dir / "wikimedia"
    out_root.mkdir(parents=True, exist_ok=True)
    species = _dataset_species_slug(dataset_id)
    species_root = out_root / species
    species_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "AnimalBreedDataCollector/1.0",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    seen_categories: set[str] = set()
    seen_files: set[str] = set()
    seed_reports: list[dict[str, Any]] = []
    for seed_url in seed_urls:
        seed_title = _commons_category_title_from_url(seed_url)
        seed_label = _slug(seed_title.split("Category:", 1)[-1])
        label_dir = species_root / seed_label
        label_dir.mkdir(parents=True, exist_ok=True)
        report = _crawl_commons_category_tree(
            category_module=category_module,
            session=session,
            category_title=seed_title,
            out_dir=label_dir,
            thumb_width=thumb_width,
            max_depth=max_depth,
            max_total_files=max_total_files,
            skip_root_files=skip_root_files,
            subcategory_allow_fn=subcategory_allow_fn,
            seen_categories=seen_categories,
            seen_files=seen_files,
        )
        report["seed_url"] = seed_url
        report["seed_label"] = seed_label
        seed_reports.append(report)

    return {
        "mode": "recursive_categories",
        "max_depth": max_depth,
        "seed_count": len(seed_urls),
        "seed_reports": seed_reports,
    }


def _crawl_commons_category_tree(
    category_module: Any,
    session: requests.Session,
    category_title: str,
    out_dir: Path,
    thumb_width: int | None,
    max_depth: int,
    max_total_files: int,
    skip_root_files: bool,
    subcategory_allow_fn: Callable[[str, str, int], bool] | None,
    seen_categories: set[str],
    seen_files: set[str],
) -> dict[str, Any]:
    stack: list[tuple[str, int]] = [(category_title, 0)]
    downloaded = 0
    skipped = 0
    failed = 0
    categories_visited = 0
    categories_enqueued = 0
    failure_samples: list[str] = []

    while stack:
        current_title, depth = stack.pop()
        if current_title in seen_categories:
            continue
        seen_categories.add(current_title)
        categories_visited += 1

        try:
            members = _iter_commons_category_members(session=session, category_title=current_title)
        except Exception as exc:
            if len(failure_samples) < 5:
                failure_samples.append(f"{current_title}: {type(exc).__name__}: {exc}")
            failed += 1
            continue

        for item in members:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            if title.startswith("File:"):
                if skip_root_files and depth == 0:
                    continue
                file_page_url = _commons_file_page_url(title)
                if file_page_url in seen_files:
                    continue
                seen_files.add(file_page_url)
                try:
                    media_url, filename = category_module._get_original_or_thumb_url(
                        session=session,
                        file_page_url=file_page_url,
                        thumb_width=thumb_width,
                    )
                    ext = Path(urlparse(media_url).path).suffix.lower().lstrip(".")
                    if ext and ext not in RASTER_EXT_DEFAULT:
                        skipped += 1
                        continue
                    safe_name = category_module._safe_filename(filename)
                    safe_name = category_module._ensure_extension(filename=safe_name, ext=ext)
                    safe_name = category_module._fit_destination_name(parent_dir=out_dir, filename=safe_name)
                    destination = out_dir / safe_name
                    if destination.exists() and destination.stat().st_size > 0:
                        skipped += 1
                        continue
                    ok = _download_commons_media_bounded(
                        session=session,
                        url=media_url,
                        dest=destination,
                        referer=file_page_url,
                    )
                    if ok:
                        downloaded += 1
                        if max_total_files > 0 and downloaded >= max_total_files:
                            break
                    else:
                        failed += 1
                        if len(failure_samples) < 5:
                            failure_samples.append(f"{title}: download_failed")
                    if max_total_files > 0 and downloaded >= max_total_files:
                        break
                except Exception as exc:
                    if len(failure_samples) < 5:
                        failure_samples.append(f"{title}: {type(exc).__name__}: {exc}")
                    failed += 1
            elif title.startswith("Category:") and depth < max_depth:
                if subcategory_allow_fn is not None and not subcategory_allow_fn(current_title, title, depth):
                    continue
                categories_enqueued += 1
                stack.append((title, depth + 1))

        if max_total_files > 0 and downloaded >= max_total_files:
            break

    return {
        "category_title": category_title,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "categories_visited": categories_visited,
        "categories_enqueued": categories_enqueued,
        "out_dir": str(out_dir),
        "failure_samples": failure_samples,
    }


def _download_wikimedia(dataset_id: str, raw_dir: Path) -> dict[str, Any]:
    out_root = raw_dir / "wikimedia"
    out_root.mkdir(parents=True, exist_ok=True)
    species = _dataset_species_slug(dataset_id)

    recursive_cfg = WIKIMEDIA_RECURSIVE_DATASETS.get(dataset_id)
    if recursive_cfg is not None:
        recursive_depth = int(recursive_cfg.get("max_depth", 2))
        max_total_files = int(recursive_cfg.get("max_total_files", 0))
        thumb_width = int(recursive_cfg.get("thumb_width", 512))
        skip_root_files = bool(recursive_cfg.get("skip_root_files", True))
        allow_name = str(recursive_cfg.get("subcategory_allow_fn", "")).strip()
        allow_fn: Callable[[str, str, int], bool] | None = globals().get(allow_name) if allow_name else None
        seed_urls = [str(u) for u in recursive_cfg.get("seed_urls", []) if str(u).strip()]
        seen_seed = set()
        dedup_seed: list[str] = []
        for url in seed_urls:
            if url in seen_seed:
                continue
            seen_seed.add(url)
            dedup_seed.append(url)
        seed_urls = dedup_seed
        if not seed_urls:
            raise RuntimeError(f"No Wikimedia seed categories available for recursive crawl: {dataset_id}")
        return _download_wikimedia_recursive(
            dataset_id=dataset_id,
            raw_dir=raw_dir,
            seed_urls=seed_urls,
            max_depth=recursive_depth,
            max_total_files=max_total_files,
            thumb_width=thumb_width,
            skip_root_files=skip_root_files,
            subcategory_allow_fn=allow_fn,
        )

    _ensure_old_project_imports()
    links_module = _load_old_module(OLD_SCRIPT_DIR / "download_wikimedia_links.py", "old_wikimedia_links")
    run_links_download = links_module.run_wikimedia_links_download

    links_name = WIKIMEDIA_LINK_FILE_BY_DATASET.get(dataset_id)
    if links_name:
        links_file = OLD_SPECIES_DIR / links_name
        if not links_file.exists():
            raise FileNotFoundError(f"Missing Wikimedia links file: {links_file}")
        run_links_download(
            links_file=links_file,
            out_root=out_root,
            species=species,
            sleep_s=0.05,
            retries=2,
        )
        return {"mode": "links_file", "links_file": str(links_file)}

    fallback_urls = list(WIKIMEDIA_FALLBACK_CATEGORIES.get(dataset_id, []))
    if not fallback_urls:
        # dynamic fallback through Commons category search
        for term in _species_query_terms(dataset_id):
            try:
                fallback_urls.extend(_wikimedia_search_categories(term, limit=10))
            except Exception:
                continue
    # de-dup while preserving order
    seen_f = set()
    dedup_fallback: list[str] = []
    for u in fallback_urls:
        if u in seen_f:
            continue
        seen_f.add(u)
        dedup_fallback.append(u)
    fallback_urls = dedup_fallback
    if not fallback_urls:
        raise RuntimeError(f"No Wikimedia link mapping/fallback categories for {dataset_id}")

    category_module = _load_old_module(
        OLD_SCRIPT_DIR / "download_wikimedia_category.py", "old_wikimedia_category_dl"
    )
    run_category = category_module.run_wikimedia_category_download
    out_dir = out_root / species
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_count = 0
    for url in fallback_urls:
        try:
            run_category(
                category_url=url,
                out_dir=out_dir,
                label=_slug(url.split("Category:", 1)[-1]),
                sleep_s=0.05,
                max_files=0,
            )
            ok_count += 1
        except Exception:
            continue
    if ok_count == 0:
        raise RuntimeError(f"All fallback Wikimedia categories failed for {dataset_id}")
    return {
        "mode": "fallback_categories",
        "ok_categories": ok_count,
        "attempted_categories": len(fallback_urls),
        "fallback_urls_sample": fallback_urls[:10],
    }


def _save_bytes_to_file(content: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def _maybe_extract_archive(archive_path: Path, target_dir: Path) -> bool:
    suffixes = [s.lower() for s in archive_path.suffixes]
    if suffixes[-1:] == [".zip"]:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
        return True
    if suffixes[-2:] == [".tar", ".gz"] or suffixes[-1:] == [".tar"]:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(target_dir)
        return True
    return False


def _extract_roboflow_version(dataset_id: str, source_ref: str) -> int:
    m = re.search(r"/dataset/(\d+)", source_ref)
    if m:
        return int(m.group(1))
    m = re.search(r"-v(\d+)$", dataset_id)
    if m:
        return int(m.group(1))
    return 1


def _download_roboflow(dataset_id: str, source_ref: str, raw_dir: Path) -> dict[str, Any]:
    out_dir = raw_dir / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_ref.startswith("local/"):
        local_name = Path(source_ref.replace("local/", "", 1)).name
        candidate_paths = [
            Path(os.environ.get("USERPROFILE", "")) / "Downloads" / local_name,
            Path(r"C:\Users\User\Downloads") / local_name,
        ]
        for p in candidate_paths:
            if p.exists() and p.is_file():
                copied = out_dir / p.name
                copied.write_bytes(p.read_bytes())
                extracted = _maybe_extract_archive(copied, out_dir)
                return {"mode": "local_file", "source_file": str(p), "copied_file": str(copied), "extracted": extracted}
        # Fallback: if registry has a Roboflow Universe URL, use that instead of local zip.
        source_registry = Path(__file__).resolve().parents[1] / "data" / "sources" / "source_registry_v1.csv"
        if source_registry.exists():
            with source_registry.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if str(row.get("dataset_id", "")).strip() != dataset_id:
                        continue
                    url = str(row.get("source_url", "")).strip()
                    m = re.search(r"universe\.roboflow\.com/([^/]+)/([^/]+)/dataset/(\d+)", url)
                    if m:
                        ws, prj, ver = m.group(1), m.group(2), m.group(3)
                        source_ref = f"{ws}/{prj}/dataset/{ver}"
                        break
        if source_ref.startswith("local/"):
            raise FileNotFoundError(f"Local source file not found for {dataset_id}: {local_name}")

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if api_key:
        try:
            from roboflow import Roboflow

            version = _extract_roboflow_version(dataset_id, source_ref)
            parts = [x for x in source_ref.split("/") if x]
            if len(parts) < 2:
                raise RuntimeError(f"Unexpected roboflow source_ref format: {source_ref}")
            workspace = parts[0]
            project = parts[1]
            rf = Roboflow(api_key=api_key)
            ws = rf.workspace(workspace)
            prj = ws.project(project)
            # If version was not explicitly given in source_ref, use latest known project version.
            explicit_version = bool(re.search(r"/dataset/(\d+)", source_ref))
            candidate_versions: list[int] = []
            if explicit_version:
                candidate_versions.append(version)
            else:
                try:
                    known = prj.versions()
                    candidate_versions.extend(
                        [int(v.version) for v in known if getattr(v, "version", None) is not None]
                    )
                except Exception:
                    candidate_versions.append(version)
                if not candidate_versions:
                    candidate_versions.append(version)
            # preserve order, unique
            seen_v = set()
            candidate_versions = [v for v in candidate_versions if not (v in seen_v or seen_v.add(v))]

            preferred_formats = ["yolov8", "folder", "multiclass", "clip", "coco", "voc"]
            sdk_attempts: list[dict[str, Any]] = []
            success_meta: dict[str, Any] | None = None
            for ver_no in candidate_versions:
                # First attempt via documented Roboflow API export links.
                try:
                    meta_url = f"https://api.roboflow.com/{workspace}/{project}/{ver_no}"
                    meta_resp = requests.get(meta_url, params={"api_key": api_key}, timeout=90)
                    meta_resp.raise_for_status()
                    meta = meta_resp.json()
                    exports = meta.get("version", {}).get("exports", []) or []
                    formats = [f for f in preferred_formats if f in exports] + [f for f in exports if f not in preferred_formats]
                    for fmt in formats:
                        try:
                            export_url = f"{meta_url}/{fmt}"
                            export_meta = requests.get(export_url, params={"api_key": api_key}, timeout=90)
                            export_meta.raise_for_status()
                            export_payload = export_meta.json()
                            link = str(export_payload.get("export", {}).get("link", "")).strip()
                            if not link:
                                sdk_attempts.append(
                                    {"version": ver_no, "format": fmt, "error": "missing_export_link"}
                                )
                                continue
                            zip_path = out_dir / f"{dataset_id}_v{ver_no}_{fmt}.zip"
                            blob = requests.get(link, timeout=240)
                            blob.raise_for_status()
                            zip_path.write_bytes(blob.content)
                            if zipfile.is_zipfile(zip_path):
                                with zipfile.ZipFile(zip_path, "r") as zf:
                                    zf.extractall(out_dir)
                                success_meta = {"version": ver_no, "format": fmt, "method": "roboflow_export_api"}
                                break
                            sdk_attempts.append(
                                {"version": ver_no, "format": fmt, "error": "download_not_zip"}
                            )
                        except Exception as exc:
                            sdk_attempts.append({"version": ver_no, "format": fmt, "error": str(exc)})
                    if success_meta:
                        break
                except Exception as exc:
                    sdk_attempts.append({"version": ver_no, "error": f"api_meta_error: {exc}"})

                # Fallback via SDK downloader.
                try:
                    ver = prj.version(ver_no)
                except Exception as exc:
                    sdk_attempts.append({"version": ver_no, "error": f"version_error: {exc}"})
                    continue
                for fmt in preferred_formats:
                    try:
                        ver.download(fmt, location=str(out_dir))
                        # SDK can report success without materialized files for some formats/projects.
                        # Accept only if files now exist in target folder.
                        if _count_files(out_dir) > 0:
                            success_meta = {"version": ver_no, "format": fmt, "method": "roboflow_sdk"}
                            break
                        sdk_attempts.append({"version": ver_no, "format": fmt, "error": "sdk_no_files_written"})
                    except Exception as exc:
                        sdk_attempts.append({"version": ver_no, "format": fmt, "error": str(exc)})
                if success_meta:
                    break
            if not success_meta:
                raise RuntimeError(f"roboflow_sdk_exhausted attempts={sdk_attempts}")
            return {
                "mode": "roboflow_sdk",
                "workspace": workspace,
                "project": project,
                "version": success_meta["version"],
                "format": success_meta["format"],
                "method": success_meta["method"],
            }
        except Exception as exc:
            # fall back to raw URL attempts
            sdk_error = str(exc)
    else:
        sdk_error = "ROBOFLOW_API_KEY not set"

    version = _extract_roboflow_version(dataset_id, source_ref)
    parts = [x for x in source_ref.split("/") if x]
    if "dataset" in parts:
        base_ref = "/".join(parts[: parts.index("dataset")])
    else:
        base_ref = source_ref.strip("/")

    trial_urls = [
        f"https://universe.roboflow.com/{base_ref}/dataset/{version}/download/yolov8",
        f"https://universe.roboflow.com/{base_ref}/dataset/{version}/download/coco",
        f"https://universe.roboflow.com/{base_ref}/dataset/{version}/download/voc",
        f"https://universe.roboflow.com/{base_ref}/dataset/{version}/download/folder",
        f"https://universe.roboflow.com/{base_ref}/dataset/{version}/download/multiclass",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    attempts: list[dict[str, Any]] = []
    for url in trial_urls:
        try:
            resp = requests.get(url, timeout=90, allow_redirects=True, headers=headers)
            attempt = {"url": url, "status": int(resp.status_code), "content_type": resp.headers.get("content-type", "")}
            attempts.append(attempt)
            if resp.status_code == 200:
                maybe_zip = out_dir / f"{dataset_id}.download"
                _save_bytes_to_file(resp.content, maybe_zip)
                if zipfile.is_zipfile(maybe_zip):
                    with zipfile.ZipFile(maybe_zip, "r") as zf:
                        zf.extractall(out_dir)
                    return {"mode": "direct_url_zip", "url": url, "attempts": attempts}
                if maybe_zip.stat().st_size > 0:
                    maybe_zip.rename(out_dir / f"{dataset_id}.bin")
                    return {"mode": "direct_url_nonzip", "url": url, "attempts": attempts}
        except Exception as exc:
            attempts.append({"url": url, "error": str(exc)})

    raise RuntimeError(f"Roboflow download blocked for {dataset_id}. sdk_error={sdk_error}. attempts={attempts}")


def _inat_image_url(photo: dict[str, Any]) -> str:
    url = str(photo.get("url", "")).strip()
    if not url:
        return ""
    return url.replace("/square.", "/original.")


def _download_inaturalist(dataset_id: str, raw_dir: Path) -> dict[str, Any]:
    rule = INAT_RULES.get(dataset_id)
    if not rule:
        raise RuntimeError(f"No iNaturalist rule configured for {dataset_id}")

    out_dir = raw_dir / str(rule["target_subdir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    max_images = int(rule["max_images"])
    taxon_id = int(rule["taxon_id"])
    photo_license = str(rule["photo_license"])

    params = {
        "taxon_id": taxon_id,
        "photo_license": photo_license,
        "quality_grade": "any",
        "per_page": 200,
        "order_by": "created_at",
        "order": "desc",
    }
    headers = {"User-Agent": "AnimalBreedDataCollector/1.0"}

    page = 1
    downloaded = 0
    attrib_rows: list[dict[str, str]] = []
    seen = set()
    while downloaded < max_images:
        params["page"] = page
        resp = requests.get("https://api.inaturalist.org/v1/observations", params=params, headers=headers, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        if not results:
            break
        for obs in results:
            obs_id = str(obs.get("id", ""))
            photos = obs.get("photos", []) or []
            for photo in photos:
                img_url = _inat_image_url(photo)
                if not img_url or img_url in seen:
                    continue
                seen.add(img_url)
                ext = Path(urlparse(img_url).path).suffix.lower()
                if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                    ext = ".jpg"
                file_name = f"obs{obs_id}_photo{photo.get('id', 'x')}{ext}"
                dst = out_dir / file_name
                if dst.exists() and dst.stat().st_size > 0:
                    continue
                try:
                    img = requests.get(img_url, headers=headers, timeout=120)
                    img.raise_for_status()
                    dst.write_bytes(img.content)
                    downloaded += 1
                    attrib_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "observation_id": obs_id,
                            "photo_id": str(photo.get("id", "")),
                            "image_path": str(dst.resolve()),
                            "image_url": img_url,
                            "license_code": str(photo.get("license_code", "")),
                            "attribution": str(photo.get("attribution", "")),
                            "photographer": str(photo.get("attribution", "")),
                            "observation_url": f"https://www.inaturalist.org/observations/{obs_id}",
                        }
                    )
                    if downloaded >= max_images:
                        break
                except Exception:
                    continue
            if downloaded >= max_images:
                break
        page += 1

    if attrib_rows:
        attr_path = out_dir / "_attribution.csv"
        with attr_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(attrib_rows[0].keys()))
            w.writeheader()
            w.writerows(attrib_rows)

    return {
        "taxon_id": taxon_id,
        "photo_license": photo_license,
        "downloaded": downloaded,
        "out_dir": str(out_dir),
    }


def _build_turkey_links_and_map(tmp_dir: Path) -> tuple[Path, Path]:
    index_url = "https://www.feathersite.com/Poultry/Turkeys/BRKTurkey.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(index_url, timeout=90, headers=headers)
    resp.raise_for_status()
    html = resp.text

    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    soup = BeautifulSoup(html, "html.parser")
    links: list[tuple[str, str]] = [("Index", index_url)]
    seen_urls = {index_url}

    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        full = urljoin(index_url, href)
        host = (urlparse(full).hostname or "").lower()
        if not host.endswith("feathersite.com"):
            continue
        if "BRKTurkey" in full:
            continue
        text = " ".join(a.get_text(" ", strip=True).split())
        if not text:
            continue
        if full in seen_urls:
            continue
        seen_urls.add(full)
        links.append((text, full))

    links_file = tmp_dir / "turkey_feathersite_links.txt"
    with links_file.open("w", encoding="utf-8") as f:
        for label, url in links:
            f.write(f"{label} {url}\n")

    breeds = []
    seen_canon = set()
    for label, _ in links:
        canon = _slug(label)
        if canon in {"index"}:
            continue
        if canon in seen_canon:
            continue
        seen_canon.add(canon)
        breeds.append({"canonical": canon, "aliases": [label]})

    map_file = tmp_dir / "turkey_feathersite_rename_map.yml"
    import yaml

    map_file.write_text(yaml.safe_dump({"breeds": breeds}, sort_keys=False), encoding="utf-8")
    return links_file, map_file


def _download_feathersite(dataset_id: str, raw_dir: Path) -> dict[str, Any]:
    if dataset_id != "turkey-breeds-feathersite":
        raise RuntimeError(f"Unsupported feathersite dataset id: {dataset_id}")

    _ensure_old_project_imports()
    feather_module = _load_old_module(OLD_SCRIPT_DIR / "download_feathersite_links.py", "old_feathersite_dl")
    run_feathersite = feather_module.run_feathersite_links_download

    tmp_dir = raw_dir / "_tmp_feathersite"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    links_file, rename_map = _build_turkey_links_and_map(tmp_dir)

    out_root = raw_dir / "feathersite"
    run_feathersite(
        links_file=links_file,
        rename_map_file=rename_map,
        out_root=out_root,
        species="turkey",
        sleep_s=0.05,
        retries=2,
        timeout_s=60,
    )
    return {"links_file": str(links_file), "rename_map": str(rename_map), "out_root": str(out_root)}


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a dataset source based on datasets_registry metadata.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--raw-dir", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else (repo_root / "data" / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    registry_path = repo_root / "data" / "sources" / "original_project_configs" / "datasets_registry.csv"
    registry = _load_registry(registry_path)
    ds = args.dataset_id.strip()
    if ds not in registry:
        raise KeyError(f"Unknown dataset id in registry: {ds}")
    row = registry[ds]
    source_type = row.get("source_type", "").strip().lower()
    source_ref = row.get("source_ref", "").strip()

    run_dir = repo_root / RUNS_DIR_REL
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    status = "ok"
    detail: dict[str, Any] = {}
    error = ""
    try:
        if source_type == "wikimedia":
            detail = _download_wikimedia(ds, raw_dir)
        elif source_type == "inaturalist":
            detail = _download_inaturalist(ds, raw_dir)
        elif source_type == "feathersite":
            detail = _download_feathersite(ds, raw_dir)
        elif source_type == "roboflow":
            detail = _download_roboflow(ds, source_ref, raw_dir)
        else:
            raise RuntimeError(f"Unsupported source_type for downloader: {source_type}")
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)

    finished = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    target_dir = raw_dir / ds
    if source_type == "wikimedia":
        target_dir = raw_dir / "wikimedia" / _dataset_species_slug(ds)
    elif source_type == "feathersite":
        target_dir = raw_dir / "feathersite" / "turkey"
    elif source_type == "inaturalist":
        cfg = INAT_RULES.get(ds, {})
        target_dir = raw_dir / str(cfg.get("target_subdir", ds))

    report = {
        "dataset_id": ds,
        "source_type": source_type,
        "source_ref": source_ref,
        "status": status,
        "started_utc": started,
        "finished_utc": finished,
        "target_dir": str(target_dir.resolve()),
        "file_count": _count_files(target_dir),
        "detail": detail,
        "error": error,
    }
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out = run_dir / f"download_registry_source_{_slug(ds)}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
