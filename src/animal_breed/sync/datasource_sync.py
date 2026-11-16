"""Cross-fork datasource sync guardrail with diff-based notifications."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import hmac
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import uuid4
import os
import csv


TARGET_SESSION_ID = "019e68b2-7716-7113-8854-5b16fb40ebef"
SCHEMA_VERSION = "datasource_sync.v1"
TRACKING_QUERY_KEYS = {
    "ref",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}

DEFAULT_SIGNING_KEY_ID = "datasource-sync-local-v1"

DATASOURCE_SURFACE_GLOBS: tuple[str, ...] = (
    "data/sources/**/*",
    "results/reports/sources/**/*",
    "configs/data/**/*",
)


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    source_url: str
    canonical_url: str
    record_hash: str
    file_path: str
    row_index: int


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonicalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = "https"
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or "/"
    while len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    filtered_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key.startswith("utm_") or normalized_key in TRACKING_QUERY_KEYS:
            continue
        filtered_query.append((key, value))
    query = urlencode(filtered_query, doseq=True)
    return urlunparse((scheme, host, path, "", query, ""))


def _iter_surface_files(repo_root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in DATASOURCE_SURFACE_GLOBS:
        files.update(path for path in repo_root.glob(pattern) if path.is_file())
    return sorted(files)


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _row_hash(row: dict[str, str]) -> str:
    normalized = {k: (v or "").strip() for k, v in sorted(row.items())}
    return sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()


def _infer_source_id(row: dict[str, str]) -> str:
    for key in ("dataset_id", "source_dataset", "source"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def _collect_source_entries(repo_root: Path, surface_files: list[Path]) -> dict[str, SourceEntry]:
    entries: dict[str, SourceEntry] = {}
    for path in surface_files:
        if path.suffix.lower() != ".csv":
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    continue
                if "source_url" not in reader.fieldnames:
                    continue
                for idx, row in enumerate(reader, start=1):
                    source_id = _infer_source_id(row)
                    source_url = (row.get("source_url") or "").strip()
                    canonical_url = canonicalize_url(source_url)
                    if not source_id and not canonical_url:
                        continue
                    key = f"{source_id}|{canonical_url}" if source_id else canonical_url
                    entries[key] = SourceEntry(
                        source_id=source_id,
                        source_url=source_url,
                        canonical_url=canonical_url,
                        record_hash=_row_hash(row),
                        file_path=str(path.resolve()),
                        row_index=idx,
                    )
        except OSError:
            continue
    return entries


@dataclass(frozen=True)
class SurfaceSnapshot:
    file_hashes: dict[str, str]
    entries: dict[str, SourceEntry]
    created_at: str
    source_snapshot_hash: str


def capture_snapshot(repo_root: Path) -> SurfaceSnapshot:
    files = _iter_surface_files(repo_root)
    file_hashes = {str(path.resolve()): _file_hash(path) for path in files}
    entries = _collect_source_entries(repo_root, files)
    key_material = [
        f"{key}|{entry.record_hash}"
        for key, entry in sorted(entries.items(), key=lambda item: item[0])
    ]
    source_snapshot_hash = sha256("\n".join(key_material).encode("utf-8")).hexdigest()
    return SurfaceSnapshot(
        file_hashes=file_hashes,
        entries=entries,
        created_at=utc_now(),
        source_snapshot_hash=source_snapshot_hash,
    )


def _diff_files(before: SurfaceSnapshot, after: SurfaceSnapshot) -> list[str]:
    changed: set[str] = set()
    all_paths = set(before.file_hashes) | set(after.file_hashes)
    for path in all_paths:
        if before.file_hashes.get(path) != after.file_hashes.get(path):
            changed.add(path)
    return sorted(changed)


def _diff_sources(before: SurfaceSnapshot, after: SurfaceSnapshot) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    added: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    all_keys = set(before.entries) | set(after.entries)
    for key in sorted(all_keys):
        left = before.entries.get(key)
        right = after.entries.get(key)
        if left is None and right is not None:
            added.append(
                {
                    "source_id": right.source_id,
                    "source_url": right.source_url,
                    "canonical_url": right.canonical_url,
                    "file_path": right.file_path,
                    "row_index": right.row_index,
                }
            )
        elif left is not None and right is None:
            removed.append(
                {
                    "source_id": left.source_id,
                    "source_url": left.source_url,
                    "canonical_url": left.canonical_url,
                    "file_path": left.file_path,
                    "row_index": left.row_index,
                }
            )
        elif left is not None and right is not None and left.record_hash != right.record_hash:
            updated.append(
                {
                    "source_id": right.source_id,
                    "source_url": right.source_url,
                    "canonical_url": right.canonical_url,
                    "file_path": right.file_path,
                    "row_index": right.row_index,
                }
            )
    return added, updated, removed


def infer_change_type(
    added_sources: list[dict[str, Any]],
    updated_sources: list[dict[str, Any]],
    removed_sources: list[dict[str, Any]],
) -> str:
    if added_sources and not updated_sources and not removed_sources:
        return "add"
    if removed_sources and not added_sources and not updated_sources:
        return "remove"
    if updated_sources and not added_sources and not removed_sources:
        return "update"
    if updated_sources and not removed_sources:
        return "reclassify"
    return "bulk_refresh"


def build_payload(
    origin_session_id: str,
    change_window_start: str,
    change_window_end: str,
    before: SurfaceSnapshot,
    after: SurfaceSnapshot,
    dedupe_basis: str = "canonical_url",
    action_hint: str = "Apply as authoritative datasource delta; fallback to manual_review on snapshot hash mismatch.",
) -> dict[str, Any] | None:
    affected_files = _diff_files(before, after)
    if not affected_files:
        return None
    added_sources, updated_sources, removed_sources = _diff_sources(before, after)

    change_type = (
        infer_change_type(added_sources, updated_sources, removed_sources)
        if (added_sources or updated_sources or removed_sources)
        else "bulk_refresh"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target_session_id": TARGET_SESSION_ID,
        "event_id": str(uuid4()),
        "timestamp_utc": change_window_end,
        "origin_session_id": origin_session_id,
        "change_type": change_type,
        "change_window_start": change_window_start,
        "change_window_end": change_window_end,
        "source_snapshot_hash": after.source_snapshot_hash,
        "previous_source_snapshot_hash": before.source_snapshot_hash,
        "affected_files": affected_files,
        "added_sources": added_sources,
        "updated_sources": updated_sources,
        "removed_sources": removed_sources,
        "dedupe_basis": dedupe_basis,
        "requires_fork_action": True,
        "action_hint": action_hint,
    }
    signature = _sign_payload(payload)
    payload["payload_signature"] = signature
    payload["signing_key_id"] = DEFAULT_SIGNING_KEY_ID
    return payload


def _sign_payload(payload: dict[str, Any]) -> str:
    secret = os.getenv("DATASOURCE_SYNC_SECRET", "").encode("utf-8")
    if not secret:
        # Deterministic fallback signature (non-secret) for integrity checks in local/dev.
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{sha256(normalized).hexdigest()}"
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "hmac-sha256:" + hmac.new(secret, normalized, digestmod="sha256").hexdigest()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _send_via_command(payload_path: Path) -> subprocess.CompletedProcess[str]:
    command_template = os.getenv("DATASOURCE_SYNC_SESSION_SEND_CMD", "").strip()
    if not command_template:
        raise RuntimeError(
            "DATASOURCE_SYNC_SESSION_SEND_CMD is not configured. "
            "Example: opencli codex send --thread-id {session_id} --message-file {payload_file}"
        )
    formatted = command_template.format(
        session_id=TARGET_SESSION_ID,
        payload_file=str(payload_path.resolve()),
    )
    args = shlex.split(formatted)
    return subprocess.run(args, capture_output=True, text=True, check=True)


def deliver_payload(
    payload: dict[str, Any],
    repo_root: Path,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    sync_dir = repo_root / "results" / "reports" / "sync"
    outbox_dir = sync_dir / "outbox"
    _ensure_dir(sync_dir)
    _ensure_dir(outbox_dir)

    payload_path = outbox_dir / f"{payload['event_id']}.json"
    ledger_path = sync_dir / "sent_event_ids.json"
    if ledger_path.exists():
        try:
            known_ids = set(json.loads(ledger_path.read_text(encoding="utf-8")))
        except Exception:
            known_ids = set()
        if payload["event_id"] in known_ids:
            return {
                "event_id": payload["event_id"],
                "target_session_id": TARGET_SESSION_ID,
                "status": "replay_blocked",
                "reason": "event_id already delivered",
                "payload_file": str(payload_path.resolve()),
            }

    _write_json(payload_path, payload)
    errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        try:
            process = _send_via_command(payload_path)
            receipt = {
                "event_id": payload["event_id"],
                "target_session_id": TARGET_SESSION_ID,
                "status": "sent",
                "attempts": attempt,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
                "payload_file": str(payload_path.resolve()),
                "sent_at": utc_now(),
            }
            _write_json(sync_dir / f"{payload['event_id']}.receipt.json", receipt)
            _append_ledger_event(sync_dir, payload["event_id"])
            return receipt
        except Exception as exc:  # noqa: BLE001
            error_type = "transport_error"
            text = str(exc).lower()
            if "not configured" in text:
                error_type = "configuration_error"
            elif "timed out" in text:
                error_type = "timeout_error"
            elif "permission" in text or "forbidden" in text or "unauthorized" in text:
                error_type = "auth_error"
            errors.append(f"attempt={attempt} type={error_type} error={exc}")
            if attempt < max_attempts:
                time.sleep(base_delay_seconds * (2 ** (attempt - 1)))

    failure_record = {
        "event_id": payload["event_id"],
        "target_session_id": TARGET_SESSION_ID,
        "status": "failed",
        "attempts": max_attempts,
        "errors": errors,
        "payload_file": str(payload_path.resolve()),
        "failed_at": utc_now(),
    }
    log_path = sync_dir / "delivery_failures.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
    return failure_record


def _append_ledger_event(sync_dir: Path, event_id: str) -> None:
    ledger_path = sync_dir / "sent_event_ids.json"
    if ledger_path.exists():
        try:
            known_ids = set(json.loads(ledger_path.read_text(encoding="utf-8")))
        except Exception:
            known_ids = set()
    else:
        known_ids = set()
    known_ids.add(event_id)
    ledger_path.write_text(json.dumps(sorted(known_ids), indent=2), encoding="utf-8")


class DatasourceSyncTransaction(AbstractContextManager["DatasourceSyncTransaction"]):
    """Capture datasource surface before/after and notify fork if effective changes exist."""

    def __init__(
        self,
        repo_root: str | Path,
        origin_session_id: str,
        dedupe_basis: str = "canonical_url",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.origin_session_id = origin_session_id
        self.dedupe_basis = dedupe_basis
        self.started_at = utc_now()
        self.before = capture_snapshot(self.repo_root)
        self.last_payload_path: str | None = None
        self.last_receipt: dict[str, Any] | None = None

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            return None
        after = capture_snapshot(self.repo_root)
        finished_at = utc_now()
        payload = build_payload(
            origin_session_id=self.origin_session_id,
            change_window_start=self.started_at,
            change_window_end=finished_at,
            before=self.before,
            after=after,
            dedupe_basis=self.dedupe_basis,
        )
        if payload is None:
            return None
        self.last_receipt = deliver_payload(payload, repo_root=self.repo_root)
        self.last_payload_path = self.last_receipt.get("payload_file")
        require_success = os.getenv("DATASOURCE_SYNC_REQUIRE_SUCCESS", "1").strip() not in {"0", "false", "False"}
        if require_success and self.last_receipt.get("status") not in {"sent", "replay_blocked"}:
            raise RuntimeError(
                f"Datasource sync delivery failed: status={self.last_receipt.get('status')} "
                f"errors={self.last_receipt.get('errors')}"
            )
        return None
