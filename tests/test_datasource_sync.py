from __future__ import annotations

import json

from animal_breed.sync import datasource_sync as ds


def test_replay_protection_blocks_known_event(tmp_path):
    repo_root = tmp_path
    sync_dir = repo_root / "results" / "reports" / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    (sync_dir / "sent_event_ids.json").write_text(json.dumps(["evt-1"]), encoding="utf-8")

    payload = {"event_id": "evt-1"}
    receipt = ds.deliver_payload(payload=payload, repo_root=repo_root, max_attempts=1)

    assert receipt["status"] == "replay_blocked"


def test_delivery_failure_records_error_class(monkeypatch, tmp_path):
    repo_root = tmp_path
    payload = {"event_id": "evt-timeout"}

    def _boom(_payload_path):
        raise RuntimeError("timed out while sending")

    monkeypatch.setattr(ds, "_send_via_command", _boom)
    receipt = ds.deliver_payload(payload=payload, repo_root=repo_root, max_attempts=1, base_delay_seconds=0.0)

    assert receipt["status"] == "failed"
    assert any("type=timeout_error" in e for e in receipt["errors"])
    assert (repo_root / "results" / "reports" / "sync" / "delivery_failures.log").exists()
