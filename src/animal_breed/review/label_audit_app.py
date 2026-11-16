"""Local app for reviewing suspicious YOLO dataset labels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd


STATIC_DIR = Path(__file__).resolve().parent / "static_label_audit"
DEFAULT_REVIEW_COLUMNS = {
    "review_status": "unreviewed",
    "review_decision": "",
    "proposed_species": "",
    "review_notes": "",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_index_csv() -> Path:
    return _repo_root() / "outputs" / "suspect_yolo_label_crops" / "index.csv"


def _default_review_csv(index_csv: Path) -> Path:
    return index_csv.with_name("reviews.csv")


@dataclass
class LabelAuditStore:
    index_csv: Path
    review_csv: Path

    def __post_init__(self) -> None:
        self.index_csv = self.index_csv.resolve()
        self.review_csv = self.review_csv.resolve()
        self._load()

    def _load(self) -> None:
        if not self.index_csv.exists():
            raise FileNotFoundError(f"Audit index not found: {self.index_csv}")

        crops_df = pd.read_csv(self.index_csv)
        grouped_rows: list[dict[str, Any]] = []
        for (dataset_id, label_name, label_slug), group in crops_df.groupby(
            ["dataset_id", "label_name", "label_slug"], sort=True
        ):
            crop_paths = [str(x) for x in group["crop_path"].tolist()]
            source_images = sorted({str(x) for x in group["source_image"].tolist()})
            contact_sheet = self.index_csv.parent / "contact_sheets" / str(dataset_id) / f"{label_slug}.jpg"
            grouped_rows.append(
                {
                    "group_id": f"{dataset_id}::{label_slug}",
                    "dataset_id": str(dataset_id),
                    "label_name": str(label_name),
                    "label_slug": str(label_slug),
                    "sample_count": int(len(group)),
                    "contact_sheet_path": str(contact_sheet),
                    "crop_paths": json.dumps(crop_paths),
                    "source_images": json.dumps(source_images),
                }
            )

        self.df = pd.DataFrame(grouped_rows)

        if self.review_csv.exists():
            review_df = pd.read_csv(self.review_csv)
            self.df = self.df.drop(
                columns=[c for c in review_df.columns if c in self.df.columns and c != "group_id"]
            )
            self.df = self.df.merge(review_df, on="group_id", how="left")

        for col, default in DEFAULT_REVIEW_COLUMNS.items():
            if col not in self.df.columns:
                self.df[col] = default
        for col, default in DEFAULT_REVIEW_COLUMNS.items():
            self.df[col] = self.df[col].fillna(default)

    def groups(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self.df.to_dict(orient="records"):
            out.append(
                {
                    "group_id": str(row["group_id"]),
                    "dataset_id": str(row["dataset_id"]),
                    "label_name": str(row["label_name"]),
                    "label_slug": str(row["label_slug"]),
                    "sample_count": int(row["sample_count"]),
                    "contact_sheet_path": str(row["contact_sheet_path"]),
                    "crop_paths": json.loads(str(row["crop_paths"])),
                    "source_images": json.loads(str(row["source_images"])),
                    "review_status": str(row.get("review_status", "unreviewed")),
                    "review_decision": str(row.get("review_decision", "")),
                    "proposed_species": str(row.get("proposed_species", "")),
                    "review_notes": str(row.get("review_notes", "")),
                }
            )
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "index_csv": str(self.index_csv),
            "review_csv": str(self.review_csv),
            "total_groups": int(len(self.df)),
            "status_counts": self.df["review_status"].value_counts().to_dict(),
            "decision_counts": self.df["review_decision"].replace("", "unset").value_counts().to_dict(),
        }

    def save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        group_id = str(payload.get("group_id", "")).strip()
        if not group_id:
            raise ValueError("group_id is required")
        match = self.df.index[self.df["group_id"] == group_id]
        if len(match) == 0:
            raise KeyError(f"Unknown group_id: {group_id}")
        idx = int(match[0])
        self.df.at[idx, "review_status"] = str(payload.get("review_status", "reviewed"))
        self.df.at[idx, "review_decision"] = str(payload.get("review_decision", ""))
        self.df.at[idx, "proposed_species"] = str(payload.get("proposed_species", ""))
        self.df.at[idx, "review_notes"] = str(payload.get("review_notes", ""))
        review_only = self.df[["group_id", *DEFAULT_REVIEW_COLUMNS.keys()]].copy()
        self.review_csv.parent.mkdir(parents=True, exist_ok=True)
        review_only.to_csv(self.review_csv, index=False)
        return {"ok": True, "group_id": group_id}


class LabelAuditRequestHandler(BaseHTTPRequestHandler):
    store: LabelAuditStore

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _write_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._write_bytes((STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/static/"):
            rel = parsed.path.removeprefix("/static/")
            file_path = (STATIC_DIR / rel).resolve()
            if not file_path.exists() or STATIC_DIR.resolve() not in file_path.parents:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            self._write_bytes(file_path.read_bytes(), content_type)
            return
        if parsed.path == "/api/session":
            self._write_json({"summary": self.store.summary(), "groups": self.store.groups()})
            return
        if parsed.path == "/api/file":
            raw_path = parse_qs(parsed.query).get("path", [""])[0]
            if not raw_path:
                self.send_error(HTTPStatus.BAD_REQUEST, "path is required")
                return
            file_path = Path(raw_path)
            if not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "file not found")
                return
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            self._write_bytes(file_path.read_bytes(), content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/review":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        try:
            result = self.store.save_review(payload)
        except KeyError as exc:
            self._write_json({"ok": False, "error": str(exc)}, status=404)
            return
        except ValueError as exc:
            self._write_json({"ok": False, "error": str(exc)}, status=400)
            return
        self._write_json(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local label audit app for suspicious YOLO labels.")
    parser.add_argument("--index-csv", type=Path, default=None, help="Audit crop index CSV.")
    parser.add_argument("--review-csv", type=Path, default=None, help="CSV file for saved label review decisions.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    index_csv = args.index_csv.resolve() if args.index_csv else _default_index_csv()
    review_csv = args.review_csv.resolve() if args.review_csv else _default_review_csv(index_csv)
    store = LabelAuditStore(index_csv=index_csv, review_csv=review_csv)
    LabelAuditRequestHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), LabelAuditRequestHandler)
    print(f"Label audit app running at http://{args.host}:{args.port}/")
    print(f"Index:   {index_csv}")
    print(f"Reviews: {review_csv}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
