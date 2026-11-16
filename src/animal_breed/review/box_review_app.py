"""Small local app for reviewing and adjusting detection boxes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd


STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_REVIEW_COLUMNS = {
    "review_status": "unreviewed",
    "review_decision": "",
    "review_notes": "",
    "reviewed_box_xmin": pd.NA,
    "reviewed_box_ymin": pd.NA,
    "reviewed_box_xmax": pd.NA,
    "reviewed_box_ymax": pd.NA,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_candidates_csv() -> Path:
    mining_dir = _repo_root() / "results" / "mining"
    if not mining_dir.exists():
        raise FileNotFoundError("results/mining does not exist yet.")
    preferred = sorted(
        mining_dir.glob("unlabeled_candidates_high_priority*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if preferred:
        return preferred[0]
    fallback = sorted(
        mining_dir.glob("unlabeled_candidates*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if fallback:
        return fallback[0]
    raise FileNotFoundError("No mining candidate CSV files found in results/mining.")


def _default_review_csv(candidates_csv: Path) -> Path:
    return candidates_csv.with_name(f"{candidates_csv.stem}.reviews.csv")


@dataclass
class ReviewStore:
    candidates_csv: Path
    review_csv: Path

    def __post_init__(self) -> None:
        self.candidates_csv = self.candidates_csv.resolve()
        self.review_csv = self.review_csv.resolve()
        self._load()

    def _load(self) -> None:
        if not self.candidates_csv.exists():
            raise FileNotFoundError(f"Candidates CSV not found: {self.candidates_csv}")
        self.df = pd.read_csv(self.candidates_csv)
        if "candidate_id" not in self.df.columns:
            self.df.insert(0, "candidate_id", [f"cand_{i:06d}" for i in range(len(self.df))])
        if self.review_csv.exists():
            review_df = pd.read_csv(self.review_csv)
            self.df = self.df.drop(columns=[c for c in review_df.columns if c in self.df.columns and c != "candidate_id"])
            self.df = self.df.merge(review_df, on="candidate_id", how="left")
        for col, default_value in DEFAULT_REVIEW_COLUMNS.items():
            if col not in self.df.columns:
                self.df[col] = default_value
        self.df["review_status"] = self.df["review_status"].fillna("unreviewed")
        self.df["review_decision"] = self.df["review_decision"].fillna("")
        self.df["review_notes"] = self.df["review_notes"].fillna("")
        for left, right, out in (
            ("reviewed_box_xmin", "bbox_xmin", "display_box_xmin"),
            ("reviewed_box_ymin", "bbox_ymin", "display_box_ymin"),
            ("reviewed_box_xmax", "bbox_xmax", "display_box_xmax"),
            ("reviewed_box_ymax", "bbox_ymax", "display_box_ymax"),
        ):
            self.df[left] = pd.to_numeric(self.df[left], errors="coerce")
            self.df[right] = pd.to_numeric(self.df[right], errors="coerce")
            self.df[out] = self.df[left].where(self.df[left].notna(), self.df[right])

    def records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self.df.to_dict(orient="records"):
            out.append(
                {
                    "candidate_id": row["candidate_id"],
                    "image_path": str(row["image_path"]),
                    "predicted_species": str(row.get("predicted_species", "")),
                    "predicted_breed": str(row.get("predicted_breed", "")),
                    "confidence": float(row.get("confidence", 0.0)),
                    "source_dataset": str(row.get("source_dataset", "")),
                    "existing_labels_in_image": str(row.get("existing_labels_in_image", "")),
                    "conflict_type": str(row.get("conflict_type", "")),
                    "agreement_iou": float(row.get("agreement_iou", 0.0)),
                    "agreement_pass": int(row.get("agreement_pass", 0)),
                    "likely_false_positive": int(row.get("likely_false_positive", 0)),
                    "original_box": {
                        "xmin": float(row["bbox_xmin"]),
                        "ymin": float(row["bbox_ymin"]),
                        "xmax": float(row["bbox_xmax"]),
                        "ymax": float(row["bbox_ymax"]),
                    },
                    "review_box": {
                        "xmin": float(row["display_box_xmin"]),
                        "ymin": float(row["display_box_ymin"]),
                        "xmax": float(row["display_box_xmax"]),
                        "ymax": float(row["display_box_ymax"]),
                    },
                    "review_status": str(row.get("review_status", "unreviewed")),
                    "review_decision": str(row.get("review_decision", "")),
                    "review_notes": str(row.get("review_notes", "")),
                }
            )
        return out

    def summary(self) -> dict[str, Any]:
        status_counts = self.df["review_status"].value_counts().to_dict()
        decision_counts = self.df["review_decision"].replace("", "unset").value_counts().to_dict()
        return {
            "candidates_csv": str(self.candidates_csv),
            "review_csv": str(self.review_csv),
            "total_records": int(len(self.df)),
            "review_status_counts": status_counts,
            "review_decision_counts": decision_counts,
        }

    def save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(payload.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ValueError("candidate_id is required")
        match = self.df.index[self.df["candidate_id"] == candidate_id]
        if len(match) == 0:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        idx = int(match[0])
        review_box = payload.get("review_box", {}) or {}
        for key, column in (
            ("xmin", "reviewed_box_xmin"),
            ("ymin", "reviewed_box_ymin"),
            ("xmax", "reviewed_box_xmax"),
            ("ymax", "reviewed_box_ymax"),
        ):
            value = review_box.get(key)
            self.df.at[idx, column] = float(value) if value is not None else pd.NA
        self.df.at[idx, "display_box_xmin"] = float(review_box.get("xmin", self.df.at[idx, "bbox_xmin"]))
        self.df.at[idx, "display_box_ymin"] = float(review_box.get("ymin", self.df.at[idx, "bbox_ymin"]))
        self.df.at[idx, "display_box_xmax"] = float(review_box.get("xmax", self.df.at[idx, "bbox_xmax"]))
        self.df.at[idx, "display_box_ymax"] = float(review_box.get("ymax", self.df.at[idx, "bbox_ymax"]))
        self.df.at[idx, "review_status"] = str(payload.get("review_status", "reviewed"))
        self.df.at[idx, "review_decision"] = str(payload.get("review_decision", ""))
        self.df.at[idx, "review_notes"] = str(payload.get("review_notes", ""))
        review_only = self.df[["candidate_id", *DEFAULT_REVIEW_COLUMNS.keys()]].copy()
        self.review_csv.parent.mkdir(parents=True, exist_ok=True)
        review_only.to_csv(self.review_csv, index=False)
        return {"ok": True, "candidate_id": candidate_id}


class ReviewRequestHandler(BaseHTTPRequestHandler):
    store: ReviewStore

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
            html = (STATIC_DIR / "index.html").read_bytes()
            self._write_bytes(html, "text/html; charset=utf-8")
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
            self._write_json(
                {
                    "summary": self.store.summary(),
                    "records": self.store.records(),
                }
            )
            return
        if parsed.path == "/api/image":
            query = parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]
            if not raw_path:
                self.send_error(HTTPStatus.BAD_REQUEST, "path is required")
                return
            image_path = Path(raw_path)
            if not image_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "image not found")
                return
            content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
            self._write_bytes(image_path.read_bytes(), content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/review":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
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
    parser = argparse.ArgumentParser(description="Run a local box review app for mining candidates.")
    parser.add_argument("--candidates-csv", type=Path, default=None, help="Candidate CSV to review.")
    parser.add_argument("--review-csv", type=Path, default=None, help="Separate CSV file for saved review edits.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    candidates_csv = args.candidates_csv.resolve() if args.candidates_csv else _default_candidates_csv()
    review_csv = args.review_csv.resolve() if args.review_csv else _default_review_csv(candidates_csv)
    store = ReviewStore(candidates_csv=candidates_csv, review_csv=review_csv)
    ReviewRequestHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), ReviewRequestHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Box review app running at {url}")
    print(f"Candidates: {candidates_csv}")
    print(f"Reviews:    {review_csv}")
    print(f"Open:       {url}?csv={quote(str(candidates_csv))}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
