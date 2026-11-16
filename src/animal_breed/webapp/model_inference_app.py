"""Small local web app for shared-species detection and optional breed inference."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from animal_breed.data.transforms import classification_eval_transforms


STATIC_DIR = Path(__file__).resolve().parent / "static_model_inference"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _format_breed_label(label: str) -> str:
    if ":" in label:
        label = label.split(":", 1)[1]
    return label.replace("_", " ")


def _load_classifier_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, list[str], int]:
    payload = torch.load(path, map_location=device)
    label_to_index = payload["label_to_index"]
    labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    classifier_cfg = payload.get("config", {}).get("model", {}).get("classifier", {})
    architecture = classifier_cfg.get("architecture", "tf_efficientnetv2_s.in21k_ft_in1k")
    img_size = int(classifier_cfg.get("img_size", 384))

    import timm

    model = timm.create_model(str(architecture), pretrained=False, num_classes=len(labels))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, labels, img_size


def _predict_topk(
    model: torch.nn.Module,
    labels: list[str],
    image: np.ndarray,
    img_size: int,
    device: torch.device,
    topk: int,
) -> list[dict[str, Any]]:
    transform = classification_eval_transforms(img_size)
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(topk, probs.shape[0])
        values, indices = torch.topk(probs, k=k)
    out: list[dict[str, Any]] = []
    for value, index in zip(values, indices):
        raw_label = labels[int(index)]
        out.append(
            {
                "label": raw_label,
                "display_label": _format_breed_label(raw_label),
                "confidence": float(value),
            }
        )
    return out


def _crop_from_xyxy(image: np.ndarray, xyxy: list[float], pad_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = [float(value) for value in xyxy]
    box_width = max(1.0, x1 - x0)
    box_height = max(1.0, y1 - y0)
    pad_x = box_width * pad_ratio
    pad_y = box_height * pad_ratio
    x0i = int(max(0, x0 - pad_x))
    y0i = int(max(0, y0 - pad_y))
    x1i = int(min(width, x1 + pad_x))
    y1i = int(min(height, y1 + pad_y))
    return image[y0i:y1i, x0i:x1i]


def _safe_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "upload"


def _draw_label_block(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    color: tuple[int, int, int],
    lines: list[str],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_gap = 6
    padding_x = 5
    padding_y = 5
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_width = max((size[0] for size in text_sizes), default=0)
    text_height = sum(size[1] for size in text_sizes) + line_gap * max(0, len(lines) - 1)
    block_top = max(0, y0 - text_height - (padding_y * 2) - 4)
    block_right = min(canvas.shape[1], x0 + text_width + (padding_x * 2))
    cv2.rectangle(canvas, (x0, block_top), (block_right, y0), color, -1)

    cursor_y = block_top + padding_y
    for line, size in zip(lines, text_sizes):
        baseline_y = cursor_y + size[1]
        cv2.putText(
            canvas,
            line,
            (x0 + padding_x, baseline_y),
            font,
            font_scale,
            (18, 24, 38),
            thickness,
            cv2.LINE_AA,
        )
        cursor_y = baseline_y + line_gap


def _resolve_shared_detector_weights(repo_root: Path) -> Path:
    preferred = [
        repo_root / "artifacts" / "shared_species_detector" / "weights" / "best.pt",
        repo_root / "runs" / "detect" / "artifacts" / "shared_species_detector" / "weights" / "best.pt",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate.resolve()

    fallback_root = repo_root / "runs" / "detect" / "artifacts"
    if fallback_root.exists():
        candidates = [
            path.resolve()
            for path in fallback_root.rglob("best.pt")
            if "shared" in str(path).lower() and "weights" in str(path).lower()
        ]
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return candidates[0]

    raise FileNotFoundError(
        "Shared species detector not found. Expected artifacts/shared_species_detector/weights/best.pt."
    )


def _resolve_breed_models_dir(repo_root: Path) -> Path | None:
    preferred = repo_root / "artifacts" / "latest" / "hierarchical" / "breed"
    if preferred.exists():
        return preferred.resolve()

    candidates = [
        path.resolve()
        for path in repo_root.glob("artifacts/two-stage-breed-models-*/checkpoints/breed_models")
        if any(path.glob("*/best.pt"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


@dataclass(frozen=True)
class AppPaths:
    detector_weights: Path
    breed_models_dir: Path | None
    uploads_dir: Path
    annotated_dir: Path


class ModelInferenceStore:
    def __init__(
        self,
        detector_weights: Path | None = None,
        breed_models_dir: Path | None = None,
        detector_conf: float = 0.25,
        detector_iou: float = 0.50,
        crop_pad_ratio: float = 0.08,
    ) -> None:
        repo_root = _repo_root()
        uploads_dir = repo_root / "outputs" / "model_inference_app" / "uploads"
        annotated_dir = repo_root / "outputs" / "model_inference_app" / "annotated"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)

        self.paths = AppPaths(
            detector_weights=(detector_weights or _resolve_shared_detector_weights(repo_root)).resolve(),
            breed_models_dir=(breed_models_dir or _resolve_breed_models_dir(repo_root)),
            uploads_dir=uploads_dir.resolve(),
            annotated_dir=annotated_dir.resolve(),
        )
        self.detector_conf = float(detector_conf)
        self.detector_iou = float(detector_iou)
        self.crop_pad_ratio = float(crop_pad_ratio)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._detector: YOLO | None = None
        self._detector_names: dict[int, str] = {}
        self._breed_models: dict[str, tuple[torch.nn.Module, list[str], int] | None] = {}
        self._lock = threading.Lock()

    def _load_detector(self) -> YOLO:
        if self._detector is None:
            self._detector = YOLO(str(self.paths.detector_weights))
        return self._detector

    def _species_names(self) -> dict[int, str]:
        if not self._detector_names:
            model = self._load_detector()
            names = getattr(model.model, "names", None) or getattr(model, "names", None) or {}
            self._detector_names = {int(key): str(value) for key, value in dict(names).items()}
        return self._detector_names

    def _breed_model_species(self) -> list[str]:
        root = self.paths.breed_models_dir
        if root is None or not root.exists():
            return []
        species: list[str] = []
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "best.pt").exists():
                species.append(child.name)
        return species

    def _load_breed_bundle(
        self,
        species_id: str,
    ) -> tuple[torch.nn.Module, list[str], int] | None:
        if species_id not in self._breed_models:
            root = self.paths.breed_models_dir
            checkpoint = root / species_id / "best.pt" if root is not None else None
            if checkpoint is None or not checkpoint.exists():
                self._breed_models[species_id] = None
            else:
                self._breed_models[species_id] = _load_classifier_checkpoint(checkpoint, self.device)
        return self._breed_models[species_id]

    def status(self) -> dict[str, Any]:
        species_names = list(self._species_names().values())
        breed_species = self._breed_model_species()
        return {
            "detector_weights": str(self.paths.detector_weights),
            "breed_models_dir": str(self.paths.breed_models_dir) if self.paths.breed_models_dir else "",
            "device": str(self.device),
            "detector_conf": self.detector_conf,
            "detector_iou": self.detector_iou,
            "supported_species": species_names,
            "breed_model_species": breed_species,
            "breed_model_count": len(breed_species),
        }

    def infer_base64(self, filename: str, image_base64: str) -> dict[str, Any]:
        encoded = image_base64.split(",", 1)[-1]
        raw = base64.b64decode(encoded)
        suffix = Path(filename or "upload.jpg").suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".jpg"
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        stem = _safe_stem(Path(filename or "upload").stem)
        upload_path = self.paths.uploads_dir / f"{stem}_{stamp}_{uuid4().hex[:8]}{suffix}"
        upload_path.write_bytes(raw)
        return self.infer_path(upload_path, original_filename=filename or upload_path.name)

    def infer_path(self, image_path: Path, original_filename: str | None = None) -> dict[str, Any]:
        image_path = image_path.resolve()
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        with self._lock:
            detector = self._load_detector()
            results = detector.predict(
                source=str(image_path),
                conf=self.detector_conf,
                iou=self.detector_iou,
                verbose=False,
            )
        result = results[0]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        names = result.names or self._species_names()

        detections: list[dict[str, Any]] = []
        if result.boxes is not None:
            boxes = result.boxes
            for index in range(len(boxes)):
                xyxy = [float(value) for value in boxes.xyxy[index].tolist()]
                species_conf = float(boxes.conf[index].item())
                class_index = int(boxes.cls[index].item())
                species_id = str(names.get(class_index, class_index))
                crop = _crop_from_xyxy(rgb, xyxy, self.crop_pad_ratio)
                breed_prediction: dict[str, Any] | None = None
                breed_bundle = self._load_breed_bundle(species_id)
                if breed_bundle is not None and crop.size > 0:
                    breed_model, breed_labels, breed_img_size = breed_bundle
                    breed_topk = _predict_topk(
                        breed_model,
                        breed_labels,
                        crop,
                        breed_img_size,
                        self.device,
                        topk=3,
                    )
                    if breed_topk:
                        breed_prediction = {
                            "label": breed_topk[0]["label"],
                            "display_label": breed_topk[0]["display_label"],
                            "confidence": breed_topk[0]["confidence"],
                            "topk": breed_topk,
                        }

                detections.append(
                    {
                        "index": index + 1,
                        "bbox_xyxy": xyxy,
                        "species_id": species_id,
                        "species_display": species_id.replace("_", " "),
                        "species_confidence": species_conf,
                        "breed_model_available": breed_bundle is not None,
                        "breed_prediction": breed_prediction,
                    }
                )

        annotated_path = self._write_annotated_image(image_bgr, detections, image_path.stem)
        detected_species = sorted({row["species_id"] for row in detections})
        return {
            "filename": original_filename or image_path.name,
            "input_image_path": str(image_path),
            "annotated_image_path": str(annotated_path),
            "input_image_url": f"/api/image?path={image_path.as_posix()}",
            "annotated_image_url": f"/api/image?path={annotated_path.as_posix()}",
            "image_width": int(image_bgr.shape[1]),
            "image_height": int(image_bgr.shape[0]),
            "detection_count": len(detections),
            "detected_species": detected_species,
            "detections": detections,
        }

    def _write_annotated_image(
        self,
        image_bgr: np.ndarray,
        detections: list[dict[str, Any]],
        image_stem: str,
    ) -> Path:
        canvas = image_bgr.copy()
        for detection in detections:
            x0, y0, x1, y1 = [int(round(value)) for value in detection["bbox_xyxy"]]
            color = (26, 214, 138) if detection["breed_model_available"] else (71, 163, 255)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3)
            label_lines = [f"{detection['species_display']} {detection['species_confidence']:.2f}"]
            breed_prediction = detection.get("breed_prediction")
            if breed_prediction:
                label_lines.append(
                    f"{breed_prediction['display_label']} {breed_prediction['confidence']:.2f}"
                )
                alternatives = breed_prediction.get("topk", [])[1:3]
                if alternatives:
                    alt_text = ", ".join(
                        f"{item['display_label']} {item['confidence']:.2f}" for item in alternatives
                    )
                    label_lines.append(f"Alt: {alt_text}")
            _draw_label_block(canvas, x0, y0, color, label_lines)

        annotated_path = self.paths.annotated_dir / f"{_safe_stem(image_stem)}_{uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(annotated_path), canvas)
        return annotated_path.resolve()

    def allowed_image_path(self, raw_path: str) -> Path:
        path = Path(raw_path).resolve()
        allowed_roots = [self.paths.uploads_dir, self.paths.annotated_dir]
        if any(root == path or root in path.parents for root in allowed_roots):
            return path
        raise PermissionError("Requested image path is outside the app output directories.")


class ModelInferenceRequestHandler(BaseHTTPRequestHandler):
    store: ModelInferenceStore

    def log_message(self, format: str, *args: object) -> None:
        return

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
        if parsed.path == "/api/status":
            self._write_json({"ok": True, "status": self.store.status()})
            return
        if parsed.path == "/api/image":
            raw_path = parse_qs(parsed.query).get("path", [""])[0]
            if not raw_path:
                self.send_error(HTTPStatus.BAD_REQUEST, "path is required")
                return
            try:
                image_path = self.store.allowed_image_path(raw_path)
            except PermissionError as exc:
                self.send_error(HTTPStatus.FORBIDDEN, str(exc))
                return
            if not image_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "image not found")
                return
            content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
            self._write_bytes(image_path.read_bytes(), content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/infer":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            filename = str(payload.get("filename", "upload.jpg"))
            image_base64 = str(payload.get("image_base64", ""))
            if not image_base64:
                raise ValueError("image_base64 is required")
            result = self.store.infer_base64(filename, image_base64)
        except ValueError as exc:
            self._write_json({"ok": False, "error": str(exc)}, status=400)
            return
        except FileNotFoundError as exc:
            self._write_json({"ok": False, "error": str(exc)}, status=500)
            return
        self._write_json({"ok": True, "result": result})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local model inference web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8770, help="Bind port.")
    parser.add_argument("--detector-weights", type=Path, default=None, help="Override shared detector weights path.")
    parser.add_argument("--breed-models-dir", type=Path, default=None, help="Override breed models directory.")
    parser.add_argument("--detector-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--detector-iou", type=float, default=0.50, help="YOLO IoU threshold.")
    parser.add_argument("--crop-pad-ratio", type=float, default=0.08, help="Extra crop context ratio.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = ModelInferenceStore(
        detector_weights=args.detector_weights.resolve() if args.detector_weights else None,
        breed_models_dir=args.breed_models_dir.resolve() if args.breed_models_dir else None,
        detector_conf=args.detector_conf,
        detector_iou=args.detector_iou,
        crop_pad_ratio=args.crop_pad_ratio,
    )
    ModelInferenceRequestHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), ModelInferenceRequestHandler)
    status = store.status()
    print(f"Model inference app running at http://{args.host}:{args.port}/")
    print(f"Detector: {status['detector_weights']}")
    print(f"Breed models: {status['breed_models_dir'] or 'species-only'}")
    print(f"Breed-capable species: {', '.join(status['breed_model_species']) or 'none'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
