"""Run two-stage inference: YOLO species detector -> species-conditioned breed classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from ultralytics import YOLO

from animal_breed.data.transforms import classification_eval_transforms
from animal_breed.utils.artifacts import build_run_id, create_artifact_bundle, write_json
from animal_breed.utils.logging import setup_logging


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def _load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str], int]:
    payload = torch.load(path, map_location=device)
    label_to_index = payload["label_to_index"]
    labels = [label for label, _ in sorted(label_to_index.items(), key=lambda x: x[1])]
    classifier_cfg = payload.get("config", {}).get("model", {}).get("classifier", {})
    architecture = classifier_cfg.get("architecture", "tf_efficientnetv2_s.in21k_ft_in1k")
    img_size = int(classifier_cfg.get("img_size", 384))

    import timm

    model = timm.create_model(str(architecture), pretrained=False, num_classes=len(labels))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, labels, img_size


def _predict_topk(model: torch.nn.Module, labels: list[str], image: np.ndarray, img_size: int, device: torch.device, topk: int = 2) -> list[tuple[str, float]]:
    transform = classification_eval_transforms(img_size)
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(topk, probs.shape[0])
        vals, idxs = torch.topk(probs, k=k)
    return [(labels[int(idx)], float(val)) for val, idx in zip(vals, idxs)]


def _crop_from_xyxy(image: np.ndarray, xyxy: list[float], pad_ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    px = bw * pad_ratio
    py = bh * pad_ratio
    x0i = int(max(0, x0 - px))
    y0i = int(max(0, y0 - py))
    x1i = int(min(w, x1 + px))
    y1i = int(min(h, y1 + py))
    return image[y0i:y1i, x0i:x1i]


def run_two_stage_inference(cfg: DictConfig) -> dict[str, Any]:
    source = Path(str(cfg.eval.source_path))
    out_json = Path(str(cfg.eval.output_json_path))
    detector_weights = Path(str(cfg.eval.detector_weights_path))
    breed_models_dir = Path(str(cfg.eval.breed_models_dir))
    species_model_path = Path(str(cfg.eval.species_model_path))
    detector_conf = float(cfg.model.pipeline.detector_conf_threshold)
    iou_thres = float(cfg.model.pipeline.detector_iou_threshold)
    pad_ratio = float(cfg.model.pipeline.crop_context_ratio)
    arbitration_gap = float(getattr(cfg.model.pipeline, "arbitration_conf_gap", 0.15))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device != "cpu" else "cpu")
    detector = YOLO(str(detector_weights))

    species_model, species_labels, species_img_size = _load_checkpoint(species_model_path, device)

    breed_cache: dict[str, tuple[torch.nn.Module, list[str], int] | None] = {}
    rows: list[dict[str, Any]] = []

    image_paths = _collect_images(source)
    if not image_paths:
        raise FileNotFoundError(f"No images found at {source}")

    # Batched detector inference for speed.
    detector_results = detector.predict(source=[str(p) for p in image_paths], conf=detector_conf, iou=iou_thres, verbose=False)

    for image_path, result in zip(image_paths, detector_results):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        detections: list[dict[str, Any]] = []
        if result.boxes is not None:
            boxes = result.boxes
            names = result.names
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i].item())
                cls_idx = int(boxes.cls[i].item())
                detector_species = str(names.get(cls_idx, str(cls_idx)))

                crop = _crop_from_xyxy(rgb, xyxy, pad_ratio)
                if crop.size == 0:
                    continue

                species_top2 = _predict_topk(species_model, species_labels, crop, species_img_size, device, topk=2)
                species_pred, species_conf = species_top2[0]
                arbitration_triggered = False
                if len(species_top2) > 1 and species_pred != detector_species:
                    if (species_conf - species_top2[1][1]) < arbitration_gap:
                        arbitration_triggered = True
                        species_pred = detector_species

                if species_pred not in breed_cache:
                    breed_ckpt = breed_models_dir / species_pred / "best.pt"
                    breed_cache[species_pred] = _load_checkpoint(breed_ckpt, device) if breed_ckpt.exists() else None

                breed_pred = ""
                breed_conf = None
                breed_bundle = breed_cache[species_pred]
                if breed_bundle is not None:
                    breed_model, breed_labels, breed_img_size = breed_bundle
                    breed_top = _predict_topk(breed_model, breed_labels, crop, breed_img_size, device, topk=1)
                    if breed_top:
                        breed_pred, breed_conf = breed_top[0]

                detections.append(
                    {
                        "bbox_xyxy": [float(x) for x in xyxy],
                        "detector_species": detector_species,
                        "detector_conf": conf,
                        "species_pred": species_pred,
                        "species_conf": species_conf,
                        "species_top2": [{"label": a, "conf": b} for a, b in species_top2],
                        "arbitration_triggered": arbitration_triggered,
                        "breed_pred": breed_pred,
                        "breed_conf": breed_conf,
                    }
                )

        rows.append({"image_path": str(image_path.resolve()), "detections": detections})

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    run_id = build_run_id("infer2stage", cfg)
    bundle = create_artifact_bundle(run_id, artifacts_root=str(cfg.paths.artifacts_dir))
    write_json(bundle.predictions_dir / "two_stage_predictions.json", {"rows": rows})

    return {"images": len(rows), "output": str(out_json.resolve()), "artifact_root": str(bundle.root.resolve())}


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    stats = run_two_stage_inference(cfg)
    logger.info("Two-stage inference done: %s", stats)


if __name__ == "__main__":
    main()
