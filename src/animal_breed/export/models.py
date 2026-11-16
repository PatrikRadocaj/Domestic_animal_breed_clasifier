"""Model export utility for detector/classifier artifacts."""

from __future__ import annotations

from pathlib import Path

import hydra
import timm
import torch
from omegaconf import DictConfig
from ultralytics import YOLO

from animal_breed.utils.artifacts import build_run_id, create_artifact_bundle, write_json
from animal_breed.utils.logging import setup_logging


def _export_classifier_to_onnx(checkpoint_path: Path, out_path: Path, device: torch.device) -> str:
    payload = torch.load(checkpoint_path, map_location=device)
    label_to_index = payload["label_to_index"]
    labels = [label for label, _ in sorted(label_to_index.items(), key=lambda x: x[1])]
    classifier_cfg = payload.get("config", {}).get("model", {}).get("classifier", {})
    architecture = classifier_cfg.get("architecture", "tf_efficientnetv2_s.in21k_ft_in1k")
    img_size = int(classifier_cfg.get("img_size", 384))

    model = timm.create_model(str(architecture), pretrained=False, num_classes=len(labels))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    )
    return str(out_path.resolve())


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    run_id = build_run_id("export", cfg)
    bundle = create_artifact_bundle(run_id, artifacts_root=str(cfg.paths.artifacts_dir))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device != "cpu" else "cpu")

    exports: dict[str, str] = {}

    detector_weights = Path(str(cfg.eval.single_stage_detector_weights))
    if detector_weights.exists():
        det_model = YOLO(str(detector_weights))
        det_model.export(format="onnx", dynamic=True, simplify=True)
        exports["single_stage_detector_onnx"] = str(detector_weights.resolve())

    two_stage_detector = Path(str(cfg.eval.two_stage_detector_weights))
    if two_stage_detector.exists():
        det_model_2 = YOLO(str(two_stage_detector))
        det_model_2.export(format="onnx", dynamic=True, simplify=True)
        exports["two_stage_detector_onnx"] = str(two_stage_detector.resolve())

    species_ckpt = Path(str(cfg.eval.species_model_path))
    if species_ckpt.exists():
        exports["species_classifier_onnx"] = _export_classifier_to_onnx(
            species_ckpt,
            bundle.checkpoints_dir / "species_classifier.onnx",
            device,
        )

    write_json(bundle.metrics_dir / "export_summary.json", exports)
    logger.info("Model export summary: %s", exports)


if __name__ == "__main__":
    main()
