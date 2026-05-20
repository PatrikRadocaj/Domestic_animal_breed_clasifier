# Animal Breed Detection

Research-grade comparison of two approaches for animal breed recognition:

1. Two-stage pipeline: YOLO11 species detector -> animal crop -> breed classifier.
2. Single-stage pipeline: YOLO11 detector with breed-level classes or hierarchical heads.

The repository is organized for reproducible experiments with Hydra/OmegaConf, versioned
dataset manifests, explicit taxonomy files, and repeatable audit/split/training entrypoints.

## Milestone 1

- Freeze taxonomy v0.
- Build and validate a canonical dataset manifest.
- Audit class balance, hierarchy coverage, and missing annotations.
- Produce reproducible train/validation/test splits.
- Prepare baseline configs for YOLO11 species detection and EfficientNetV2 breed classification.

## Quick Start

```bash
python -m pip install -e ".[dev]"
animal-build-manifest
animal-validate-manifest data.manifest_path=data/manifests/manifest_v0.csv
animal-audit-dataset data.manifest_path=data/manifests/manifest_v0.csv
```

