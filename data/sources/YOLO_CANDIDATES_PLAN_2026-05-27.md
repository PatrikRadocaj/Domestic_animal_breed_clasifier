# YOLO Candidate Acquisition Plan (2026-05-27)

This file complements `data/sources/yolo_candidates_2026-05-27.csv` and focuses on the fastest path to improving single-stage YOLO competitiveness.

## Priority Batch 1 (high impact)

1. `open-images-v7-animals-yolo-kaggle`
2. `oxford-pets-by-breed-detection`
3. `cattle-detection-yprhp-v1`
4. `aerial-sheep-riis-v1`
5. `pig-counting-j2gjo-v1`

## Why this batch

- Adds both **breed-level** (`oxford-pets`) and **species-level** robustness (`cattle/sheep/pig`).
- Balances curated labels with larger-scale domain diversity.
- Keeps all sources in YOLO-ready or easy-to-export formats.

## Download notes

- Kaggle datasets can be downloaded via `kaggle datasets download -d <owner/dataset> --unzip`.
- Roboflow Universe/Public datasets can be exported as YOLO through dataset pages.
- Open Images V7 full-scale training recipe is documented by Ultralytics:
  - `https://docs.ultralytics.com/datasets/detect/open-images-v7/`

## Integration checkpoints before training

- Normalize class names into taxonomy slugs.
- Remove non-target classes for breed experiments.
- Validate bbox quality and class leakage.
- Re-run source provenance report after ingest.
