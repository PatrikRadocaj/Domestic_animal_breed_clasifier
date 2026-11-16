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
animal-data-qa data.manifest_path=data/manifests/manifest_v0.csv
animal-source-provenance-report data.manifest_path=data/manifests/manifest_v0.csv
```

## AMD GPU Runtime

This repo supports an isolated AMD ROCm-on-Windows runtime for GPU training without
changing the main project Python environment.

Project wrappers:

- `tools/amd_gpu_env.ps1`
  - prepares PATH and ROCm-related environment variables for the isolated runtime
- `tools/python_amd.ps1`
  - runs arbitrary Python commands inside the isolated AMD runtime
- `tools/amd_gpu_shell.ps1`
  - opens a PowerShell session with the isolated AMD runtime on PATH
- `tools/train_species_aware_hybrid_amd.ps1`
  - launches the hybrid species-aware trainer on the AMD GPU with `device=0`

Typical runtime location:

- `C:\amd_rocm_py312_real`

The wrappers also fall back to:

- `./.gpu_amd_setup/python312_embed`

Quick checks:

```powershell
.\tools\python_amd.ps1 -m animal_breed.runtime.amd_gpu_probe
.\tools\python_amd.ps1 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\tools\amd_gpu_shell.ps1
```

Hybrid training on the isolated AMD runtime:

```powershell
.\tools\train_species_aware_hybrid_amd.ps1 data.species_aware.hybrid.train.enabled=true
```

The wrappers set GPU-specific PATH entries per command instead of modifying the global
Windows PATH permanently. That keeps the ROCm stack isolated from other projects.

## Artifact Contract

Each training/evaluation run writes:

- `artifacts/<run_id>/checkpoints`
- `artifacts/<run_id>/metrics`
- `artifacts/<run_id>/predictions`
- `artifacts/<run_id>/provenance`
- `artifacts/<run_id>/qa`

Mandatory records include snapshot hashes (`manifest`, `taxonomy`, `source registry`) and resolved config metadata.

## Cross-Fork Datasource Sync Guardrail

Datasource-related changes are monitored and can be sent to fork session:
`019e68b2-7716-7113-8854-5b16fb40ebef`.

- Payload schema: `datasource_sync.v1`
- Trigger surface:
  - `data/sources/**/*`
  - `results/reports/sources/**/*`
  - `configs/data/**/*`
- Delivery command (required):
  - set `DATASOURCE_SYNC_SESSION_SEND_CMD`
  - template must support `{session_id}` and `{payload_file}`
  - example:
    - `opencli codex send --thread-id {session_id} --message-file {payload_file}`

Wrap any datasource-changing command:

```bash
animal-datasource-sync-notify -- animal-source-provenance-report
```

`animal-source-provenance-report` is already wrapped internally and emits notifications automatically when effective datasource diffs are detected.

## Recycled From Original Project

- Original `data/processed` source-provenance artifacts were imported to:
  - `data/sources/original_project_processed/`
- Original dataset registry and mapping files were imported to:
  - `data/sources/original_project_configs/`
- Active source registry used by this repo:
  - `data/sources/source_registry_v1.csv`

When building manifests from a raw tree laid out as
`root/<source_dataset>/species_id/breed_slug/image.ext`,
the builder now resolves `source_url` and `source_type` from
`data.source_registry_path`.

## Experiments Aligned With Project Proposal

### A) Two-stage (YOLO species detector -> EfficientNetV2 hierarchical breed classifiers)

```bash
# 1) Prepare YOLO species dataset and train detector
animal-prepare-yolo model/detector=yolo11_species
animal-train-detector model/detector=yolo11_species

# 2) Train EfficientNetV2 stage-1 species classifier + stage-2 breed-per-species classifiers
animal-train-hierarchical-classifier model/classifier=efficientnetv2_s

# 3) Run two-stage inference (set paths in configs/eval/two_stage_infer.yaml)
animal-infer-two-stage model/pipeline=two_stage eval=two_stage_infer
```

### B) Single-stage (YOLO breed detector/classifier)

```bash
animal-prepare-yolo model/detector=yolo11_breed
animal-train-detector model/detector=yolo11_breed
animal-train-yolo-curriculum
```

## Unified Evaluation + Promotion Gate

```bash
animal-eval eval=benchmark
animal-mine-hard-negatives eval=benchmark
animal-export-models eval=benchmark
```

## Targeted Species Training + Unlabeled-Animal Mining

### Part A: Species-aware dataset filtering and training prep

Build a per-species eligibility table and generate filtered YOLO training sets:

```bash
animal-train-species-aware
```

Outputs:

- `results/reports/species_aware/species_dataset_eligibility_<timestamp>.csv`
- `results/reports/species_aware/species_aware_training_report_<timestamp>.json`
- per-species YOLO datasets under `data/processed/species_aware/<species_id>/`

Enable actual species-model training by overriding:

```bash
animal-train-species-aware data.species_aware.train.enabled=true
```

### Part B: Cross-dataset unlabeled-animal mining

Run trained species detectors on all images and produce pseudo-label candidates:

```bash
animal-mine-unlabeled-animals
```

Required detector location (default):

- `artifacts/species_detectors/<species_id>/weights/best.pt`

Mining outputs:

- `results/mining/unlabeled_candidates.csv`
- `results/mining/unlabeled_candidates_high_priority.csv`
- `results/mining/per_species_summary.json`
- `results/mining/spotcheck_queue.csv`
- `results/mining/visual_samples/*.jpg`

Manual review app:

```bash
animal-review-boxes
```

Notes:

- By default it opens the newest `results/mining/unlabeled_candidates_high_priority*.csv`.
- Reviews are saved separately beside the source CSV as `*.reviews.csv`.
- You can also target a specific file:

```bash
python tools/run_box_review_app.py --candidates-csv results/mining/unlabeled_candidates_shared_multi_full_target_sources_v1.csv
```

Notes:

- Original labels are never modified.
- Candidate labels are kept separate for review/integration.
- Optional agreement filtering (flip-TTA consistency) is controlled via `eval.mining.agreement_check.*`.
- Default `eval.mining.source_mode=yolo_only` mines YOLO-labeled detection datasets, which is the safe default for finding missed background animals.
- Default mining also requires at least one verified existing label in the image and suppresses same-species rediscovery, which keeps the review queue focused on likely missed cross-object candidates.
- When `max_images` is used, the default sampler spreads the budget across source datasets instead of consuming only the first alphabetic dataset.
- Use `eval.mining.source_mode=all_raw` only when you intentionally want to scan the full raw tree, including non-YOLO breed datasets.
- Sharded runs can write separate artifacts with `eval.mining.output_name_suffix=...`; if omitted and `shard_count > 1`, shard-specific suffixes are added automatically.

## Cross-Dataset Label Normalization

A canonical mapping file now normalizes species/breed labels across mixed folders and datasets:

- `configs/mappings/cross_dataset_label_map.yaml`

Normalization is executed in `animal-build-manifest` before taxonomy matching:

1. label cleanup (lowercase, trim, separator normalization, safe noise-token removal)
2. canonical alias resolution
3. dataset-specific overrides
4. taxonomy resolution

Integration point:

- `src/animal_breed/data/manifests.py` via `CrossDatasetLabelMap`

Outputs for review:

- `results/reports/unmapped_labels_report.csv`
- `results/reports/manual_review_labels.csv`

How to add new aliases:

1. Add `canonical_id` entry under `species` or `breeds`.
2. Append new alias variants in the relevant `aliases` list.
3. If a label is dataset-specific/ambiguous, add a rule under `dataset_overrides`.

How to regenerate and validate mapping + unmapped report:

```bash
animal-validate-label-mapping
```

Example label conversions:

- `COW` -> `cattle`
- `Holstein-Friesian breed` -> `cattle:holstein_friesian`
- `peacock` (dataset `farm-animals-v1i-yolo26-roboflow`) -> `peafowl`
- `cpw` (dataset `farm-animals-v1i-yolo26-roboflow`) -> `manual_review`

