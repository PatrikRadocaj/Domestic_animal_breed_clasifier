"""
Creates a clean professor-ready copy of the project.
Includes: all source code, configs, tools, tests, final model checkpoints.
Excludes: datasets, training logs, intermediate checkpoints, caches, dev tooling.
"""

import os, shutil, json, textwrap
from pathlib import Path

SRC  = Path(r"C:\Users\User\Desktop\Strojno ucenje\strojno-ucenje")
DEST = Path(r"C:\Users\User\Desktop\Strojno ucenje\animal-breed-detection-predaja")

if DEST.exists():
    shutil.rmtree(DEST)
DEST.mkdir(parents=True)

SKIP_DIRS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
             ".git", ".gpu_amd_setup", ".venv", "node_modules",
             ".tmp_download_repo", "animal_breed_detection.egg-info"}

def copy_tree(src_dir, dst_dir):
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        rel = Path(root).relative_to(src_dir)
        (Path(dst_dir) / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, Path(dst_dir) / rel / f)

# ── 1. Source code, configs, tools, tests ────────────────────────────────────
print("Copying source code...")
for d in ["src", "configs", "tools", "tests", "notebooks", ".github"]:
    if (SRC / d).exists():
        copy_tree(SRC / d, DEST / d)
        print(f"  {d}/")

for f in ["pyproject.toml", "README.md", ".gitignore"]:
    if (SRC / f).exists():
        shutil.copy2(SRC / f, DEST / f)
        print(f"  {f}")

# ── 2. Small required data files (taxonomy + source registry for CLI) ─────────
print("\nCopying required data config files...")
for rel in ["data/label_maps/taxonomy_v0.yaml",
            "data/sources/source_registry_v1.csv",
            "data/manifests/SCHEMA.md"]:
    src_f = SRC / rel
    dst_f = DEST / rel
    if src_f.exists():
        dst_f.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_f, dst_f)
        print(f"  {rel}")

# ── 3. Final models ────────────────────────────────────────────────────────────
print("\nCopying final models...")

breed_metrics = {
    "cat": 0.5822, "cattle": 0.5984, "chicken": 0.8959, "dog": 0.4855,
    "donkey": 0.9835, "dromedary": 0.7784, "duck": 0.5524, "goat": 0.8610,
    "goose": 0.7841, "horse": 0.5810, "peafowl": 0.9457, "pig": 0.9674,
    "pigeon": 0.8676, "rabbit": 0.9375, "sheep": 0.3532,
}

# 3a. Species detector — put in artifacts/ (webapp auto-discovers this path)
sp_src = SRC / "runs/detect/artifacts/hybrid_shared_detector/hybrid_shared_full18_phr_5more_box25/weights/best.pt"
sp_dst = DEST / "artifacts/shared_species_detector/weights/best.pt"
sp_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(sp_src, sp_dst)
mb = sp_src.stat().st_size / 1024 / 1024
print(f"  artifacts/shared_species_detector/weights/best.pt  ({mb:.1f} MB)")

# 3b. Breed classifiers — put in artifacts/ so webapp auto-discovers
#     Name the bundle "two-stage-breed-models-final" for clarity
BREED_DST_ROOT = DEST / "artifacts/two-stage-breed-models-final/checkpoints/breed_models"

# Main bundle (13 species — the latest full run)
BREED_BUNDLE = SRC / "artifacts/two-stage-breed-models-20260605T164150Z-232659c6/checkpoints/breed_models"
for species in sorted(os.listdir(BREED_BUNDLE)):
    src_pt = BREED_BUNDLE / species / "best.pt"
    if not src_pt.exists():
        continue
    dst_dir = BREED_DST_ROOT / species
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pt, dst_dir / "best.pt")
    mb = src_pt.stat().st_size / 1024 / 1024
    print(f"  breed_models/{species}/best.pt  ({mb:.1f} MB)")

# Cat + cattle from their own earlier run
extras = {
    "cat":    SRC / "artifacts/two-stage-breed-models-20260605T140643Z-eab03834/checkpoints/breed_models/cat/best.pt",
    "cattle": SRC / "artifacts/two-stage-breed-models-20260605T140643Z-eab03834/checkpoints/breed_models/cattle/best.pt",
}
for species, src_pt in extras.items():
    if not src_pt.exists():
        print(f"  ! {species} not found")
        continue
    dst_dir = BREED_DST_ROOT / species
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pt, dst_dir / "best.pt")
    mb = src_pt.stat().st_size / 1024 / 1024
    print(f"  breed_models/{species}/best.pt  ({mb:.1f} MB)")

# Write a model_info.json per species
for species in sorted(os.listdir(BREED_DST_ROOT)):
    info = {
        "architecture": "tf_efficientnetv2_s.in21k_ft_in1k",
        "input_size": "384x384",
        "val_top1": breed_metrics.get(species)
    }
    (BREED_DST_ROOT / species / "model_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")

# 3c. Also keep a clean models/ with human-friendly README
MODELS_REF = DEST / "models"
MODELS_REF.mkdir()
(MODELS_REF / "README.md").write_text(textwrap.dedent("""\
    # Model checkpoints

    The actual checkpoint files live in `artifacts/` (the paths the code expects).
    This folder is just a reference guide.

    ## Species detector
        artifacts/shared_species_detector/weights/best.pt
        - YOLO11s, input 640x640, 18 classes
        - P=0.644  R=0.685  mAP@50=0.679  mAP@50-95=0.365

    ## Breed classifiers (EfficientNetV2-S, input 384x384)
        artifacts/two-stage-breed-models-final/checkpoints/breed_models/<species>/best.pt

    | Species    | val_top1 | Species    | val_top1 |
    |------------|----------|------------|----------|
    | donkey     | 98.4 %   | dromedary  | 77.8 %   |
    | pig        | 96.7 %   | cattle     | 59.8 %   |
    | peafowl    | 94.6 %   | horse      | 58.1 %   |
    | rabbit     | 93.8 %   | cat        | 58.2 %   |
    | chicken    | 89.6 %   | duck       | 55.2 %   |
    | pigeon     | 86.8 %   | dog        | 48.6 %   |
    | goat       | 86.1 %   | sheep      | 35.3 %   |
    | goose      | 78.4 %   |            |          |

    ## Quick start

    ```bash
    pip install -e ".[dev]"

    # Run the web demo (auto-discovers models from artifacts/)
    python tools/run_model_inference_app.py

    # Two-stage inference on a folder of images
    animal-infer-two-stage \\
      eval.source_path=path/to/images \\
      eval.detector_weights_path=artifacts/shared_species_detector/weights/best.pt \\
      eval.breed_models_dir=artifacts/two-stage-breed-models-final/checkpoints/breed_models \\
      eval.output_json_path=results/predictions.json
    ```
"""), encoding="utf-8")

# ── 4. Patch run_model_inference_app.py to pass explicit model paths ──────────
# The webapp auto-discovers from artifacts/ which we've set up, but also
# update the wrapper to make the paths obvious and show explicit usage.
webapp_wrapper = DEST / "tools/run_model_inference_app.py"
webapp_wrapper.write_text(textwrap.dedent("""\
    \"\"\"Launch the local model inference web application.

    Models are auto-discovered from:
      artifacts/shared_species_detector/weights/best.pt
      artifacts/two-stage-breed-models-final/checkpoints/breed_models/

    Or pass explicit overrides:
      python tools/run_model_inference_app.py \\
        --detector-weights artifacts/shared_species_detector/weights/best.pt \\
        --breed-models-dir artifacts/two-stage-breed-models-final/checkpoints/breed_models
    \"\"\"

    from pathlib import Path
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    from animal_breed.webapp.model_inference_app import main

    if __name__ == "__main__":
        main()
"""), encoding="utf-8")

# ── 5. Final count and size ───────────────────────────────────────────────────
print("\n--- Summary ---")
total_files = total_bytes = 0
for root, dirs, files in os.walk(DEST):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in files:
        total_files += 1
        try:
            total_bytes += os.path.getsize(os.path.join(root, f))
        except Exception:
            pass
print(f"  Files:      {total_files}")
print(f"  Total size: {total_bytes / 1024 / 1024:.0f} MB")
print(f"  Dest:       {DEST}")
print("\nDone.")
