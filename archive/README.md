# Finished project preservation

The final July 6 experiment weights are in `models/final/`. These are the actual
saved `best.pt` files, not a fresh selection based on metrics from other runs.
Git LFS is required to download the model binaries and compressed training records.

```powershell
git lfs install
git lfs pull
python tools/restore_finished_project.py --verify-only
python -m pip install -e ".[dev]"
python tools/run_model_inference_app.py --detector-weights models/final/species/best.pt --breed-models-dir models/final/breed
```

`catalog.json` records the original repository-relative path, archived path,
size and SHA-256 for each preserved file. The archive contains final species and
breed dataset manifests, bounding boxes, labels, train/validation/test assignments,
taxonomy, aliases, dataset summaries, experiment metrics, run configurations,
snapshot hashes and saved human review decisions. CSV/JSONL and other training
records are gzip-compressed without changing their content.

Run `python tools/restore_finished_project.py` to unpack the training records into
their original repository-relative locations. Existing files are never overwritten
unless their content already matches. Some recorded image paths are absolute paths
on the original Windows computer; relocate them after downloading the datasets.
Use the `source`, `source_url` and `license` columns of the manifests together with
`data/sources/source_registry_v1.csv`, the archived download configuration and
`tools/download_registry_source.py` to identify and retrieve image sources.
Download credentials may need to be supplied again; no credentials are archived.

Raw image bytes, cached runtime installations, optimizer/last checkpoints, temporary
outputs and local submission ZIP files are intentionally excluded. Source references
preserve dataset provenance but do not guarantee that third-party downloads will
remain available or recreate an identical raw image collection. Keep a separate
raw-image backup if byte-for-byte dataset recovery is required.

Historical seminar slides and documents may describe older runs. For final model
results use the archived July 6 metrics and resolved run configurations.
