# Dataset Audit Notebook Plan

Use `tools/audit_dataset.py` for the reproducible audit. This notebook placeholder is reserved for
interactive inspection of the generated reports:

- `results/reports/dataset_audit_v0.json`
- `results/reports/species_counts_v0.csv`
- `results/reports/breed_counts_v0.csv`

Recommended visual checks:

- long-tail histogram by breed,
- species-level sample grids,
- suspected duplicate clusters,
- high-risk classes with fewer than 10, 25, and 50 images,
- missing-box percentage by source.

