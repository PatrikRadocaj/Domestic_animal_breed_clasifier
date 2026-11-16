# Cross-Dataset Label Mapping

Main file:

- `configs/mappings/cross_dataset_label_map.yaml`

This mapping normalizes species and breed labels before taxonomy matching in
`animal-build-manifest` (`src/animal_breed/data/manifests.py`).

## How To Add Aliases

1. Add/locate the canonical entry in `species` or `breeds`.
2. Add new variants in `aliases`.
3. If a label is dataset-specific, add it in `dataset_overrides`.
4. Use `null` in an override to force `manual_review`.

## Reports

During manifest build, unresolved labels are written to:

- `results/reports/unmapped_labels_report.csv`
- `results/reports/manual_review_labels.csv`

## Validation

Run:

```bash
animal-validate-label-mapping
```

This validates mapping integrity and regenerates unresolved-label reports by scanning current raw roots.

