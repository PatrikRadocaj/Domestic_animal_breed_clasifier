# Canonical Manifest Schema

Each row represents one labeled image instance. For classification-only sources, bbox columns remain empty.
For detection-capable sources, bbox columns store the visible animal box in absolute pixel coordinates.

| column | meaning |
| --- | --- |
| image_id | Stable deterministic image/instance id. |
| path | Local image path. |
| source | Dataset/source identifier. |
| license | Source license or `unknown`. |
| species_id | Canonical taxonomy species id. |
| breed_id | Canonical taxonomy breed id using `<species>:<breed>`. |
| width, height | Image dimensions in pixels. |
| bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax | Optional absolute pixel box. |
| is_crowd | Whether the box represents a group/crowd annotation. |
| split_group | Group id used to prevent leakage across splits. |
| duplicate_cluster | Near-duplicate cluster id once duplicate mining is run. |
| quality_tags | Semicolon-separated tags such as `occluded`, `small`, `low_light`. |
| split | Added by the split script: `train`, `val`, or `test`. |

