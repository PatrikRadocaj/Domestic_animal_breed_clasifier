import pandas as pd

from animal_breed.data.quality.qa import run_quality_checks


def _make_manifest(tmp_path, valid_bbox: bool) -> pd.DataFrame:
    import cv2
    import numpy as np

    p1 = tmp_path / "img1.jpg"
    p2 = tmp_path / "img2.jpg"
    for p in [p1, p2]:
        cv2.imwrite(str(p), np.full((32, 32, 3), 127, dtype=np.uint8))

    x2 = 24 if valid_bbox else 4
    return pd.DataFrame(
        [
            {
                "image_id": "a",
                "path": str(p1),
                "source": "s1",
                "license": "unknown",
                "species_id": "dog",
                "breed_id": "dog:beagle",
                "width": 32,
                "height": 32,
                "bbox_xmin": 8,
                "bbox_ymin": 8,
                "bbox_xmax": x2,
                "bbox_ymax": 24,
                "is_crowd": False,
                "split_group": "g1",
                "duplicate_cluster": "d1",
                "quality_tags": "",
                "split": "train",
            },
            {
                "image_id": "b",
                "path": str(p2),
                "source": "s1",
                "license": "unknown",
                "species_id": "dog",
                "breed_id": "dog:beagle",
                "width": 32,
                "height": 32,
                "bbox_xmin": 8,
                "bbox_ymin": 8,
                "bbox_xmax": 24,
                "bbox_ymax": 24,
                "is_crowd": False,
                "split_group": "g2",
                "duplicate_cluster": "d2",
                "quality_tags": "",
                "split": "val",
            },
        ]
    )


def test_quality_gate_catches_bad_bbox(tmp_path):
    df = _make_manifest(tmp_path, valid_bbox=False)
    _, report = run_quality_checks(
        df,
        taxonomy_path="data/label_maps/taxonomy_v0.yaml",
        config={
            "strictness": "benchmark",
            "dedup": {"enabled": False},
        },
    )
    assert report["valid"] is False
    assert any(issue["code"] == "bbox_bounds" for issue in report["issues"])


def test_quality_gate_passes_clean_sample(tmp_path):
    df = _make_manifest(tmp_path, valid_bbox=True)
    _, report = run_quality_checks(
        df,
        taxonomy_path="data/label_maps/taxonomy_v0.yaml",
        config={
            "strictness": "benchmark",
            "dedup": {"enabled": False},
        },
    )
    assert report["valid"] is True
