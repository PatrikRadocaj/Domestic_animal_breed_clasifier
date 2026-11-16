from pathlib import Path

import numpy as np

from animal_breed.mining.unlabeled_animals import (
    _flip_box_to_original,
    _iou_xyxy,
    _limit_images_balanced,
    _nms,
    _parse_shared_model_boxes,
    _predict_species_detections,
    _resolve_output_stem,
)


def test_flip_box_to_original():
    # Original width 200; flipped box [20, 10, 60, 50] maps back to [140, 10, 180, 50]
    box = _flip_box_to_original((20.0, 10.0, 60.0, 50.0), 200)
    assert box == (140.0, 10.0, 180.0, 50.0)


def test_nms_keeps_highest_confidence():
    dets = [
        {"confidence": 0.9, "bbox_xyxy": [10.0, 10.0, 60.0, 60.0]},
        {"confidence": 0.8, "bbox_xyxy": [12.0, 12.0, 58.0, 58.0]},
        {"confidence": 0.7, "bbox_xyxy": [120.0, 120.0, 160.0, 160.0]},
    ]
    out = _nms(dets, iou_thr=0.5)
    assert len(out) == 2
    assert out[0]["confidence"] == 0.9
    assert out[1]["confidence"] == 0.7


def test_iou_is_zero_for_disjoint_boxes():
    iou = _iou_xyxy((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0))
    assert iou == 0.0


def test_resolve_output_stem_uses_suffix_only_when_present():
    assert _resolve_output_stem("unlabeled_candidates", "") == "unlabeled_candidates"
    assert _resolve_output_stem("unlabeled_candidates", "shard_001") == "unlabeled_candidates_shard_001"


class _FakeScalar:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeXYXYRow:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeTensorField:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        value = self._values[idx]
        if isinstance(value, (list, tuple)):
            return _FakeXYXYRow(value)
        return _FakeScalar(float(value))


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensorField(
            [
                [0.0, 0.0, 95.0, 95.0],
                [10.0, 10.0, 40.0, 40.0],
                [50.0, 50.0, 80.0, 80.0],
            ]
        )
        self.cls = _FakeTensorField([0.0, 0.0, 0.0])
        self.conf = _FakeTensorField([0.9, 0.6, 0.8])

    def __len__(self):
        return 3


class _FakeResult:
    def __init__(self):
        self.boxes = _FakeBoxes()
        self.names = {0: "cat"}


class _FakeSharedBoxes:
    def __init__(self):
        self.xyxy = _FakeTensorField(
            [
                [0.0, 0.0, 20.0, 20.0],
                [10.0, 10.0, 30.0, 30.0],
                [20.0, 20.0, 40.0, 40.0],
            ]
        )
        self.cls = _FakeTensorField([0.0, 1.0, 2.0])
        self.conf = _FakeTensorField([0.9, 0.8, 0.7])

    def __len__(self):
        return 3


class _FakeSharedResult:
    def __init__(self):
        self.boxes = _FakeSharedBoxes()
        self.names = {0: "cat", 1: "rabbit", 2: "llama"}


class _FakeModel:
    def predict(self, source, conf, iou, verbose=False):
        return [_FakeResult()]


def test_predict_species_detections_filters_large_boxes_and_applies_topk():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = _predict_species_detections(
        model=_FakeModel(),
        image_bgr=image,
        species_id="cat",
        conf_thr=0.01,
        nms_iou=0.5,
        min_area=50.0,
        min_side=5.0,
        max_area_frac=0.6,
        max_side_frac=0.95,
        max_per_species=1,
    )
    assert len(out) == 1
    assert out[0]["confidence"] == 0.8


def test_parse_shared_model_boxes_respects_allowed_species():
    out = _parse_shared_model_boxes(_FakeSharedResult(), allowed_species={"cat", "rabbit"})
    assert [row["predicted_species"] for row in out] == ["cat", "rabbit"]


def test_limit_images_balanced_round_robins_across_datasets():
    paths = [
        r"C:\repo\data\raw\yolo_sources\dataset_a\train\images\a1.jpg",
        r"C:\repo\data\raw\yolo_sources\dataset_a\train\images\a2.jpg",
        r"C:\repo\data\raw\yolo_sources\dataset_b\train\images\b1.jpg",
        r"C:\repo\data\raw\yolo_sources\dataset_b\train\images\b2.jpg",
        r"C:\repo\data\raw\yolo_sources\dataset_c\train\images\c1.jpg",
    ]
    out = _limit_images_balanced([Path(p) for p in paths], max_images=4, balanced_by_dataset=True)
    assert [p.parts[-4] for p in out] == ["dataset_a", "dataset_b", "dataset_c", "dataset_a"]
