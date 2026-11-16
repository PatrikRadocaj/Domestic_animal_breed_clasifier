from pathlib import Path

import pytest
import yaml

from animal_breed.data.label_mapping import CrossDatasetLabelMap


def _write_map(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_alias_resolution_species_and_breed(tmp_path):
    path = _write_map(
        tmp_path,
        {
            "version": "v",
            "normalization": {"noise_tokens": ["class", "breed"]},
            "species": [
                {"canonical_id": "cattle", "aliases": ["Cow", "COWS"]},
            ],
            "breeds": [
                {"canonical_id": "cattle:holstein_friesian", "species_id": "cattle", "aliases": ["Holstein-Friesian"]},
            ],
            "dataset_overrides": {},
        },
    )
    mapping = CrossDatasetLabelMap.from_file(path)

    species = mapping.resolve_species("  COW class ", "dataset-a")
    assert species.status == "resolved"
    assert species.canonical_id == "cattle"

    breed = mapping.resolve_breed("cattle", "Holstein Friesian breed", "dataset-a")
    assert breed.status == "resolved"
    assert breed.canonical_id == "cattle:holstein_friesian"


def test_dataset_specific_override(tmp_path):
    path = _write_map(
        tmp_path,
        {
            "version": "v",
            "species": [{"canonical_id": "peafowl", "aliases": ["peacock"]}],
            "breeds": [],
            "dataset_overrides": {
                "farm": {
                    "species": {
                        "cpw": None,
                        "peacock": "peafowl",
                    }
                }
            },
        },
    )
    mapping = CrossDatasetLabelMap.from_file(path)

    manual = mapping.resolve_species("cpw", "farm")
    assert manual.status == "manual_review"

    resolved = mapping.resolve_species("peacock", "farm")
    assert resolved.status == "resolved"
    assert resolved.canonical_id == "peafowl"


def test_unknown_and_ambiguous_label_behavior(tmp_path):
    payload = {
        "version": "v",
        "species": [
            {"canonical_id": "cat", "aliases": ["feline"]},
            {"canonical_id": "dog", "aliases": ["feline"]},
        ],
        "breeds": [],
        "dataset_overrides": {},
    }
    with pytest.raises(ValueError):
        CrossDatasetLabelMap.from_file(_write_map(tmp_path, payload))

    payload["species"][1]["aliases"] = ["canine"]
    mapping = CrossDatasetLabelMap.from_file(_write_map(tmp_path, payload))
    unknown = mapping.resolve_species("not_in_map", "dataset-a")
    assert unknown.status == "unmapped"
    assert unknown.canonical_id is None

