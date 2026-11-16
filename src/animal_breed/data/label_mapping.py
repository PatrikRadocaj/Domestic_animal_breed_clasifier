"""Cross-dataset label normalization and canonical mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml


_SEP_RE = re.compile(r"[\s\-_./\\]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9_ ]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


@dataclass(frozen=True)
class LabelResolution:
    status: str  # resolved | unmapped | manual_review
    canonical_id: str | None
    normalized_label: str
    reason: str


class CrossDatasetLabelMap:
    """Resolve species/breed aliases into canonical IDs across datasets."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.version = str(raw.get("version", "unknown"))
        normalization = raw.get("normalization", {}) or {}
        self.noise_tokens = {str(x).strip().lower() for x in normalization.get("noise_tokens", []) if str(x).strip()}

        self.species_canonical: dict[str, dict[str, Any]] = {}
        self.breed_canonical: dict[str, dict[str, Any]] = {}
        self._duplicate_species_ids: set[str] = set()
        self._duplicate_breed_ids: set[str] = set()
        self.species_alias_index: dict[str, set[str]] = {}
        self.breed_alias_index: dict[str, set[str]] = {}
        self.dataset_species_overrides: dict[str, dict[str, str | None]] = {}
        self.dataset_breed_overrides: dict[str, dict[str, dict[str, str | None]]] = {}

        self._parse_species(raw.get("species", []) or [])
        self._parse_breeds(raw.get("breeds", []) or [])
        self._parse_overrides(raw.get("dataset_overrides", {}) or {})
        self._validate()

    @staticmethod
    def from_file(path: str | Path) -> "CrossDatasetLabelMap":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return CrossDatasetLabelMap(raw)

    def normalize(self, value: str) -> str:
        text = str(value).strip().lower()
        text = _SEP_RE.sub(" ", text)
        text = _NON_ALNUM_RE.sub(" ", text)
        tokens = [tok for tok in text.split() if tok and tok not in self.noise_tokens]
        normalized = "_".join(tokens)
        normalized = _MULTI_UNDERSCORE_RE.sub("_", normalized).strip("_")
        return normalized

    def resolve_species(self, label: str, dataset_id: str) -> LabelResolution:
        normalized = self.normalize(label)
        ds = str(dataset_id).strip()
        ds_override = self.dataset_species_overrides.get(ds, {})
        if normalized in ds_override:
            canonical = ds_override[normalized]
            if canonical is None:
                return LabelResolution("manual_review", None, normalized, "dataset_override_manual_review")
            if canonical not in self.species_canonical:
                return LabelResolution("manual_review", None, normalized, "dataset_override_unknown_canonical")
            return LabelResolution("resolved", canonical, normalized, "dataset_override")

        matches = self.species_alias_index.get(normalized, set())
        if len(matches) > 1:
            return LabelResolution("manual_review", None, normalized, "ambiguous_alias")
        if len(matches) == 1:
            canonical = next(iter(matches))
            return LabelResolution("resolved", canonical, normalized, "alias")
        return LabelResolution("unmapped", None, normalized, "unmapped_label")

    def resolve_breed(self, species_id: str, label: str, dataset_id: str) -> LabelResolution:
        normalized = self.normalize(label)
        species_norm = self.normalize(species_id)
        ds = str(dataset_id).strip()
        ds_breed_overrides = self.dataset_breed_overrides.get(ds, {})

        for key in (species_norm, "__any__"):
            if key in ds_breed_overrides and normalized in ds_breed_overrides[key]:
                canonical = ds_breed_overrides[key][normalized]
                if canonical is None:
                    return LabelResolution("manual_review", None, normalized, "dataset_override_manual_review")
                if canonical not in self.breed_canonical:
                    return LabelResolution("manual_review", None, normalized, "dataset_override_unknown_canonical")
                if not canonical.startswith(f"{species_norm}:"):
                    return LabelResolution("manual_review", None, normalized, "dataset_override_species_mismatch")
                return LabelResolution("resolved", canonical, normalized, "dataset_override")

        key = f"{species_norm}|{normalized}"
        matches = self.breed_alias_index.get(key, set())
        if len(matches) > 1:
            return LabelResolution("manual_review", None, normalized, "ambiguous_alias")
        if len(matches) == 1:
            canonical = next(iter(matches))
            return LabelResolution("resolved", canonical, normalized, "alias")
        return LabelResolution("unmapped", None, normalized, "unmapped_label")

    def _parse_species(self, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            canonical = self.normalize(str(entry.get("canonical_id", "")))
            if not canonical:
                continue
            if canonical in self.species_canonical:
                self._duplicate_species_ids.add(canonical)
            aliases = {self.normalize(canonical), *(self.normalize(str(x)) for x in entry.get("aliases", []) or [])}
            aliases.discard("")
            self.species_canonical[canonical] = {"aliases": sorted(aliases)}
            for alias in aliases:
                self.species_alias_index.setdefault(alias, set()).add(canonical)

    def _parse_breeds(self, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            canonical_raw = str(entry.get("canonical_id", "")).strip().lower()
            if ":" not in canonical_raw:
                continue
            species_id, breed_slug = canonical_raw.split(":", 1)
            species_id = self.normalize(species_id)
            breed_slug = self.normalize(breed_slug)
            canonical = f"{species_id}:{breed_slug}"
            if canonical in self.breed_canonical:
                self._duplicate_breed_ids.add(canonical)
            aliases = {breed_slug, *(self.normalize(str(x)) for x in entry.get("aliases", []) or [])}
            aliases.discard("")
            self.breed_canonical[canonical] = {"species_id": species_id, "aliases": sorted(aliases)}
            for alias in aliases:
                key = f"{species_id}|{alias}"
                self.breed_alias_index.setdefault(key, set()).add(canonical)

    def _parse_overrides(self, entries: dict[str, Any]) -> None:
        for dataset_id, payload in entries.items():
            ds = str(dataset_id).strip()
            if not ds:
                continue
            ds_species: dict[str, str | None] = {}
            for alias, canonical in (payload.get("species", {}) or {}).items():
                alias_norm = self.normalize(str(alias))
                if not alias_norm:
                    continue
                ds_species[alias_norm] = None if canonical is None else self.normalize(str(canonical))
            self.dataset_species_overrides[ds] = ds_species

            ds_breeds: dict[str, dict[str, str | None]] = {}
            raw_breeds = payload.get("breeds", {}) or {}
            if raw_breeds and all(not isinstance(v, dict) for v in raw_breeds.values()):
                raw_breeds = {"__any__": raw_breeds}
            for species_key, alias_map in raw_breeds.items():
                skey = "__any__" if str(species_key).strip() == "__any__" else self.normalize(str(species_key))
                ds_breeds.setdefault(skey, {})
                if not isinstance(alias_map, dict):
                    raise ValueError(
                        f"Dataset override breeds for dataset '{ds}' and species key '{species_key}' "
                        f"must be a mapping, got {type(alias_map)!r}."
                    )
                for alias, canonical in (alias_map or {}).items():
                    alias_norm = self.normalize(str(alias))
                    if not alias_norm:
                        continue
                    ds_breeds[skey][alias_norm] = None if canonical is None else str(canonical).strip().lower()
            self.dataset_breed_overrides[ds] = ds_breeds

    def _validate(self) -> None:
        errors: list[str] = []

        if self._duplicate_species_ids:
            errors.append(f"Duplicate canonical species IDs detected: {sorted(self._duplicate_species_ids)}")
        if self._duplicate_breed_ids:
            errors.append(f"Duplicate canonical breed IDs detected: {sorted(self._duplicate_breed_ids)}")

        ambiguous_species = sorted([alias for alias, ids in self.species_alias_index.items() if len(ids) > 1])
        if ambiguous_species:
            errors.append(f"Ambiguous species aliases map to multiple canonicals: {ambiguous_species[:20]}")

        ambiguous_breeds = sorted([alias for alias, ids in self.breed_alias_index.items() if len(ids) > 1])
        if ambiguous_breeds:
            errors.append(f"Ambiguous breed aliases map to multiple canonicals: {ambiguous_breeds[:20]}")

        for breed_id, payload in self.breed_canonical.items():
            species_id = str(payload["species_id"])
            if not breed_id.startswith(f"{species_id}:"):
                errors.append(f"Breed canonical '{breed_id}' does not match species '{species_id}'.")
            if species_id not in self.species_canonical:
                errors.append(f"Breed canonical '{breed_id}' references missing species '{species_id}'.")

        for ds, overrides in self.dataset_species_overrides.items():
            for alias, canonical in overrides.items():
                if canonical is None:
                    continue
                if canonical not in self.species_canonical:
                    errors.append(
                        f"Dataset override species canonical '{canonical}' not found (dataset '{ds}', alias '{alias}')."
                    )
        for ds, overrides in self.dataset_breed_overrides.items():
            for species_key, alias_map in overrides.items():
                for alias, canonical in alias_map.items():
                    if canonical is None:
                        continue
                    if canonical not in self.breed_canonical:
                        errors.append(
                            f"Dataset override breed canonical '{canonical}' not found (dataset '{ds}', alias '{alias}')."
                        )
                        continue
                    if species_key != "__any__" and not canonical.startswith(f"{species_key}:"):
                        errors.append(
                            f"Dataset override breed '{canonical}' mismatches species '{species_key}' (dataset '{ds}')."
                        )

        if errors:
            raise ValueError("Cross-dataset label map validation failed:\n- " + "\n- ".join(errors))
