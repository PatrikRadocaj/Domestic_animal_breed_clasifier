"""Taxonomy loading, validation, and label normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_label(value: str) -> str:
    """Normalize free-form species/breed labels into stable slugs."""
    normalized = _NORMALIZE_RE.sub("_", value.strip().lower())
    return normalized.strip("_")


@dataclass(frozen=True)
class Breed:
    id: str
    species_id: str
    name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Species:
    id: str
    name: str
    aliases: tuple[str, ...]


class Taxonomy:
    """Validated hierarchy of species and breeds."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.version = str(raw["version"])
        self.species = self._parse_species(raw.get("species", []))
        self.breeds = self._parse_breeds(raw.get("breeds", {}))
        self.species_to_breeds = self._build_species_to_breeds()
        self.species_alias_to_id = self._build_species_aliases()
        self.breed_alias_to_id = self._build_breed_aliases()
        self._validate()

    @staticmethod
    def from_file(path: str | Path) -> "Taxonomy":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return Taxonomy(raw)

    def _parse_species(self, entries: list[dict[str, Any]]) -> dict[str, Species]:
        parsed: dict[str, Species] = {}
        for item in entries:
            species_id = normalize_label(item["id"])
            aliases = tuple(normalize_label(x) for x in item.get("aliases", []))
            parsed[species_id] = Species(
                id=species_id,
                name=str(item["name"]),
                aliases=aliases,
            )
        return parsed

    def _parse_breeds(self, entries: dict[str, list[dict[str, Any]]]) -> dict[str, Breed]:
        parsed: dict[str, Breed] = {}
        for species_key, breed_items in entries.items():
            species_id = normalize_label(species_key)
            for item in breed_items:
                breed_id = str(item["id"]).strip().lower()
                aliases = tuple(normalize_label(x) for x in item.get("aliases", []))
                parsed[breed_id] = Breed(
                    id=breed_id,
                    species_id=species_id,
                    name=str(item["name"]),
                    aliases=aliases,
                )
        return parsed

    def _build_species_to_breeds(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {species_id: [] for species_id in self.species}
        for breed in self.breeds.values():
            mapping.setdefault(breed.species_id, []).append(breed.id)
        return {k: sorted(v) for k, v in mapping.items()}

    def _build_species_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for species in self.species.values():
            aliases[normalize_label(species.id)] = species.id
            aliases[normalize_label(species.name)] = species.id
            for alias in species.aliases:
                aliases[alias] = species.id
        return aliases

    def _build_breed_aliases(self) -> dict[tuple[str, str], str]:
        aliases: dict[tuple[str, str], str] = {}
        for breed in self.breeds.values():
            labels = {breed.id.split(":", 1)[-1], normalize_label(breed.name), *breed.aliases}
            for label in labels:
                aliases[(breed.species_id, label)] = breed.id
        return aliases

    def _validate(self) -> None:
        missing_species = set(self.species_to_breeds) - set(self.species)
        if missing_species:
            raise ValueError(f"Breeds reference unknown species: {sorted(missing_species)}")

        for breed in self.breeds.values():
            if not breed.id.startswith(f"{breed.species_id}:"):
                raise ValueError(
                    f"Breed id '{breed.id}' must use '<species_id>:<breed_slug>' format."
                )

    def resolve_species(self, label: str) -> str | None:
        return self.species_alias_to_id.get(normalize_label(label))

    def resolve_breed(self, species_id: str, label: str) -> str | None:
        normalized_species = normalize_label(species_id)
        normalized_label = normalize_label(label)
        direct_id = f"{normalized_species}:{normalized_label}"
        if direct_id in self.breeds:
            return direct_id
        return self.breed_alias_to_id.get((normalized_species, normalized_label))

    def breed_species(self, breed_id: str) -> str | None:
        breed = self.breeds.get(breed_id)
        return None if breed is None else breed.species_id

    def breed_records(self) -> list[dict[str, str]]:
        return [
            {
                "breed_id": breed.id,
                "species_id": breed.species_id,
                "breed_name": breed.name,
            }
            for breed in sorted(self.breeds.values(), key=lambda item: item.id)
        ]

    def species_records(self) -> list[dict[str, str]]:
        return [
            {"species_id": species.id, "species_name": species.name}
            for species in sorted(self.species.values(), key=lambda item: item.id)
        ]


def load_taxonomy(path: str | Path) -> Taxonomy:
    return Taxonomy.from_file(path)

