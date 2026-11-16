from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_breed.data.label_mapping import CrossDatasetLabelMap

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
SPLIT_TOKENS = {"train", "valid", "val", "test", "images", "labels", "metadata", "annotations"}
PLACEHOLDER = {
    "", "unknown", "unlabeled", "unlabelled", "undefined", "none", "null", "na", "n_a", "not_applicable",
    "other", "others", "mixed", "hybrid", "unidentified", "background", "object", "animal", "animals",
    "class", "classes", "label", "labels", "breed", "breeds", "species", "dataset", "image", "images",
    "empty", "mixed_breed", "unidentified_mixed",
}
PLACEHOLDER_TOKENS = {
    "unknown", "unlabeled", "unlabelled", "undefined", "unidentified", "mixed", "other", "others", "empty"
}
INVALID_TOKENS = {
    "jpg", "jpeg", "png", "webp", "txt", "csv", "json", "yaml", "xml", "parquet", "download", "downloads",
    "roboflow", "kaggle", "wikimedia", "inaturalist", "img", "image", "file", "folder", "source", "valid",
    "back_view", "side_view",
}
SPECIES_GROUPS = {
    "alpaca", "bactrian_camel", "cat", "cattle", "chicken", "dog", "donkey", "dromedary", "duck", "goat",
    "goose", "horse", "llama", "peafowl", "pig", "pigeon", "rabbit", "sheep", "turkey", "camel", "water_buffalo",
}
NO_STAGE2_DEFAULT = {"alpaca", "llama", "dromedary", "bactrian_camel", "camel", "water_buffalo"}

BIOLOGICAL_SPECIES = {
    "dog": "Canis lupus familiaris",
    "cat": "Felis catus",
    "cattle": "Bos taurus / Bos indicus domestic cattle complex",
    "horse": "Equus ferus caballus",
    "sheep": "Ovis aries",
    "goat": "Capra hircus",
    "pig": "Sus scrofa domesticus",
    "rabbit": "Oryctolagus cuniculus domesticus",
    "chicken": "Gallus gallus domesticus",
    "turkey": "Meleagris gallopavo domesticus",
    "duck": "Domestic duck lineages: Anas platyrhynchos domesticus and Cairina moschata domesticus",
    "goose": "Domestic goose lineages: Anser anser domesticus and Anser cygnoides domesticus",
    "donkey": "Equus asinus",
    "peafowl": "Pavo cristatus / Pavo muticus / Afropavo congensis group",
    "pigeon": "Columba livia domestica and domestic pigeon breeds",
    "llama": "Lama glama",
    "alpaca": "Vicugna pacos",
    "dromedary": "Camelus dromedarius",
    "bactrian_camel": "Camelus bactrianus / Camelus ferus where source indicates wild Bactrian",
    "camel": "Camelus spp. generic",
    "water_buffalo": "Bubalus bubalis",
}

SPECIES_WITHIN_GROUP = {
    "peafowl": {
        "indian_peafowl": "Pavo cristatus",
        "blue_peafowl": "Pavo cristatus",
        "green_peafowl": "Pavo muticus",
        "java_green_peafowl": "Pavo muticus",
        "congo_peafowl": "Afropavo congensis",
    },
    "duck": {
        "mallard": "Anas platyrhynchos",
        "muscovy": "Cairina moschata",
        "mandarin_duck": "Aix galericulata",
        "wood_duck": "Aix sponsa",
    },
    "goose": {
        "greylag_goose": "Anser anser",
        "swan_goose": "Anser cygnoides",
        "canada_goose": "Branta canadensis",
        "egyptian_goose": "Alopochen aegyptiaca",
    },
    "camel": {
        "dromedary": "Camelus dromedarius",
        "arabian_camel": "Camelus dromedarius",
        "bactrian_camel": "Camelus bactrianus",
        "wild_bactrian_camel": "Camelus ferus",
    },
}

MORPHS = {
    "peafowl": {"white", "pied", "black_shoulder", "spalding", "opal", "bronze", "purple", "cameo", "peach", "silver_pied"},
    "cat": {"tabby", "calico", "tortoiseshell", "torbie", "tuxedo", "dilute_calico", "dilute_tortoiseshell", "black", "white", "ginger", "orange", "silver", "tiger"},
    "rabbit": {"albino", "broken", "solid", "agouti", "otter", "self", "white", "black"},
    "dromedary": {"black", "brown", "white", "light", "dark"},
    "bactrian_camel": {"white", "brown", "black", "wild"},
    "duck": {"white", "black", "blue", "pied", "crested"},
    "goose": {"white", "grey", "gray", "pied", "buff"},
    "pigeon": {"white", "pied", "checker", "bar", "red", "blue_bar"},
}

SYNONYM_CANONICAL = {
    "golden_retriever": {"golden", "golden_retriever", "golden_retrievers"},
    "labrador_retriever": {"labrador", "lab", "labrador_retriever"},
    "german_shepherd": {"german_shepherd", "german_shepherd_dog", "alsatian"},
    "british_shorthair": {"british_short", "british_shorthair", "british"},
    "american_shorthair": {"american_short", "american_shorthair"},
    "friesian": {"friesian", "frisian"},
    "bactrian_camel": {"bactrian", "bactrian_camel", "two_hump_camel", "two_humped_camel"},
    "dromedary": {"arabian_camel", "dromedary", "single_hump_camel", "one_hump_camel"},
    "indian_peafowl": {"indian_peafowl", "blue_peafowl", "peacock_blue", "pavo_cristatus"},
    "green_peafowl": {"green_peafowl", "java_green_peafowl", "pavo_muticus"},
    "congo_peafowl": {"congo_peafowl", "afropavo_congensis"},
}


def slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"n\d{8}_", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def is_csv_label_column(column_name: Any) -> bool:
    """Return True only for CSV columns likely to contain semantic animal labels."""
    col = slug(column_name)
    if not col:
        return False
    if col.startswith("unnamed"):
        return False
    if col in {
        "id", "index", "idx", "row", "rowid", "row_id", "image_id", "annotation_id", "object_id",
        "url", "uri", "path", "file", "filename", "file_name", "image", "image_name", "image_path",
        "photo", "photos", "med_photo", "med_photos", "thumbnail", "source", "dataset",
    }:
        return False
    if col.endswith("_id") and col not in {"species_id", "breed_id", "class_id", "label_id"}:
        return False
    if any(tok in col for tok in ["bbox", "x_min", "y_min", "xmax", "ymax", "width", "height"]):
        return False
    if any(tok in col for tok in ["breed", "species", "class", "label"]):
        return True
    # Allow "name" only when it is not an image/file/source name column.
    return col in {"name", "category_name", "class_name", "label_name", "breed_name", "species_name"}


def title_label(canonical: str) -> str:
    return canonical.replace(":", ":").replace("_", " ").title().replace(":", ":")


@dataclass
class RawLabel:
    original_label: str
    source_dataset: str
    source_kind: str
    count: int = 1
    provenance: str = ""


class AuditEngine:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.label_map = CrossDatasetLabelMap.from_file(repo_root / "configs" / "mappings" / "cross_dataset_label_map.yaml")
        self.species_alias: dict[str, str] = {}
        self.breed_alias: dict[tuple[str, str], str] = {}
        for canonical, payload in self.label_map.species_canonical.items():
            self.species_alias[slug(canonical)] = canonical
            for a in payload.get("aliases", []):
                self.species_alias[slug(a)] = canonical
        for canonical, payload in self.label_map.breed_canonical.items():
            species = payload.get("species_id", str(canonical).split(":", 1)[0])
            self.breed_alias[(species, slug(str(canonical).split(":", 1)[-1]))] = canonical
            for a in payload.get("aliases", []):
                self.breed_alias[(species, slug(a))] = canonical

    def infer_species_group(self, original: str, source_dataset: str) -> str:
        ds = slug(source_dataset)
        raw = slug(original)
        combined = f"{ds}_{raw}"
        # Dataset-name hints first because many non-YOLO folder labels are just breed names.
        hints = [
            ("bactrian_camel", ["bactrian"]), ("dromedary", ["dromedary", "arabian_camel", "camel_classification"]),
            ("water_buffalo", ["water_buffalo", "buffalo"]), ("cattle", ["cattle", "cow", "bovine", "bull"]),
            ("horse", ["horse"]), ("sheep", ["sheep"]), ("goat", ["goat"]), ("pig", ["pig", "swine"]),
            ("rabbit", ["rabbit", "bunny"]), ("chicken", ["chicken", "galliform", "hen", "rooster"]),
            ("turkey", ["turkey"]), ("duck", ["duck"]), ("goose", ["goose", "geese"]),
            ("donkey", ["donkey", "burro"]), ("peafowl", ["peafowl", "peacock"]), ("pigeon", ["pigeon"]),
            ("llama", ["llama"]), ("alpaca", ["alpaca"]), ("cat", ["cat", "feline"]), ("dog", ["dog", "canine"]),
        ]
        for species, tokens in hints:
            if any(tok in combined for tok in tokens):
                return species
        return self.species_alias.get(raw, "")

    def classify(self, item: RawLabel) -> dict[str, Any]:
        raw = item.original_label
        raw_slug = slug(raw)
        source_dataset = item.source_dataset
        species_group = self.infer_species_group(raw, source_dataset)
        biological_species = BIOLOGICAL_SPECIES.get(species_group, "")
        reason = ""
        confidence = "medium"
        canonical_label = ""
        label_type = "review"
        stage2_action = "review"
        direct_action = "review"
        notes = ""

        species_res = self.label_map.resolve_species(raw, source_dataset)
        if species_res.status == "resolved":
            species_group = species_res.canonical_id or species_group
            biological_species = BIOLOGICAL_SPECIES.get(species_group, biological_species)
            canonical_label = species_group
            label_type = "species" if species_group in SPECIES_GROUPS else "fallback_species"
            stage2_action = "exclude"
            direct_action = "keep_as_species_fallback"
            reason = "valid species-level detector label; not a finer Stage-2 class by itself"
            confidence = "high"
            if species_group in {"camel"}:
                label_type = "fallback_species"
                notes = "Generic camel should be split to dromedary/bactrian when supported; otherwise fallback only."
        elif raw_slug.startswith("excluded_") or raw_slug in PLACEHOLDER or any(tok in PLACEHOLDER_TOKENS for tok in raw_slug.split("_")):
            label_type = "exclude"
            stage2_action = "exclude"
            direct_action = "exclude"
            canonical_label = ""
            reason = "unknown/unlabeled/generic placeholder; not visually trainable"
            confidence = "high"
        elif raw_slug in INVALID_TOKENS or re.fullmatch(r"\d+", raw_slug or "") or len(raw_slug) <= 1:
            label_type = "exclude"
            stage2_action = "exclude"
            direct_action = "exclude"
            reason = "source artifact, numeric id, filename token, or context token"
            confidence = "high"
        else:
            if not species_group:
                # Try species-within-group labels before giving up.
                for group, members in SPECIES_WITHIN_GROUP.items():
                    if raw_slug in members:
                        species_group = group
                        break
            # Species-within-group, e.g. Indian peafowl under peafowl.
            if species_group in SPECIES_WITHIN_GROUP and raw_slug in SPECIES_WITHIN_GROUP[species_group]:
                canonical_label = f"{species_group}:{raw_slug}"
                biological_species = SPECIES_WITHIN_GROUP[species_group][raw_slug]
                label_type = "species_within_group"
                stage2_action = "keep"
                direct_action = "keep_as_specific_label"
                reason = "biological species/lineage inside broader project detector group; visually meaningful"
                confidence = "high"
            elif species_group and raw_slug in MORPHS.get(species_group, set()):
                canonical_label = f"{species_group}:{raw_slug}"
                label_type = "morph"
                stage2_action = "keep"
                direct_action = "keep_as_specific_label"
                reason = "visually distinct morph/color phenotype accepted as useful class"
                confidence = "medium"
            elif species_group:
                # High-confidence breed alias through existing map.
                breed_res = self.label_map.resolve_breed(species_group, raw, source_dataset)
                if breed_res.status == "resolved" and breed_res.canonical_id:
                    canonical_label = breed_res.canonical_id
                    label_type = "breed"
                    stage2_action = "keep"
                    direct_action = "keep_as_specific_label"
                    reason = f"formal/known breed resolved by canonical alias map ({breed_res.reason})"
                    confidence = "high"
                elif raw_slug == species_group or raw_slug in {slug(species_group), slug(species_group + 's')}:
                    canonical_label = species_group
                    label_type = "fallback_species"
                    stage2_action = "exclude"
                    direct_action = "keep_as_species_fallback"
                    reason = "generic species label incorrectly usable as breed; fallback only for direct YOLO"
                    confidence = "high"
                elif any(raw_slug in vals for vals in SYNONYM_CANONICAL.values()):
                    canon_slug = next(k for k, vals in SYNONYM_CANONICAL.items() if raw_slug in vals)
                    canonical_label = f"{species_group}:{canon_slug}" if species_group not in {canon_slug, "camel"} else canon_slug
                    if canon_slug in {"dromedary", "bactrian_camel", "indian_peafowl", "green_peafowl", "congo_peafowl"}:
                        label_type = "species_within_group" if species_group in {"camel", "peafowl"} else "species"
                    else:
                        label_type = "synonym/spelling variant of another label"
                    stage2_action = "merge"
                    direct_action = "merge"
                    reason = "high-confidence spelling/synonym normalization"
                    confidence = "high"
                else:
                    canonical_label = f"{species_group}:{raw_slug}"
                    label_type = "review"
                    stage2_action = "review"
                    direct_action = "review"
                    reason = "not in canonical alias map; may be real breed/subspecies/morph but needs manual review"
                    confidence = "low"
            else:
                canonical_label = raw_slug
                label_type = "review"
                stage2_action = "review"
                direct_action = "review"
                reason = "could not infer project species group from label or source dataset"
                confidence = "low"

        # Conservative rule: species with no meaningful split do not create Stage-2 classes unless morph/species-within-group is explicit.
        if species_group in NO_STAGE2_DEFAULT and label_type == "breed":
            stage2_action = "review"
            direct_action = "review"
            reason = "species/group normally has no robust formal breed split in this project; requires manual validation"
            confidence = "medium"
        if label_type in {"species", "fallback_species"} and species_group in NO_STAGE2_DEFAULT:
            stage2_action = "exclude"
            direct_action = "keep_as_species_fallback"

        return {
            "original_label": raw,
            "source_dataset": source_dataset,
            "source_kind": item.source_kind,
            "label_count_observed": item.count,
            "provenance": item.provenance,
            "project_stage1_species_group": species_group,
            "biological_species_if_known": biological_species,
            "proposed_canonical_label": canonical_label,
            "canonical_label_type": label_type,
            "stage2_action": stage2_action,
            "direct_yolo_action": direct_action,
            "reason": reason,
            "confidence": confidence,
            "notes": notes,
        }


def add_counter(counter: dict[tuple[str, str, str], Counter], source: str, kind: str, label: str, provenance: str = "") -> None:
    label = str(label or "").strip()
    if not label:
        return
    counter[(source, kind, provenance)][label] += 1


def extract_yolo_labels(repo: Path, counter: dict[tuple[str, str, str], Counter]) -> None:
    root = repo / "data" / "raw" / "yolo_sources"
    for ds in sorted(p for p in root.iterdir() if p.is_dir()):
        for yaml_name in ["data.yaml", "dataset.yaml"]:
            ypath = ds / yaml_name
            if ypath.exists():
                try:
                    data = yaml.safe_load(ypath.read_text(encoding="utf-8", errors="ignore")) or {}
                    names = data.get("names", [])
                    if isinstance(names, dict):
                        names = [names[k] for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else str(x))]
                    for name in names or []:
                        add_counter(counter, ds.name, "yolo_data_yaml_class", str(name), str(ypath.relative_to(repo)))
                except Exception as exc:
                    add_counter(counter, ds.name, "unreadable_yaml", f"YAML_ERROR:{exc}", str(ypath.relative_to(repo)))
        # PHR metadata preserves breed/subspecies provenance.
        meta_root = ds / "metadata"
        if meta_root.exists():
            for mp in meta_root.rglob("*.json"):
                try:
                    payload = json.loads(mp.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    continue
                for obj in payload.get("objects", []) or []:
                    for key in ["source_label_name", "raw_label", "subspecies", "species", "class_name"]:
                        val = obj.get(key)
                        val_slug = slug(val)
                        if key in {"source_label_name", "raw_label", "subspecies"} and (
                            val_slug in PLACEHOLDER or "unknown" in val_slug or "unlabeled" in val_slug
                        ):
                            continue
                        if val:
                            add_counter(counter, ds.name, f"metadata_object_{key}", str(val), str(mp.parent.relative_to(ds)))


def extract_non_yolo_labels(repo: Path, counter: dict[tuple[str, str, str], Counter]) -> None:
    root = repo / "data" / "raw" / "non_yolo_sources"
    for ds in sorted(p for p in root.iterdir() if p.is_dir()):
        # Folder labels: every folder containing images contributes its non-split path tokens as raw labels.
        for folder in ds.rglob("*"):
            if not folder.is_dir():
                continue
            if any(slug(part).startswith("excluded") for part in folder.relative_to(ds).parts):
                continue
            try:
                has_images = any(child.is_file() and child.suffix.lower() in IMAGE_EXTS for child in folder.iterdir())
            except Exception:
                continue
            if not has_images:
                continue
            rel_parts = [part for part in folder.relative_to(ds).parts if slug(part) not in SPLIT_TOKENS]
            if not rel_parts:
                continue
            # Leaf is usually the label; include parent path in provenance.
            add_counter(counter, ds.name, "non_yolo_folder_label", rel_parts[-1], str(folder.relative_to(ds)))
        # CSV labels where available.
        for cp in ds.rglob("*.csv"):
            try:
                df = pd.read_csv(cp, nrows=50000)
            except Exception:
                continue
            label_cols = [c for c in df.columns if is_csv_label_column(c)]
            for col in label_cols:
                vals = df[col].dropna().astype(str)
                for val, cnt in vals.value_counts().head(500).items():
                    key = (ds.name, f"csv_column_{col}", str(cp.relative_to(ds)))
                    counter[key][str(val)] += int(cnt)


def extract_processed_manifest_labels(
    repo: Path,
    counter: dict[tuple[str, str, str], Counter],
    processed_run_name: str = "",
) -> None:
    # Processed manifests are not raw sources, but they preserve provenance labels from prior rebuilds.
    runs = repo / "data" / "processed" / "breed_runs"
    if not runs.exists():
        return
    if processed_run_name:
        run_dirs = [runs / processed_run_name]
    else:
        run_dirs = sorted(p for p in runs.iterdir() if p.is_dir())
    csv_paths: list[Path] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            continue
        csv_paths.extend(sorted(run_dir.glob("label_review_report.csv")))
        csv_paths.extend(sorted(run_dir.glob("manifest*.csv")))
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {"source", "source_dataset", "raw_label", "species_id", "breed_id"})
        except Exception:
            continue
        if "raw_label" not in df.columns:
            continue
        source_col = "source" if "source" in df.columns else "source_dataset" if "source_dataset" in df.columns else None
        for row in df.dropna(subset=["raw_label"]).to_dict(orient="records"):
            src = str(row.get(source_col, "processed_manifest")) if source_col else "processed_manifest"
            label = str(row.get("raw_label", ""))
            key = (src, "processed_manifest_raw_label", str(csv_path.relative_to(repo)))
            counter[key][label] += 1


def build_items(counter: dict[tuple[str, str, str], Counter]) -> list[RawLabel]:
    items: list[RawLabel] = []
    for (source, kind, prov), c in counter.items():
        for label, count in c.items():
            items.append(RawLabel(original_label=label, source_dataset=source, source_kind=kind, count=int(count), provenance=prov))
    return items


def write_summary(rows: list[dict[str, Any]], out_dir: Path, active_stage1: list[str]) -> None:
    df = pd.DataFrame(rows)
    summaries: dict[str, Any] = {}
    summaries["final_stage1_species_groups"] = sorted(active_stage1 or sorted(set(df["project_stage1_species_group"].dropna()) - {""}))

    keep_stage2 = df[df["stage2_action"].isin(["keep", "merge"])]
    class_counts = keep_stage2.groupby("project_stage1_species_group")["proposed_canonical_label"].nunique().sort_index()
    stage2_trainable = {
        species: sorted(keep_stage2.loc[keep_stage2["project_stage1_species_group"].eq(species), "proposed_canonical_label"].dropna().unique().tolist())
        for species, count in class_counts.items()
        if int(count) >= 2
    }
    summaries["final_stage2_trainable_classes_per_species_group"] = stage2_trainable
    skipped = {}
    for sp in sorted(set(df["project_stage1_species_group"].dropna()) - {""}):
        n = int(class_counts.get(sp, 0))
        if n < 2:
            skipped[sp] = "fewer than two accepted finer visually meaningful labels after strict audit"
    summaries["species_groups_skipped_for_stage2"] = skipped

    direct = df[df["direct_yolo_action"].isin(["keep_as_specific_label", "keep_as_species_fallback", "merge"])]
    direct_classes = sorted(set(direct["proposed_canonical_label"].dropna()) | {sp for sp in skipped if sp})
    summaries["final_direct_breed_yolo_classes"] = direct_classes

    summaries["morphs_accepted_as_trainable_visual_categories"] = (
        df[df["canonical_label_type"].eq("morph")][["project_stage1_species_group", "proposed_canonical_label", "reason"]]
        .drop_duplicates()
        .sort_values(["project_stage1_species_group", "proposed_canonical_label"])
        .to_dict(orient="records")
    )
    summaries["biological_species_grouped_under_project_species_label"] = {
        group: members for group, members in SPECIES_WITHIN_GROUP.items() if group in summaries["final_stage1_species_groups"] or group in set(df["project_stage1_species_group"])
    }
    summaries["synonym_merge_map"] = (
        df[df["stage2_action"].eq("merge")][["original_label", "source_dataset", "proposed_canonical_label", "reason"]]
        .drop_duplicates()
        .sort_values(["proposed_canonical_label", "original_label"])
        .to_dict(orient="records")
    )
    summaries["excluded_labels_with_reasons"] = (
        df[df["stage2_action"].eq("exclude")][["original_label", "source_dataset", "project_stage1_species_group", "reason", "confidence"]]
        .drop_duplicates()
        .sort_values(["reason", "source_dataset", "original_label"])
        .to_dict(orient="records")
    )
    summaries["uncertain_labels_requiring_manual_review"] = (
        df[df["stage2_action"].eq("review")][["original_label", "source_dataset", "project_stage1_species_group", "proposed_canonical_label", "reason", "confidence"]]
        .drop_duplicates()
        .sort_values(["project_stage1_species_group", "source_dataset", "original_label"])
        .to_dict(orient="records")
    )
    summaries["counts"] = {
        "raw_label_rows": int(len(df)),
        "unique_original_labels": int(df["original_label"].nunique()),
        "source_datasets": int(df["source_dataset"].nunique()),
        "stage2_keep_or_merge_rows": int(len(keep_stage2)),
        "stage2_trainable_species_groups": int(len(stage2_trainable)),
        "direct_yolo_candidate_classes": int(len(direct_classes)),
        "review_rows": int((df["stage2_action"] == "review").sum()),
        "exclude_rows": int((df["stage2_action"] == "exclude").sum()),
    }
    (out_dir / "label_taxonomy_audit_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown compact summary.
    md = []
    md.append("# Label Taxonomy QA Audit\n")
    md.append("## Counts\n")
    for k, v in summaries["counts"].items():
        md.append(f"- `{k}`: `{v}`")
    md.append("\n## Final Stage 1 Species Groups\n")
    md.append(", ".join(f"`{x}`" for x in summaries["final_stage1_species_groups"]))
    md.append("\n\n## Stage 2 Trainable Classes Per Species/Group\n")
    for sp, classes in stage2_trainable.items():
        md.append(f"- `{sp}`: {len(classes)} classes")
    md.append("\n## Species/Groups Skipped For Stage 2\n")
    for sp, reason in skipped.items():
        md.append(f"- `{sp}`: {reason}")
    md.append("\n## Morphs Accepted\n")
    morph_rows = summaries["morphs_accepted_as_trainable_visual_categories"]
    if morph_rows:
        for r in morph_rows[:80]:
            md.append(f"- `{r['proposed_canonical_label']}`: {r['reason']}")
    else:
        md.append("- none accepted from current raw labels")
    md.append("\n## Manual Review Needed\n")
    review_df = df[df["stage2_action"].eq("review")]
    if not review_df.empty:
        top = review_df.groupby(["project_stage1_species_group", "original_label"]).size().sort_values(ascending=False).head(100)
        for (sp, lab), cnt in top.items():
            md.append(f"- `{sp}` / `{lab}`: {int(cnt)} source occurrences")
    else:
        md.append("- none")
    (out_dir / "label_taxonomy_audit_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "reports" / "label_taxonomy_audit_20260706"))
    parser.add_argument("--active-hybrid-spec", default=str(REPO_ROOT / "data" / "processed" / "species_aware" / "hybrid_shared_qa_full_20260706_b12w12_yolo40" / "hybrid_dataset_spec.json"))
    parser.add_argument("--skip-processed-manifests", action="store_true")
    parser.add_argument("--processed-run-name", default="")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counter: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    extract_yolo_labels(REPO_ROOT, counter)
    extract_non_yolo_labels(REPO_ROOT, counter)
    if not args.skip_processed_manifests:
        extract_processed_manifest_labels(REPO_ROOT, counter, processed_run_name=str(args.processed_run_name or ""))
    items = build_items(counter)
    engine = AuditEngine(REPO_ROOT)
    rows = [engine.classify(item) for item in items]
    df = pd.DataFrame(rows)
    df = df.sort_values(["source_dataset", "source_kind", "original_label"])
    audit_csv = out_dir / "label_taxonomy_audit.csv"
    df.to_csv(audit_csv, index=False, encoding="utf-8")
    df.to_json(out_dir / "label_taxonomy_audit.json", orient="records", indent=2, force_ascii=False)

    active_stage1: list[str] = []
    spec_path = Path(args.active_hybrid_spec)
    if spec_path.exists():
        try:
            active_stage1 = list((json.loads(spec_path.read_text(encoding="utf-8"))).get("class_names", []))
        except Exception:
            active_stage1 = []
    write_summary(rows, out_dir, active_stage1)
    print(json.dumps({"out_dir": str(out_dir.resolve()), "rows": len(rows), "audit_csv": str(audit_csv.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
