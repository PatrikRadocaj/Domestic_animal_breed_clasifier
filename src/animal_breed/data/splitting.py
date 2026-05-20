"""Create reproducible train/validation/test splits from a manifest."""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from animal_breed.data.manifest_schema import read_manifest
from animal_breed.utils.logging import setup_logging


def _safe_stratify(labels: pd.Series, min_count: int) -> pd.Series | None:
    counts = labels.value_counts()
    if counts.empty or counts.min() < min_count:
        return None
    return labels


def create_splits(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    stratify_by: str,
    group_by: str,
    seed: int,
    min_count_for_stratify: int,
) -> pd.DataFrame:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    work = df.copy()
    if group_by in work.columns:
        units = work.groupby(group_by, as_index=False).agg({stratify_by: "first"})
        unit_ids = units[group_by]
        unit_labels = units[stratify_by]
    else:
        group_by = "image_id"
        units = work[["image_id", stratify_by]].rename(columns={"image_id": group_by})
        unit_ids = units[group_by]
        unit_labels = units[stratify_by]

    stratify = _safe_stratify(unit_labels, min_count_for_stratify)
    train_ids, temp_ids = train_test_split(
        unit_ids,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )

    temp = units[units[group_by].isin(temp_ids)]
    relative_val = val_size / (val_size + test_size)
    temp_stratify = _safe_stratify(temp[stratify_by], min_count_for_stratify)
    val_ids, test_ids = train_test_split(
        temp[group_by],
        train_size=relative_val,
        random_state=seed,
        stratify=temp_stratify,
    )

    split_map = {unit_id: "train" for unit_id in train_ids}
    split_map.update({unit_id: "val" for unit_id in val_ids})
    split_map.update({unit_id: "test" for unit_id in test_ids})

    work["split"] = work[group_by].map(split_map)
    if work["split"].isna().any():
        raise ValueError("Internal split assignment error: some rows have no split.")
    return work


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)
    df = read_manifest(cfg.data.manifest_path)
    split_df = create_splits(
        df=df,
        train_size=float(cfg.data.splitting.train_size),
        val_size=float(cfg.data.splitting.val_size),
        test_size=float(cfg.data.splitting.test_size),
        stratify_by=str(cfg.data.splitting.stratify_by),
        group_by=str(cfg.data.splitting.group_by),
        seed=int(cfg.seed),
        min_count_for_stratify=int(cfg.data.splitting.min_count_for_stratify),
    )

    output_path = Path(cfg.data.split_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    logger.info("Wrote split manifest to %s", output_path)
    logger.info("Split counts: %s", split_df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
