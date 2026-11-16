"""Hard negative mining from two prediction sets."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from animal_breed.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = setup_logging(cfg.logging.level)

    two_stage_path = Path(str(cfg.eval.hard_negative.two_stage_predictions_json))
    single_stage_path = Path(str(cfg.eval.hard_negative.single_stage_predictions_json))
    out_csv = Path(str(cfg.eval.hard_negative.output_csv))

    if not two_stage_path.exists() or not single_stage_path.exists():
        raise FileNotFoundError("Prediction files for hard-negative mining were not found.")

    two = json.loads(two_stage_path.read_text(encoding="utf-8"))
    single = json.loads(single_stage_path.read_text(encoding="utf-8"))

    by_image_single = {str(row.get("image_path")): row for row in single}
    mined: list[dict[str, object]] = []

    for row in two:
        image_path = str(row.get("image_path"))
        s_row = by_image_single.get(image_path)
        if s_row is None:
            continue

        two_det = row.get("detections", [])
        one_det = s_row.get("detections", [])
        if not two_det or not one_det:
            continue

        two_breed = str(two_det[0].get("breed_pred", ""))
        one_breed = str(one_det[0].get("breed_pred", ""))
        two_conf = float(two_det[0].get("breed_conf") or 0.0)
        one_conf = float(one_det[0].get("breed_conf") or 0.0)

        disagreement = two_breed != one_breed
        high_conf = (two_conf >= float(cfg.eval.hard_negative.min_confidence)) or (one_conf >= float(cfg.eval.hard_negative.min_confidence))
        if disagreement and high_conf:
            mined.append(
                {
                    "image_path": image_path,
                    "two_stage_breed": two_breed,
                    "single_stage_breed": one_breed,
                    "two_stage_conf": two_conf,
                    "single_stage_conf": one_conf,
                    "reason": "high_conf_disagreement",
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mined).to_csv(out_csv, index=False)
    logger.info("Mined %d hard negatives -> %s", len(mined), out_csv)


if __name__ == "__main__":
    main()
