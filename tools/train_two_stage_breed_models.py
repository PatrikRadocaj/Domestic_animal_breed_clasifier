"""Wrapper for editable/local execution of two-stage breed-model training."""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from animal_breed.classification.train_two_stage_breed_models import main


if __name__ == "__main__":
    main()
