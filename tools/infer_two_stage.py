"""Wrapper for editable/local execution of two-stage inference."""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from animal_breed.pipelines.infer_two_stage import main


if __name__ == "__main__":
    main()

