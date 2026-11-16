"""Wrapper for editable/local execution of the model inference app."""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from animal_breed.webapp.model_inference_app import main


if __name__ == "__main__":
    main()
