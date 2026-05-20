"""Wrapper for editable/local execution of dataset auditing."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from animal_breed.analysis.audit import main


if __name__ == "__main__":
    main()

