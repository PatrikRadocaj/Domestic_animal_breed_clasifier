"""Wrapper for editable/local execution of YOLO dataset preparation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from animal_breed.detection.yolo_dataset import main


if __name__ == "__main__":
    main()

