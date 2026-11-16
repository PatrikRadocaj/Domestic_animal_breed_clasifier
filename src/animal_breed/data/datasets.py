"""PyTorch datasets backed by the canonical manifest."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BreedClassificationDataset(Dataset):
    """Image classification dataset with optional bbox cropping."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        label_to_index: dict[str, int],
        transform,
        label_column: str = "breed_id",
        crop_from_bbox: bool = False,
        crop_context_ratio: float = 0.12,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.transform = transform
        self.label_column = label_column
        self.crop_from_bbox = crop_from_bbox
        self.crop_context_ratio = crop_context_ratio

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[index]
        path = Path(str(row.path))
        image = self.load_rgb_image(path)

        if self.crop_from_bbox:
            image = self._crop_bbox(image, row)

        transformed = self.transform(image=image)
        label = self.label_to_index[str(row[self.label_column])]
        return transformed["image"], torch.tensor(label, dtype=torch.long)

    @classmethod
    def load_rgb_image(cls, path: str | Path) -> np.ndarray:
        src = Path(path)
        image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {src}")
        return cls._normalize_image(image)

    def _crop_bbox(self, image, row):
        bbox_values = [row.bbox_xmin, row.bbox_ymin, row.bbox_xmax, row.bbox_ymax]
        if any(pd.isna(value) for value in bbox_values):
            return image

        height, width = image.shape[:2]
        xmin, ymin, xmax, ymax = [float(value) for value in bbox_values]
        box_w = max(1.0, xmax - xmin)
        box_h = max(1.0, ymax - ymin)
        margin_x = box_w * self.crop_context_ratio
        margin_y = box_h * self.crop_context_ratio

        x0 = int(max(0, xmin - margin_x))
        y0 = int(max(0, ymin - margin_y))
        x1 = int(min(width, xmax + margin_x))
        y1 = int(min(height, ymax + margin_y))
        return image[y0:y1, x0:x1]

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if image.dtype == np.uint8:
            return image

        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max
            if max_value <= 0:
                raise ValueError(f"Invalid integer image dtype max: {image.dtype}")
            image = (image.astype(np.float32) / float(max_value) * 255.0).clip(0, 255).astype(np.uint8)
            return image

        if np.issubdtype(image.dtype, np.floating):
            max_value = float(np.nanmax(image)) if image.size else 1.0
            if not np.isfinite(max_value) or max_value <= 0:
                max_value = 1.0
            scale = 255.0 if max_value <= 1.0 else 255.0 / max_value
            image = (image.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
            return image

        raise ValueError(f"Unsupported image dtype: {image.dtype}")

