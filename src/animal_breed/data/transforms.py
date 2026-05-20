"""Image transforms for classification training and validation."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def classification_train_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.65, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.4),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.ImageCompression(quality_range=(65, 100)),
                ],
                p=0.2,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(0.05, 0.18),
                hole_width_range=(0.05, 0.18),
                p=0.25,
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def classification_eval_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

