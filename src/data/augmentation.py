"""Data augmentation pipeline using Albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Optional


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Training augmentation pipeline — aggressive but realistic."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(var_limit=(10, 30), p=1.0),
        ], p=0.3),
        A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation/test transforms — only resize and normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def augment_image(image: np.ndarray, transform: Optional[A.Compose] = None) -> np.ndarray:
    """Apply augmentation to a single image."""
    if transform is None:
        transform = get_train_transforms()
    result = transform(image=image)
    return result["image"]
