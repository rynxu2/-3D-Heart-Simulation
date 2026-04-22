"""PyTorch Dataset for heart condition face classification."""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from loguru import logger

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.config import Config


LABEL_MAP = {"normal": 0, "abnormal": 1, "infarction": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


class HeartFaceDataset(Dataset):
    """Dataset loading face images with heart condition labels."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        image_size: int = 224,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or get_val_transforms(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        if image is None:
            # Return a black image if file is corrupted
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            logger.warning(f"Cannot read: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        tensor = augmented["image"]

        return tensor, label

    def get_class_weights(self) -> List[float]:
        """Compute inverse-frequency class weights for imbalanced data."""
        from collections import Counter
        counts = Counter(self.labels)
        total = len(self.labels)
        weights = [total / (len(counts) * counts[i]) for i in range(len(counts))]
        return weights


def build_dataset_from_directory(
    data_dir: str | Path,
    config: Optional[Config] = None,
) -> Tuple[List[str], List[int]]:
    """Scan directory structure and collect image paths + labels.

    Expected structure:
        data_dir/
            normal/     ← images
            abnormal/   ← images
            infarction/ ← images
    """
    data_dir = Path(data_dir)
    image_paths = []
    labels = []

    for label_name, label_id in LABEL_MAP.items():
        label_dir = data_dir / label_name
        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for img_file in sorted(label_dir.iterdir()):
            if img_file.suffix.lower() in exts:
                image_paths.append(str(img_file))
                labels.append(label_id)

    logger.info(f"Found {len(image_paths)} images across {len(LABEL_MAP)} classes")
    for name, lid in LABEL_MAP.items():
        count = labels.count(lid)
        logger.info(f"  {name}: {count} images")

    return image_paths, labels


def create_dataloaders(
    data_dir: str | Path,
    config: Optional[Config] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from directory."""
    if config is None:
        config = Config.from_yaml()

    image_paths, labels = build_dataset_from_directory(data_dir, config)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")

    # Stratified split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(config.val_split + config.test_split),
        stratify=labels,
        random_state=42,
    )
    val_ratio = config.val_split / (config.val_split + config.test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=42,
    )

    train_ds = HeartFaceDataset(train_paths, train_labels, get_train_transforms(config.image_size), config.image_size)
    val_ds = HeartFaceDataset(val_paths, val_labels, get_val_transforms(config.image_size), config.image_size)
    test_ds = HeartFaceDataset(test_paths, test_labels, get_val_transforms(config.image_size), config.image_size)

    logger.info(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
