"""SynPAIN dataset loader — download, parse, and create PyTorch DataLoaders.

Handles:
- Download via HuggingFace datasets API
- Filename parsing to extract Pain/NoPain + demographics
- Filtering images only (skip video entries)
- Label mapping (binary or 3-class)
- Stratified train/val/test split
- PyTorch Dataset wrapper with augmentation
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from loguru import logger

from src.data.augmentation import get_train_transforms, get_val_transforms


# Label mappings
SYNPAIN_BINARY_MAP = {"NoPain": 0, "Pain": 1}
SYNPAIN_BINARY_NAMES = {0: "no_pain", 1: "pain"}

SYNPAIN_3CLASS_MAP = {"NoPain": 0, "Pain": 2}
SYNPAIN_3CLASS_NAMES = {0: "normal", 1: "abnormal", 2: "infarction"}

# Ethnicity decoding from ID
ETHNICITY_MAP = {
    "0": "Black",
    "1": "South Asian",
    "2": "Middle Eastern",
    "3": "Caucasian",
    "4": "East Asian",
}


def parse_synpain_filename(filename: str) -> Dict:
    """Parse SynPAIN filename to extract metadata.

    Filename format: [ID]_[expression]_[gender]_[age].jpg
    ID format: 1[pain][gender][age][ethnicity][5-digit-id]

    Example: "1100300001_Pain_Man_Young.jpg"
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 4:
        return {"valid": False, "filename": filename}

    id_str = parts[0]
    expression = parts[1]
    gender = parts[2]
    age = parts[3]

    ethnicity = "Unknown"
    if len(id_str) >= 5:
        eth_code = id_str[4]
        ethnicity = ETHNICITY_MAP.get(eth_code, "Unknown")

    return {
        "valid": True,
        "filename": filename,
        "id": id_str,
        "expression": expression,
        "gender": gender,
        "age": age,
        "ethnicity": ethnicity,
        "is_pain": expression == "Pain",
    }


def load_synpain_dataset(
    cache_dir: Optional[str] = None,
    filter_type: str = "Images",
) -> Tuple[list, list]:
    """Load SynPAIN from HuggingFace and return (images, metadata).

    Returns:
        images: list of PIL Image objects
        metadata: list of parsed filename dicts
    """
    from datasets import load_dataset

    logger.info("Loading SynPAIN dataset from HuggingFace...")
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    ds = load_dataset("TaatiTeam/SynPAIN", split="train", **kwargs)

    logger.info(f"Raw dataset: {len(ds)} entries")
    logger.info(f"Columns: {ds.column_names}")
    logger.info(f"Features: {ds.features}")

    # Get label feature names
    label_names = ds.features["label"].names
    logger.info(f"Label classes: {label_names}")

    # Filter: keep only images (skip videos)
    filter_idx = label_names.index(filter_type) if filter_type in label_names else 0
    ds_filtered = ds.filter(lambda x: x["label"] == filter_idx)
    logger.info(f"After filtering '{filter_type}': {len(ds_filtered)} entries")

    # Extract images and parse metadata from paths
    images = []
    metadata = []

    for i, example in enumerate(ds_filtered):
        img = example["image"]
        images.append(img)

        # Try to get filename from the image path
        img_path = ""
        if hasattr(img, "filename") and img.filename:
            img_path = img.filename
        elif "image" in example and hasattr(example["image"], "filename"):
            img_path = example["image"].filename or ""

        # Parse filename for metadata
        if img_path:
            meta = parse_synpain_filename(Path(img_path).name)
        else:
            meta = {"valid": False, "filename": f"unknown_{i}"}

        metadata.append(meta)

    valid_count = sum(1 for m in metadata if m.get("valid", False))
    logger.info(f"Parsed metadata: {valid_count}/{len(metadata)} valid filenames")

    return images, metadata


def create_labels_from_metadata(
    metadata: List[Dict],
    strategy: str = "binary",
) -> Tuple[List[int], List[Dict]]:
    """Create numeric labels from parsed metadata.

    Args:
        metadata: list of parsed filename dicts
        strategy: "binary" (Pain/NoPain) or "3class" (normal/abnormal/infarction)

    Returns:
        labels: list of integer labels
        valid_metadata: filtered list (only entries with valid parsing)
    """
    label_map = SYNPAIN_BINARY_MAP if strategy == "binary" else SYNPAIN_3CLASS_MAP
    labels = []
    valid_indices = []

    for i, meta in enumerate(metadata):
        if not meta.get("valid", False):
            continue

        expression = meta.get("expression", "")
        if expression in label_map:
            labels.append(label_map[expression])
            valid_indices.append(i)

    logger.info(f"Created {len(labels)} labels (strategy={strategy})")
    counts = Counter(labels)
    for label_id, count in sorted(counts.items()):
        name_map = SYNPAIN_BINARY_NAMES if strategy == "binary" else SYNPAIN_3CLASS_NAMES
        logger.info(f"  {name_map.get(label_id, label_id)}: {count}")

    return labels, valid_indices


class SynPainDataset(Dataset):
    """PyTorch Dataset wrapper for SynPAIN images."""

    def __init__(
        self,
        images: list,
        labels: List[int],
        metadata: List[Dict],
        transform=None,
        image_size: int = 224,
    ):
        self.images = images
        self.labels = labels
        self.metadata = metadata
        self.transform = transform or get_val_transforms(image_size)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        pil_image = self.images[idx]
        label = self.labels[idx]

        # Convert PIL to numpy RGB
        image_np = np.array(pil_image.convert("RGB"))

        # Apply albumentations transform
        augmented = self.transform(image=image_np)
        tensor = augmented["image"]

        return tensor, label

    def get_class_weights(self) -> List[float]:
        """Compute inverse-frequency class weights."""
        counts = Counter(self.labels)
        total = len(self.labels)
        num_classes = len(counts)
        weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
        return weights

    def get_demographics(self) -> List[Dict]:
        """Return demographic info for bias analysis."""
        return self.metadata


def create_synpain_loaders(
    cache_dir: Optional[str] = None,
    strategy: str = "binary",
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 2,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Full pipeline: download → parse → split → DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    # Load dataset
    images, metadata = load_synpain_dataset(cache_dir=cache_dir)

    # Create labels
    labels, valid_indices = create_labels_from_metadata(metadata, strategy)

    # Filter to valid entries only
    valid_images = [images[i] for i in valid_indices]
    valid_metadata = [metadata[i] for i in valid_indices]

    if len(valid_images) == 0:
        raise ValueError("No valid images found after parsing SynPAIN filenames")

    # Stratified split
    indices = list(range(len(valid_images)))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=42,
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_ratio = val_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=42,
    )

    # Build datasets
    def subset(idx_list):
        return (
            [valid_images[i] for i in idx_list],
            [labels[i] for i in idx_list],
            [valid_metadata[i] for i in idx_list],
        )

    train_imgs, train_labels, train_meta = subset(train_idx)
    val_imgs, val_labels, val_meta = subset(val_idx)
    test_imgs, test_labels, test_meta = subset(test_idx)

    train_ds = SynPainDataset(train_imgs, train_labels, train_meta, get_train_transforms(image_size), image_size)
    val_ds = SynPainDataset(val_imgs, val_labels, val_meta, get_val_transforms(image_size), image_size)
    test_ds = SynPainDataset(test_imgs, test_labels, test_meta, get_val_transforms(image_size), image_size)

    logger.info(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    dataset_info = {
        "total_images": len(valid_images),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "class_weights": train_ds.get_class_weights(),
        "strategy": strategy,
    }

    return train_loader, val_loader, test_loader, dataset_info
