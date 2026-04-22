"""Global configuration manager for the Heart Face 3D project."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import yaml


PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class LabelBPM:
    min: int
    max: int
    pattern: str  # regular, irregular, rapid_irregular


@dataclass
class Config:
    # Labels
    labels: List[str] = field(default_factory=lambda: ["normal", "abnormal", "infarction"])
    label_to_bpm: Dict[str, LabelBPM] = field(default_factory=dict)

    # Model
    backbone: str = "efficientnet_b4"
    num_classes: int = 3
    pretrained: bool = True
    dropout: float = 0.3

    # Data
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Training
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    mixed_precision: bool = True

    # Paths
    checkpoint_path: Path = MODELS_DIR / "checkpoints" / "best_model.pth"
    data_dir: Path = DATA_DIR

    @classmethod
    def from_yaml(cls, config_name: str = "train_config.yaml") -> "Config":
        config_path = CONFIGS_DIR / config_name
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        label_to_bpm = {}
        for label, bpm_data in raw.get("label_to_bpm", {}).items():
            label_to_bpm[label] = LabelBPM(**bpm_data)

        defaults = cls()

        return cls(
            labels=raw.get("labels", defaults.labels),
            label_to_bpm=label_to_bpm,
            backbone=raw.get("model", {}).get("backbone", defaults.backbone),
            num_classes=raw.get("model", {}).get("num_classes", defaults.num_classes),
            pretrained=raw.get("model", {}).get("pretrained", defaults.pretrained),
            dropout=raw.get("model", {}).get("dropout", defaults.dropout),
            image_size=raw.get("data", {}).get("image_size", defaults.image_size),
            batch_size=raw.get("data", {}).get("batch_size", defaults.batch_size),
            num_workers=raw.get("data", {}).get("num_workers", defaults.num_workers),
            train_split=raw.get("data", {}).get("train_split", defaults.train_split),
            val_split=raw.get("data", {}).get("val_split", defaults.val_split),
            test_split=raw.get("data", {}).get("test_split", defaults.test_split),
            epochs=raw.get("training", {}).get("epochs", defaults.epochs),
            learning_rate=raw.get("training", {}).get("learning_rate", defaults.learning_rate),
            weight_decay=raw.get("training", {}).get("weight_decay", defaults.weight_decay),
            mixed_precision=raw.get("training", {}).get("mixed_precision", defaults.mixed_precision),
        )


def load_heart_config() -> dict:
    config_path = CONFIGS_DIR / "heart_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_app_config() -> dict:
    config_path = CONFIGS_DIR / "app_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
