"""Configuration for PainFormer cardiac detection."""

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
class PainFormerConfig:
    """Configuration for PainFormer zero-shot cardiac detection."""
    # Model
    backbone_path: str = "tools/PainFormer/painformer.pth"
    embed_dim: int = 160
    num_classes: int = 3
    classification_mode: str = "zeroshot"

    # Zero-shot parameters
    pain_positive_dims: List[int] = field(
        default_factory=lambda: [69, 47, 9, 23, 53, 39, 85, 77, 104, 29]
    )
    pain_negative_dims: List[int] = field(
        default_factory=lambda: [109, 117, 25, 157, 113, 65, 141, 112, 152, 148]
    )
    threshold_normal: float = 0.30
    threshold_abnormal: float = 0.65
    temporal_smooth_k: int = 5

    # Normalization (VGG-Face2)
    norm_mean: List[float] = field(default_factory=lambda: [0.6068, 0.4517, 0.3800])
    norm_std: List[float] = field(default_factory=lambda: [0.2492, 0.2173, 0.2082])

    # Data
    image_size: int = 224

    # Labels
    labels: List[str] = field(default_factory=lambda: ["normal", "abnormal", "infarction"])

    @classmethod
    def from_yaml(cls, config_name: str = "painformer_config.yaml") -> "PainFormerConfig":
        config_path = CONFIGS_DIR / config_name
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        zs = raw.get("zeroshot", {})
        norm = raw.get("normalization", {})

        return cls(
            backbone_path=raw.get("model", {}).get("backbone_path", "tools/PainFormer/painformer.pth"),
            embed_dim=raw.get("model", {}).get("embed_dim", 160),
            num_classes=raw.get("model", {}).get("num_classes", 3),
            classification_mode=raw.get("model", {}).get("classification_mode", "zeroshot"),
            pain_positive_dims=zs.get("pain_positive_dims", [69, 47, 9, 23, 53, 39, 85, 77, 104, 29]),
            pain_negative_dims=zs.get("pain_negative_dims", [109, 117, 25, 157, 113, 65, 141, 112, 152, 148]),
            threshold_normal=zs.get("threshold_normal", 0.30),
            threshold_abnormal=zs.get("threshold_abnormal", 0.65),
            temporal_smooth_k=zs.get("temporal_smooth_k", 5),
            norm_mean=norm.get("mean", [0.6068, 0.4517, 0.3800]),
            norm_std=norm.get("std", [0.2492, 0.2173, 0.2082]),
            image_size=raw.get("data", {}).get("image_size", 224),
            labels=raw.get("labels", ["normal", "abnormal", "infarction"]),
        )


def load_painformer_config() -> PainFormerConfig:
    """Load PainFormer config from YAML."""
    return PainFormerConfig.from_yaml()


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
