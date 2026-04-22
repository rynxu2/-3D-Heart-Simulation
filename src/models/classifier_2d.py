"""2D CNN classifier for heart condition detection from face images."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
from loguru import logger


class HeartFaceClassifier2D(nn.Module):
    """EfficientNet-B4 / ResNet-50 based face classifier.

    Input: (B, 3, 224, 224) face images
    Output: (B, num_classes) logits
    """

    BACKBONES = {
        "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT, 1792),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    }

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}. Options: {list(self.BACKBONES.keys())}")

        model_fn, weights, feat_dim = self.BACKBONES[backbone]
        base = model_fn(weights=weights if pretrained else None)

        # Remove original classifier head
        if "efficientnet" in backbone:
            self.features = nn.Sequential(*list(base.children())[:-1])
            # EfficientNet already has AdaptiveAvgPool2d
        else:
            # ResNet
            self.features = nn.Sequential(*list(base.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        logger.info(f"Created {backbone} classifier: {feat_dim} → 512 → {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def predict(self, x: torch.Tensor) -> tuple:
        """Return predicted class index and confidence scores."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)
        return pred_class, probs

    def freeze_backbone(self):
        """Freeze feature extractor for transfer learning warmup."""
        for param in self.features.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only classifier head will be trained")

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — all layers trainable")


def create_classifier(
    backbone: str = "efficientnet_b4",
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: Optional[str] = None,
) -> HeartFaceClassifier2D:
    """Factory function to create and move model to device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HeartFaceClassifier2D(backbone, num_classes, pretrained, dropout)
    model = model.to(device)
    logger.info(f"Model on {device} | Params: {sum(p.numel() for p in model.parameters()):,}")
    return model
