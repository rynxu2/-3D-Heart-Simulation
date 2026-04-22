"""3D CNN classifier for heart condition detection from face videos."""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights
from typing import Optional
from loguru import logger


class HeartFaceClassifier3D(nn.Module):
    """ResNet3D-18 based video classifier for temporal facial analysis.

    Input: (B, 3, T, 224, 224) — T=16 frames of face video
    Output: (B, num_classes) logits
    """

    BACKBONES = {
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT, 512),
        "mc3_18": (mc3_18, MC3_18_Weights.DEFAULT, 512),
    }

    def __init__(
        self,
        backbone: str = "r3d_18",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        num_frames: int = 16,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.num_frames = num_frames

        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}. Options: {list(self.BACKBONES.keys())}")

        model_fn, weights, feat_dim = self.BACKBONES[backbone]
        base = model_fn(weights=weights if pretrained else None)

        # Remove original FC head
        self.features = nn.Sequential(*list(base.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        logger.info(f"Created 3D-{backbone}: {feat_dim} → 256 → {num_classes}, frames={num_frames}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B, 3, T, H, W)"""
        features = self.features(x)
        return self.classifier(features)

    def predict(self, x: torch.Tensor) -> tuple:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)
        return pred_class, probs


def create_3d_classifier(
    backbone: str = "r3d_18",
    num_classes: int = 3,
    pretrained: bool = True,
    device: Optional[str] = None,
) -> HeartFaceClassifier3D:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HeartFaceClassifier3D(backbone, num_classes, pretrained)
    model = model.to(device)
    logger.info(f"3D Model on {device} | Params: {sum(p.numel() for p in model.parameters()):,}")
    return model
