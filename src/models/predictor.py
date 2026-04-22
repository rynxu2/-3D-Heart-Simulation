"""Inference module — predict heart condition from face image and map to BPM."""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from loguru import logger

from src.config import Config, LabelBPM
from src.data.face_detector import FaceDetector
from src.data.augmentation import get_val_transforms
from src.data.dataset import LABEL_NAMES
from src.models.classifier_2d import HeartFaceClassifier2D


class HeartPredictor:
    """End-to-end: face image → heart condition label + BPM."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        self.config = config or Config.from_yaml()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Face detector
        self.face_detector = FaceDetector(target_size=self.config.image_size)

        # Transform
        self.transform = get_val_transforms(self.config.image_size)

        # Load model
        self.model = HeartFaceClassifier2D(
            backbone=self.config.backbone,
            num_classes=self.config.num_classes,
            pretrained=False,
        ).to(self.device)

        if model_path:
            self._load_checkpoint(model_path)

        self.model.eval()

    def _load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded model from {path} (val_acc={ckpt.get('val_accuracy', 'N/A')})")

    def predict_image(self, image: np.ndarray) -> dict:
        """Predict from BGR image array.

        Returns:
            {
                "label": "normal" | "abnormal" | "infarction",
                "label_id": 0 | 1 | 2,
                "confidence": float,
                "all_probs": {label: prob},
                "bpm": {"min": int, "max": int, "pattern": str},
                "face_detected": bool,
            }
        """
        # Detect and crop face
        face = self.face_detector.process_image(image)
        if face is None:
            return {"face_detected": False, "label": "unknown", "confidence": 0.0}

        # Preprocess
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=face_rgb)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_label = LABEL_NAMES[pred_id]
        confidence = float(probs[pred_id])

        # BPM mapping
        bpm_info = self.config.label_to_bpm.get(pred_label, LabelBPM(60, 80, "regular"))

        return {
            "face_detected": True,
            "label": pred_label,
            "label_id": pred_id,
            "confidence": confidence,
            "all_probs": {LABEL_NAMES[i]: float(probs[i]) for i in range(len(probs))},
            "bpm": {"min": bpm_info.min, "max": bpm_info.max, "pattern": bpm_info.pattern},
        }

    def predict_file(self, image_path: str | Path) -> dict:
        """Predict from image file path."""
        image = cv2.imread(str(image_path))
        if image is None:
            return {"face_detected": False, "label": "unknown", "error": "Cannot read image"}
        return self.predict_image(image)
