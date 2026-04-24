"""Inference module — predict heart condition from face image/video and map to BPM."""

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


# 3-class label mappings
LABEL_NAMES_3CLASS = {0: "normal", 1: "abnormal", 2: "infarction"}
LABEL_NAMES_BINARY = {0: "no_pain", 1: "pain"}

# BPM mappings per label
LABEL_BPM_MAP = {
    "normal": LabelBPM(60, 80, "regular"),
    "no_pain": LabelBPM(60, 80, "regular"),
    "abnormal": LabelBPM(40, 120, "irregular"),
    "infarction": LabelBPM(90, 130, "rapid_irregular"),
    "pain": LabelBPM(90, 130, "rapid_irregular"),
}


class HeartPredictor:
    """End-to-end: face image → heart condition label + BPM."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        num_classes: int = 3,
    ):
        self.config = config or Config.from_yaml()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.label_map = LABEL_NAMES_3CLASS if num_classes == 3 else LABEL_NAMES_BINARY

        # Face detector
        self.face_detector = FaceDetector(target_size=self.config.image_size)

        # Transform
        self.transform = get_val_transforms(self.config.image_size)

        # Load model
        from src.models.classifier_2d import HeartFaceClassifier2D
        self.model = HeartFaceClassifier2D(
            backbone=self.config.backbone,
            num_classes=num_classes,
            pretrained=False,
        ).to(self.device)

        if model_path:
            self._load_checkpoint(model_path)

        self.model.eval()

    def _load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        logger.info(f"Loaded 2D model from {path}")

    def predict_image(self, image: np.ndarray) -> dict:
        """Predict from BGR image array."""
        face = self.face_detector.process_image(image)
        if face is None:
            return {"face_detected": False, "label": "unknown", "confidence": 0.0}

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=face_rgb)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_label = self.label_map[pred_id]
        confidence = float(probs[pred_id])
        bpm_info = LABEL_BPM_MAP.get(pred_label, LabelBPM(60, 80, "regular"))

        return {
            "face_detected": True,
            "label": pred_label,
            "label_id": pred_id,
            "confidence": confidence,
            "all_probs": {self.label_map[i]: float(probs[i]) for i in range(len(probs))},
            "bpm": {"min": bpm_info.min, "max": bpm_info.max, "pattern": bpm_info.pattern},
        }

    def predict_file(self, image_path: str | Path) -> dict:
        """Predict from image file path."""
        image = cv2.imread(str(image_path))
        if image is None:
            return {"face_detected": False, "label": "unknown", "error": "Cannot read image"}
        return self.predict_image(image)


class HeartPredictor3D:
    """End-to-end: face video → heart condition label + BPM (3D CNN)."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        num_classes: int = 3,
        num_frames: int = 16,
        image_size: int = 224,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.image_size = image_size
        self.label_map = LABEL_NAMES_3CLASS if num_classes == 3 else LABEL_NAMES_BINARY

        self.face_detector = FaceDetector(target_size=image_size)
        self.transform = get_val_transforms(image_size)

        from src.models.classifier_3d import HeartFaceClassifier3D
        self.model = HeartFaceClassifier3D(
            num_classes=num_classes,
            pretrained=False,
            num_frames=num_frames,
        ).to(self.device)

        if model_path:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            logger.info(f"Loaded 3D model from {model_path}")

        self.model.eval()

    def predict_video(self, video_path: str | Path) -> dict:
        """Predict from video file."""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return {"face_detected": False, "label": "unknown", "error": "Cannot read video"}

        # Extract frames
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            face = self.face_detector.process_image(frame)
            if face is None:
                face = cv2.resize(frame, (self.image_size, self.image_size))
            else:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            augmented = self.transform(image=face if face.shape[2] == 3 else cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            frames.append(augmented["image"])

        cap.release()

        if len(frames) < self.num_frames:
            # Pad
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else torch.zeros(3, self.image_size, self.image_size))

        # Stack: (T, C, H, W) → (C, T, H, W)
        video_tensor = torch.stack(frames[:self.num_frames], dim=1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(video_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_label = self.label_map[pred_id]
        confidence = float(probs[pred_id])
        bpm_info = LABEL_BPM_MAP.get(pred_label, LabelBPM(60, 80, "regular"))

        return {
            "face_detected": True,
            "label": pred_label,
            "label_id": pred_id,
            "confidence": confidence,
            "all_probs": {self.label_map[i]: float(probs[i]) for i in range(len(probs))},
            "bpm": {"min": bpm_info.min, "max": bpm_info.max, "pattern": bpm_info.pattern},
            "model_type": "3d_cnn",
            "frames_used": len(frames),
        }

    def predict_webcam_frames(self, frames: list) -> dict:
        """Predict from a list of BGR numpy frames (from webcam)."""
        if len(frames) < 2:
            return {"face_detected": False, "label": "unknown", "error": "Not enough frames"}

        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        processed = []

        for idx in indices:
            frame = frames[idx]
            face = self.face_detector.process_image(frame)
            if face is None:
                face = cv2.resize(frame, (self.image_size, self.image_size))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            else:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            augmented = self.transform(image=face)
            processed.append(augmented["image"])

        while len(processed) < self.num_frames:
            processed.append(processed[-1])

        video_tensor = torch.stack(processed[:self.num_frames], dim=1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(video_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_label = self.label_map[pred_id]
        confidence = float(probs[pred_id])
        bpm_info = LABEL_BPM_MAP.get(pred_label, LabelBPM(60, 80, "regular"))

        return {
            "face_detected": True,
            "label": pred_label,
            "label_id": pred_id,
            "confidence": confidence,
            "all_probs": {self.label_map[i]: float(probs[i]) for i in range(len(probs))},
            "bpm": {"min": bpm_info.min, "max": bpm_info.max, "pattern": bpm_info.pattern},
            "model_type": "3d_cnn",
        }
