"""Face detection and preprocessing using MediaPipe."""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger


class FaceDetector:
    """Detect and crop faces from images using MediaPipe Face Detection."""

    def __init__(self, min_confidence: float = 0.5, target_size: int = 224):
        self.target_size = target_size
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,  # 0=short-range, 1=full-range
            min_detection_confidence=min_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """Detect all faces in image. Returns list of {bbox, confidence}."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        faces = []
        if not results.detections:
            return faces

        h, w = image.shape[:2]
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Add padding (20%) for better face context
            pad_x = int(bw * 0.2)
            pad_y = int(bh * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)

            faces.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": detection.score[0],
            })

        return faces

    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop and resize face region to target_size."""
        x1, y1, x2, y2 = bbox
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (self.target_size, self.target_size))
        return face

    def process_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect the primary face and return cropped, resized face image."""
        faces = self.detect_faces(image)
        if not faces:
            logger.warning("No face detected in image")
            return None

        # Pick highest confidence face
        best = max(faces, key=lambda f: f["confidence"])
        return self.crop_face(image, best["bbox"])

    def process_file(self, image_path: str | Path) -> Optional[np.ndarray]:
        """Load image file, detect and crop face."""
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            return None

        return self.process_image(image)

    def draw_detections(self, image: np.ndarray) -> np.ndarray:
        """Draw bounding boxes on detected faces (for visualization)."""
        annotated = image.copy()
        faces = self.detect_faces(image)

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            conf = face["confidence"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"{conf:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        return annotated

    def __del__(self):
        if hasattr(self, "detector"):
            self.detector.close()
