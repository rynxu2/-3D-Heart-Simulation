"""Data collector — capture face images from webcam or import from folder."""

import cv2
import os
import time
from pathlib import Path
from typing import Optional
from loguru import logger

from src.data.face_detector import FaceDetector


class DataCollector:
    """Collect and organize face images for training.

    Supports:
    - Webcam capture with live face detection
    - Batch import from existing image folders
    - Auto face-crop and save to labeled directories
    """

    def __init__(self, output_dir: str | Path, target_size: int = 224):
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.face_detector = FaceDetector(target_size=target_size)

        # Create label directories
        for label in ["normal", "abnormal", "infarction"]:
            (self.output_dir / label).mkdir(parents=True, exist_ok=True)

    def capture_from_webcam(
        self,
        label: str,
        num_images: int = 50,
        delay_ms: int = 500,
        camera_id: int = 0,
    ) -> int:
        """Capture face images from webcam with live preview.

        Args:
            label: "normal", "abnormal", or "infarction"
            num_images: Number of images to capture
            delay_ms: Delay between captures (ms)
            camera_id: Camera device ID

        Returns:
            Number of images saved
        """
        if label not in ["normal", "abnormal", "infarction"]:
            raise ValueError(f"Invalid label: {label}")

        save_dir = self.output_dir / label
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return 0

        saved = 0
        existing = len(list(save_dir.glob("*.jpg")))
        logger.info(f"Capturing '{label}' faces. Press 'q' to quit, 's' to skip frame.")

        while saved < num_images:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw detection on preview
            annotated = self.face_detector.draw_detections(frame)
            cv2.putText(
                annotated,
                f"Label: {label} | Saved: {saved}/{num_images} | Press SPACE to capture",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            cv2.imshow("Data Collector", annotated)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                continue

            # Auto-capture or manual (SPACE key)
            if key == 32 or delay_ms > 0:
                face = self.face_detector.process_image(frame)
                if face is not None:
                    filename = f"{label}_{existing + saved:05d}.jpg"
                    cv2.imwrite(str(save_dir / filename), face)
                    saved += 1
                    logger.debug(f"Saved: {filename}")

        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Captured {saved} images for '{label}'")
        return saved

    def import_from_folder(
        self,
        source_dir: str | Path,
        label: str,
        crop_faces: bool = True,
    ) -> int:
        """Import images from a folder, optionally crop faces.

        Args:
            source_dir: Directory containing images
            label: Target label
            crop_faces: Whether to auto-detect and crop faces

        Returns:
            Number of images imported
        """
        source_dir = Path(source_dir)
        save_dir = self.output_dir / label
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        imported = 0
        existing = len(list(save_dir.glob("*.jpg")))

        for img_file in sorted(source_dir.iterdir()):
            if img_file.suffix.lower() not in exts:
                continue

            image = cv2.imread(str(img_file))
            if image is None:
                continue

            if crop_faces:
                face = self.face_detector.process_image(image)
                if face is None:
                    logger.debug(f"No face in {img_file.name}, skipping")
                    continue
                save_image = face
            else:
                save_image = cv2.resize(image, (self.target_size, self.target_size))

            filename = f"{label}_{existing + imported:05d}.jpg"
            cv2.imwrite(str(save_dir / filename), save_image)
            imported += 1

        logger.info(f"Imported {imported} images for '{label}' from {source_dir}")
        return imported

    def get_dataset_stats(self) -> dict:
        """Get current dataset statistics."""
        stats = {}
        total = 0
        for label in ["normal", "abnormal", "infarction"]:
            label_dir = self.output_dir / label
            count = len(list(label_dir.glob("*.jpg"))) + len(list(label_dir.glob("*.png")))
            stats[label] = count
            total += count

        stats["total"] = total
        stats["balanced"] = len(set(stats[l] for l in ["normal", "abnormal", "infarction"])) == 1

        logger.info(f"Dataset stats: {stats}")
        return stats
