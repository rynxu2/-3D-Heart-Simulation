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

    def capture_video_clip(
        self,
        label: str,
        duration_sec: float = 2.5,
        num_frames: int = 16,
        camera_id: int = 0,
    ) -> int:
        """Capture short video clips from webcam for 3D CNN training.

        Records ~2.5s of video, extracts 16 equally-spaced frames,
        and saves them as a frame folder.

        Args:
            label: "normal", "abnormal", or "infarction"
            duration_sec: Clip duration in seconds
            num_frames: Number of frames to extract per clip
            camera_id: Camera device ID

        Returns:
            Number of clips saved
        """
        import time as _time

        if label not in ["normal", "abnormal", "infarction"]:
            raise ValueError(f"Invalid label: {label}")

        save_dir = self.output_dir / label
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        saved_clips = 0
        existing = len([d for d in save_dir.iterdir() if d.is_dir()]) if save_dir.exists() else 0

        logger.info(
            f"Recording '{label}' video clips ({duration_sec}s each, {num_frames} frames).\n"
            f"Press SPACE to start recording, 'q' to quit."
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = self.face_detector.draw_detections(frame)
            cv2.putText(
                annotated,
                f"Label: {label} | Clips: {saved_clips} | SPACE=Record  Q=Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            cv2.imshow("Video Clip Recorder", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != 32:  # SPACE
                continue

            # Record clip
            all_frames = []
            start = _time.time()
            cv2.putText(annotated, "● RECORDING", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            while _time.time() - start < duration_sec:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame.copy())
                preview = frame.copy()
                elapsed = _time.time() - start
                cv2.putText(preview, f"REC {elapsed:.1f}s / {duration_sec}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Video Clip Recorder", preview)
                cv2.waitKey(1)

            if len(all_frames) < num_frames:
                logger.warning(f"Not enough frames ({len(all_frames)}), skipping clip")
                continue

            # Extract equally-spaced frames with face crop
            indices = [int(i) for i in
                       __import__("numpy").linspace(0, len(all_frames) - 1, num_frames)]

            clip_dir = save_dir / f"clip_{existing + saved_clips:05d}"
            clip_dir.mkdir(parents=True, exist_ok=True)

            valid_frames = 0
            for fi, frame_idx in enumerate(indices):
                raw = all_frames[frame_idx]
                face = self.face_detector.process_image(raw)
                if face is not None:
                    cv2.imwrite(str(clip_dir / f"frame_{fi:02d}.jpg"), face)
                    valid_frames += 1
                else:
                    # Save full frame resized as fallback
                    resized = cv2.resize(raw, (self.target_size, self.target_size))
                    cv2.imwrite(str(clip_dir / f"frame_{fi:02d}.jpg"), resized)
                    valid_frames += 1

            saved_clips += 1
            logger.info(f"Clip saved: {clip_dir.name} ({valid_frames} frames)")

        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Captured {saved_clips} video clips for '{label}'")
        return saved_clips

    def generate_pseudo_videos(
        self,
        source_dir: str | Path,
        label: str,
        num_frames: int = 16,
        crop_faces: bool = True,
    ) -> int:
        """Generate pseudo-video clips from static SynPAIN images.

        Takes each image in source_dir, applies condition-specific temporal
        augmentations (rotation, breathing, pulse, skin tone), and saves
        as a 16-frame folder suitable for 3D CNN training.

        Motion profiles per condition:
        - normal:     gentle breathing, steady pulse
        - abnormal:   irregular breathing, mild tremor, variable saturation
        - infarction: rapid tremor, fast pulse, skin pallor

        Args:
            source_dir: Folder with .jpg/.png face images (e.g. SynPAIN export)
            label: "normal", "abnormal", or "infarction"
            num_frames: Frames per clip (default 16)
            crop_faces: Whether to detect + crop faces first

        Returns:
            Number of video clips generated
        """
        import numpy as np

        if label not in ["normal", "abnormal", "infarction"]:
            raise ValueError(f"Invalid label: {label}")

        source_dir = Path(source_dir)
        save_dir = self.output_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        source_images = sorted([
            f for f in source_dir.iterdir()
            if f.suffix.lower() in exts
        ])

        if not source_images:
            logger.warning(f"No images found in {source_dir}")
            return 0

        existing_clips = len([d for d in save_dir.iterdir() if d.is_dir()]) if save_dir.exists() else 0
        generated = 0

        # Condition-specific motion parameters
        MOTION = {
            "normal":     {"rot": 0.5, "scale_amp": 0.008, "bright": 2,  "tx": 0.5, "ty": 0,   "sat": 1.0,  "rot_freq": 2,  "scale_freq": 3,  "bright_freq": 4},
            "abnormal":   {"rot": 1.5, "scale_amp": 0.02,  "bright": 5,  "tx": 1.5, "ty": 1,   "sat": 0.92, "rot_freq": 4,  "scale_freq": 6,  "bright_freq": 8},
            "infarction": {"rot": 3.0, "scale_amp": 0.03,  "bright": 8,  "tx": 3,   "ty": 2,   "sat": 0.85, "rot_freq": 8,  "scale_freq": 12, "bright_freq": 16},
        }
        m = MOTION[label]

        for img_path in source_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Optionally crop face
            if crop_faces:
                face = self.face_detector.process_image(img)
                if face is not None:
                    img = face
                else:
                    img = cv2.resize(img, (self.target_size, self.target_size))
            else:
                img = cv2.resize(img, (self.target_size, self.target_size))

            h, w = img.shape[:2]
            center = (w // 2, h // 2)

            clip_dir = save_dir / f"clip_{existing_clips + generated:05d}"
            clip_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_frames):
                t = i / num_frames

                # Temporal transforms
                angle = np.sin(t * m["rot_freq"] * np.pi) * m["rot"]
                scale = 1.0 + np.sin(t * m["scale_freq"] * np.pi) * m["scale_amp"]
                brightness = np.sin(t * m["bright_freq"] * np.pi) * m["bright"]
                if label == "infarction":
                    brightness -= 10  # overall darker (pallor)

                M = cv2.getRotationMatrix2D(center, angle, scale)
                M[0, 2] += np.sin(t * (m["rot_freq"] - 1) * np.pi) * m["tx"]
                M[1, 2] += np.cos(t * (m["rot_freq"] - 2) * np.pi) * m["ty"]

                warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                warped = cv2.convertScaleAbs(warped, alpha=1.0, beta=brightness)

                # Saturation change (skin perfusion simulation)
                if m["sat"] != 1.0:
                    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:, :, 1] *= m["sat"]
                    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                    warped = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                cv2.imwrite(str(clip_dir / f"frame_{i:02d}.jpg"), warped)

            generated += 1

            if generated % 50 == 0:
                logger.info(f"  [{label}] {generated}/{len(source_images)} clips generated")

        logger.info(f"Generated {generated} pseudo-video clips for '{label}' from {source_dir}")
        return generated

    def get_dataset_stats(self) -> dict:
        """Get current dataset statistics."""
        stats = {}
        total = 0
        for label in ["normal", "abnormal", "infarction"]:
            label_dir = self.output_dir / label
            images = len(list(label_dir.glob("*.jpg"))) + len(list(label_dir.glob("*.png")))
            clips = len([d for d in label_dir.iterdir() if d.is_dir()]) if label_dir.exists() else 0
            stats[label] = {"images": images, "clips": clips}
            total += images + clips

        stats["total"] = total
        stats["balanced"] = len(set(
            stats[l]["images"] + stats[l]["clips"]
            for l in ["normal", "abnormal", "infarction"]
        )) == 1

        logger.info(f"Dataset stats: {stats}")
        return stats
