"""Extract face frames from video files for training data.

Scans data/raw/{normal,abnormal,infarction}/ for .mp4/.avi files,
extracts frames at regular intervals, detects + crops faces,
and saves them as .jpg images in the same directory.

Usage:
    python scripts/extract_frames_from_videos.py
    python scripts/extract_frames_from_videos.py --data-dir data/raw --fps 2 --max-frames 100
    python scripts/extract_frames_from_videos.py --label normal --fps 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from loguru import logger

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LABELS = ["normal", "abnormal", "infarction"]


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
    max_frames: int = 200,
    target_size: int = 224,
    use_face_detection: bool = True,
):
    """Extract frames from a video file.

    Args:
        video_path: Path to video file
        output_dir: Where to save extracted frames
        fps: Frames per second to extract (e.g., 2 = every 0.5s)
        max_frames: Maximum frames to extract per video
        target_size: Output image size
        use_face_detection: Try to detect and crop faces

    Returns:
        Number of frames saved
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))

    # Face detector (optional)
    face_cascade = None
    if use_face_detection:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

    saved = 0
    frame_idx = 0
    video_name = video_path.stem

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        output_img = None

        # Try face detection first
        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
            )

            if len(faces) > 0:
                # Take largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                # Add padding (30% around face)
                pad = int(max(w, h) * 0.3)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)

                face_crop = frame[y1:y2, x1:x2]
                output_img = cv2.resize(face_crop, (target_size, target_size))

        # Fallback: center crop
        if output_img is None:
            h, w = frame.shape[:2]
            size = min(h, w)
            y1 = (h - size) // 2
            x1 = (w - size) // 2
            center_crop = frame[y1:y1 + size, x1:x1 + size]
            output_img = cv2.resize(center_crop, (target_size, target_size))

        # Save
        filename = f"{video_name}_frame_{saved:04d}.jpg"
        cv2.imwrite(str(output_dir / filename), output_img)
        saved += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video files")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--label", type=str, choices=LABELS, default=None)
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frames to extract per second of video")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Max frames per video")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--no-face-detect", action="store_true",
                        help="Skip face detection, just center-crop")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    labels = [args.label] if args.label else LABELS
    total = 0

    for label in labels:
        label_dir = data_dir / label
        if not label_dir.exists():
            logger.warning(f"Directory not found: {label_dir}")
            continue

        videos = [f for f in label_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS]
        if not videos:
            logger.info(f"[{label}] No video files found")
            continue

        logger.info(f"[{label}] Found {len(videos)} videos, extracting frames...")

        for video_path in videos:
            count = extract_frames(
                video_path=video_path,
                output_dir=label_dir,
                fps=args.fps,
                max_frames=args.max_frames,
                target_size=args.size,
                use_face_detection=not args.no_face_detect,
            )
            logger.info(f"  {video_path.name} → {count} frames")
            total += count

    logger.info(f"\n✅ Extracted {total} frames total")
    logger.info(f"Now you can run:")
    logger.info(f"  python scripts/generate_pseudo_videos.py --input-dir {data_dir} --output-dir data/video_clips")
    logger.info(f"  OR")
    logger.info(f"  python scripts/tpsmm_video_generator.py --source-dir {data_dir} --output-dir data/video_clips --no-tpsmm")


if __name__ == "__main__":
    main()
