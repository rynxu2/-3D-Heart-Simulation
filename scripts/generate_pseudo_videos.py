"""Quick pseudo-video generation from static images (no TPSMM needed).

Wrapper around tpsmm_video_generator.py's SimplePseudoAnimator for
fast dataset augmentation without external dependencies.

Usage:
    # Generate for all labels
    python scripts/generate_pseudo_videos.py --input-dir data/raw --output-dir data/video_clips

    # Generate for specific label
    python scripts/generate_pseudo_videos.py --input-dir data/raw --output-dir data/video_clips --label infarction

    # Custom frame count
    python scripts/generate_pseudo_videos.py --input-dir data/raw --output-dir data/video_clips --frames 32
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger

LABELS = ["normal", "abnormal", "infarction"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def generate_pseudo_clip(
    image_path: Path,
    output_dir: Path,
    label: str = "normal",
    num_frames: int = 16,
    target_size: int = 224,
) -> bool:
    """Generate a pseudo-video clip from a single image.

    Applies condition-specific temporal augmentations:
    - normal:     gentle breathing, steady pulse
    - abnormal:   irregular breathing, mild tremor, variable saturation
    - infarction: rapid tremor, fast pulse, skin pallor, strong grimace
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    img = cv2.resize(img, (target_size, target_size))
    h, w = target_size, target_size
    center = (w // 2, h // 2)

    clip_name = image_path.stem
    clip_dir = output_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        t = i / num_frames

        if label == "infarction":
            angle = np.sin(t * 8 * np.pi) * 3.0
            scale = 1.0 + np.sin(t * 12 * np.pi) * 0.03
            brightness = -10 + np.sin(t * 16 * np.pi) * 8
            tx = np.sin(t * 6 * np.pi) * 3
            ty = np.cos(t * 4 * np.pi) * 2
            sat_factor = 0.85
        elif label == "abnormal":
            angle = np.sin(t * 4 * np.pi) * 1.5
            scale = 1.0 + np.sin(t * 6 * np.pi) * 0.02
            brightness = np.sin(t * 8 * np.pi) * 5
            tx = np.sin(t * 3 * np.pi) * 1.5
            ty = np.cos(t * 2 * np.pi) * 1
            sat_factor = 0.92
        else:
            angle = np.sin(t * 2 * np.pi) * 0.5
            scale = 1.0 + np.sin(t * 3 * np.pi) * 0.008
            brightness = np.sin(t * 4 * np.pi) * 2
            tx = np.sin(t * 1.5 * np.pi) * 0.5
            ty = 0
            sat_factor = 1.0

        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        warped = cv2.convertScaleAbs(warped, alpha=1.0, beta=brightness)

        if sat_factor != 1.0:
            hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= sat_factor
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            warped = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(str(clip_dir / f"frame_{i:02d}.jpg"), warped)

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-videos from static images")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--label", type=str, choices=LABELS, default=None)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    labels = [args.label] if args.label else LABELS

    total = 0
    for label in labels:
        label_in = input_root / label
        label_out = output_root / label

        if not label_in.exists():
            logger.warning(f"Skipping {label}: {label_in} not found")
            continue

        label_out.mkdir(parents=True, exist_ok=True)
        images = [f for f in label_in.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        logger.info(f"[{label}] Processing {len(images)} images...")

        for img_path in tqdm(images, desc=label):
            if generate_pseudo_clip(img_path, label_out, label, args.frames, args.size):
                total += 1

    logger.info(f"✅ Generated {total} pseudo-video clips → {output_root}")
    logger.info(f"Train: python scripts/train_3dcnn.py --data-dir {output_root}")


if __name__ == "__main__":
    main()
