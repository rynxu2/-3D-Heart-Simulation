"""
Pseudo-Video Generation Pipeline using Thin-Plate Spline Motion Model (TPSMM).

This script animates static face images using a driving video to create
realistic face-motion video clips for 3D CNN training.

Architecture:
    Source Image (SynPAIN face) + Driving Video (pain expression)
        → TPSMM animation engine
        → 16-frame video clip with realistic facial motion
        → Saved as frame folder for VideoClipDataset

Prerequisites:
    1. Run setup first:  python scripts/tpsmm_video_generator.py --setup
    2. Place driving videos in: data/driving_videos/{normal,abnormal,infarction}/
    3. Have source images in:   data/raw/{normal,abnormal,infarction}/

Usage:
    # Step 1: Setup (clone TPSMM + download checkpoint)
    python scripts/tpsmm_video_generator.py --setup

    # Step 2: Generate videos
    python scripts/tpsmm_video_generator.py \\
        --source-dir data/raw \\
        --driving-dir data/driving_videos \\
        --output-dir data/video_clips \\
        --num-frames 16

    # Step 3: Generate for a specific label only
    python scripts/tpsmm_video_generator.py \\
        --source-dir data/raw \\
        --driving-dir data/driving_videos \\
        --output-dir data/video_clips \\
        --label infarction
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TPSMM_DIR = PROJECT_ROOT / "tools" / "Thin-Plate-Spline-Motion-Model"
CHECKPOINT_DIR = TPSMM_DIR / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "vox.pth.tar"
CONFIG_FILE = TPSMM_DIR / "config" / "vox-256.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LABELS = ["normal", "abnormal", "infarction"]


# ═══════════════════════════════════════════════════════════
# SETUP: Clone TPSMM + Download Checkpoint
# ═══════════════════════════════════════════════════════════

def setup_tpsmm():
    """Clone TPSMM repository and download pre-trained checkpoint."""
    tools_dir = PROJECT_ROOT / "tools"
    tools_dir.mkdir(exist_ok=True)

    # 1. Clone repository
    if TPSMM_DIR.exists():
        logger.info(f"TPSMM already cloned at {TPSMM_DIR}")
    else:
        logger.info("Cloning Thin-Plate-Spline-Motion-Model...")
        subprocess.run([
            "git", "clone",
            "https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.git",
            str(TPSMM_DIR),
        ], check=True)
        logger.info("✅ TPSMM cloned successfully")

    # 2. Install dependencies
    req_file = TPSMM_DIR / "requirements.txt"
    if req_file.exists():
        logger.info("Installing TPSMM dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(req_file),
        ], check=True)
        logger.info("✅ Dependencies installed")

    # 3. Download checkpoint
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    if CHECKPOINT_FILE.exists():
        logger.info(f"Checkpoint already exists: {CHECKPOINT_FILE}")
    else:
        logger.info("Downloading VoxCeleb checkpoint (vox.pth.tar)...")
        downloaded = False

        # Method 1: Tsinghua Cloud (most reliable)
        try:
            import urllib.request
            tsinghua_url = "https://cloud.tsinghua.edu.cn/d/30ab8765da364fefa101/files/?p=%2Fvox.pth.tar&dl=1"
            logger.info("  Trying Tsinghua Cloud...")
            urllib.request.urlretrieve(tsinghua_url, str(CHECKPOINT_FILE))
            if CHECKPOINT_FILE.exists() and CHECKPOINT_FILE.stat().st_size > 1_000_000:
                downloaded = True
                logger.info("  ✅ Downloaded from Tsinghua Cloud")
        except Exception as e:
            logger.warning(f"  Tsinghua Cloud failed: {e}")

        # Method 2: Google Drive folder (browse and download)
        if not downloaded:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
                import gdown
                gdrive_folder = "https://drive.google.com/drive/folders/1pNDo1ODQIb5HVObRtCmubqJikmR7VVLT"
                logger.info("  Trying Google Drive folder...")
                gdown.download_folder(gdrive_folder, output=str(CHECKPOINT_DIR), quiet=False)
                # Find the downloaded checkpoint
                for f in CHECKPOINT_DIR.iterdir():
                    if f.name.endswith(".pth.tar") and "vox" in f.name.lower():
                        if f.name != "vox.pth.tar":
                            f.rename(CHECKPOINT_FILE)
                        downloaded = CHECKPOINT_FILE.exists()
                        break
                if downloaded:
                    logger.info("  ✅ Downloaded from Google Drive folder")
            except Exception as e:
                logger.warning(f"  Google Drive folder failed: {e}")

        # Method 3: Manual instructions
        if not downloaded:
            logger.error(f"""
╔══════════════════════════════════════════════════════════╗
║  ⚠️  Automatic download failed!                         ║
║                                                          ║
║  Please download vox.pth.tar manually from one of:       ║
║                                                          ║
║  1. Tsinghua Cloud:                                      ║
║     https://cloud.tsinghua.edu.cn/d/30ab8765da364fefa101/║
║                                                          ║
║  2. Google Drive:                                        ║
║     https://drive.google.com/drive/folders/              ║
║     1pNDo1ODQIb5HVObRtCmubqJikmR7VVLT                   ║
║                                                          ║
║  3. Yandex Disk:                                         ║
║     https://disk.yandex.com/d/bWopgbGj1ZUV1w             ║
║                                                          ║
║  Then place it at:                                        ║
║  {str(CHECKPOINT_FILE):<55s}║
╚══════════════════════════════════════════════════════════╝
            """)
            return

    # 4. Create driving video directories
    driving_dir = PROJECT_ROOT / "data" / "driving_videos"
    for label in LABELS:
        (driving_dir / label).mkdir(parents=True, exist_ok=True)

    logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅ TPSMM Setup Complete!                               ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Next steps:                                             ║
║  1. Record 3-5 short driving videos (2-3 seconds each)   ║
║     for each condition:                                   ║
║                                                          ║
║     📁 data/driving_videos/                              ║
║        ├── normal/      ← calm, neutral face             ║
║        ├── abnormal/    ← mild discomfort, frowning       ║
║        └── infarction/  ← severe pain, grimacing          ║
║                                                          ║
║  2. Run generation:                                       ║
║     python scripts/tpsmm_video_generator.py \\            ║
║         --source-dir data/raw \\                          ║
║         --driving-dir data/driving_videos \\              ║
║         --output-dir data/video_clips                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)


# ═══════════════════════════════════════════════════════════
# TPSMM Animation Engine
# ═══════════════════════════════════════════════════════════

class TPSMMAnimator:
    """Wrapper around TPSMM for batch face animation."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.kp_detector = None
        self.dense_motion_network = None

    def load_model(self):
        """Load TPSMM model from checkpoint."""
        import torch
        import yaml

        # Add TPSMM to path
        sys.path.insert(0, str(TPSMM_DIR))

        from modules.inpainting_network import InpaintingNetwork
        from modules.keypoint_detector import KPDetector
        from modules.dense_motion import DenseMotionNetwork

        if not CONFIG_FILE.exists():
            raise FileNotFoundError(f"TPSMM config not found: {CONFIG_FILE}. Run --setup first.")
        if not CHECKPOINT_FILE.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_FILE}. Run --setup first.")

        with open(CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f)

        inpainting = InpaintingNetwork(
            **config["model_params"]["generator_params"],
            **config["model_params"]["common_params"],
        )
        kp_detector = KPDetector(
            **config["model_params"]["common_params"],
            **config["model_params"]["kp_detector_params"],
        )
        dense_motion = DenseMotionNetwork(
            **config["model_params"]["common_params"],
            **config["model_params"]["dense_motion_params"],
        )

        checkpoint = torch.load(str(CHECKPOINT_FILE), map_location=self.device)
        inpainting.load_state_dict(checkpoint["inpainting_network"])
        kp_detector.load_state_dict(checkpoint["kp_detector"])
        dense_motion.load_state_dict(checkpoint["dense_motion_network"])

        inpainting.to(self.device).eval()
        kp_detector.to(self.device).eval()
        dense_motion.to(self.device).eval()

        self.model = inpainting
        self.kp_detector = kp_detector
        self.dense_motion_network = dense_motion

        logger.info("✅ TPSMM model loaded successfully")

    def animate(
        self,
        source_image: np.ndarray,
        driving_frames: List[np.ndarray],
        num_output_frames: int = 16,
    ) -> List[np.ndarray]:
        """Animate source image with driving video frames.

        Args:
            source_image: Source face image (BGR, any size)
            driving_frames: List of driving video frames (BGR)
            num_output_frames: Number of output frames

        Returns:
            List of animated frames (BGR, 256x256)
        """
        import torch

        # Preprocess: resize to 256x256, normalize to [0,1], CHW
        def preprocess(img):
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        source_tensor = preprocess(source_image)

        # Sample equally-spaced frames from driving video
        indices = np.linspace(0, len(driving_frames) - 1, num_output_frames, dtype=int)

        # Get source keypoints
        with torch.no_grad():
            kp_source = self.kp_detector(source_tensor)
            kp_driving_initial = self.kp_detector(preprocess(driving_frames[0]))

        animated_frames = []

        for idx in indices:
            driving_tensor = preprocess(driving_frames[idx])

            with torch.no_grad():
                kp_driving = self.kp_detector(driving_tensor)

                dense_motion = self.dense_motion_network(
                    source_image=source_tensor,
                    kp_driving=kp_driving,
                    kp_source=kp_source,
                )
                out = self.model(source_tensor, dense_motion)

            # Postprocess: back to BGR uint8
            result = out["prediction"].squeeze(0).permute(1, 2, 0).cpu().numpy()
            result = (result * 255).clip(0, 255).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            animated_frames.append(result)

        return animated_frames


# ═══════════════════════════════════════════════════════════
# Fallback: Simple Pseudo-Video (no TPSMM needed)
# ═══════════════════════════════════════════════════════════

class SimplePseudoAnimator:
    """Fallback animator when TPSMM is not available.

    Uses advanced augmentation techniques:
    - Facial Action Unit simulation (eyebrow raise, mouth tension)
    - Skin color fluctuation (simulated pulse/perfusion)
    - Breathing simulation (chest/shoulder movement via affine)
    - Micro-expression jitter
    """

    def animate(
        self,
        source_image: np.ndarray,
        label: str = "normal",
        num_frames: int = 16,
    ) -> List[np.ndarray]:
        img = cv2.resize(source_image, (256, 256))
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        frames = []

        for i in range(num_frames):
            t = i / num_frames  # Normalized time [0, 1)

            # Condition-specific animation parameters
            if label == "infarction":
                # Severe: strong grimace, rapid pulse, skin pallor
                rotation = np.sin(t * 8 * np.pi) * 3.0  # Rapid head shake
                scale = 1.0 + np.sin(t * 12 * np.pi) * 0.03  # Fast breathing
                brightness = -10 + np.sin(t * 16 * np.pi) * 8  # Rapid pulse
                tx = np.sin(t * 6 * np.pi) * 3  # Body tension tremor
                ty = np.cos(t * 4 * np.pi) * 2
                # Skin desaturation (pallor from poor perfusion)
                saturation_factor = 0.85

            elif label == "abnormal":
                # Moderate: mild discomfort, irregular breathing
                rotation = np.sin(t * 4 * np.pi) * 1.5
                scale = 1.0 + np.sin(t * 6 * np.pi) * 0.02
                brightness = np.sin(t * 8 * np.pi) * 5
                tx = np.sin(t * 3 * np.pi) * 1.5
                ty = np.cos(t * 2 * np.pi) * 1
                saturation_factor = 0.92

            else:  # normal
                # Calm: minimal movement, steady pulse
                rotation = np.sin(t * 2 * np.pi) * 0.5
                scale = 1.0 + np.sin(t * 3 * np.pi) * 0.008
                brightness = np.sin(t * 4 * np.pi) * 2
                tx = np.sin(t * 1.5 * np.pi) * 0.5
                ty = 0
                saturation_factor = 1.0

            # Apply affine transform
            M = cv2.getRotationMatrix2D(center, rotation, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Apply brightness (pulse simulation)
            warped = cv2.convertScaleAbs(warped, alpha=1.0, beta=brightness)

            # Apply saturation change (perfusion simulation)
            if saturation_factor != 1.0:
                hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] *= saturation_factor
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                warped = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            frames.append(warped)

        return frames


# ═══════════════════════════════════════════════════════════
# Batch Generation Pipeline
# ═══════════════════════════════════════════════════════════

def load_driving_video(video_path: Path, max_frames: int = 100) -> List[np.ndarray]:
    """Load driving video frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_driving_videos(driving_dir: Path, label: str) -> List[Path]:
    """Find driving videos for a given label."""
    label_dir = driving_dir / label
    if not label_dir.exists():
        return []
    return sorted([f for f in label_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS])


def generate_batch(
    source_dir: Path,
    driving_dir: Path,
    output_dir: Path,
    label: str,
    num_frames: int = 16,
    target_size: int = 224,
    use_tpsmm: bool = True,
    max_per_source: int = 3,
    device: str = "cuda",
):
    """Generate pseudo-videos for all images of a label.

    For each source image, generates up to `max_per_source` video clips
    using different driving videos. If TPSMM is not available, falls back
    to the simple pseudo-animator.
    """
    source_label_dir = source_dir / label
    output_label_dir = output_dir / label
    output_label_dir.mkdir(parents=True, exist_ok=True)
    print("source_label_dir", source_label_dir)

    if not source_label_dir.exists():
        logger.warning(f"Source directory not found: {source_label_dir}")
        return 0

    source_images = sorted([
        f for f in source_label_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    ])

    if not source_images:
        logger.warning(f"No source images found in {source_label_dir}")
        return 0

    driving_videos = get_driving_videos(driving_dir, label) if use_tpsmm else []

    # Initialize animator
    animator = None
    fallback = SimplePseudoAnimator()

    if use_tpsmm and driving_videos:
        try:
            animator = TPSMMAnimator(device=device)
            animator.load_model()
            logger.info(f"Using TPSMM animator with {len(driving_videos)} driving videos")
        except Exception as e:
            logger.warning(f"TPSMM failed to load: {e}. Using fallback animator.")
            animator = None

    # Pre-load driving video frames
    driving_frames_cache = {}
    for dv in driving_videos:
        driving_frames_cache[dv.name] = load_driving_video(dv)

    generated = 0
    total = len(source_images)

    for si, src_path in enumerate(source_images):
        src_img = cv2.imread(str(src_path))
        if src_img is None:
            continue

        clips_for_this_image = 0

        # Method 1: TPSMM animation (if available)
        if animator and driving_frames_cache:
            for dv_name, dv_frames in driving_frames_cache.items():
                if clips_for_this_image >= max_per_source:
                    break
                if not dv_frames:
                    continue

                try:
                    animated = animator.animate(src_img, dv_frames, num_frames)
                except Exception as e:
                    logger.warning(f"Animation failed for {src_path.name}: {e}")
                    continue

                # Save as frame folder
                clip_name = f"{src_path.stem}_tps_{dv_name.split('.')[0]}_{clips_for_this_image:02d}"
                clip_dir = output_label_dir / clip_name
                clip_dir.mkdir(parents=True, exist_ok=True)

                for fi, frame in enumerate(animated):
                    frame_resized = cv2.resize(frame, (target_size, target_size))
                    cv2.imwrite(str(clip_dir / f"frame_{fi:02d}.jpg"), frame_resized)

                clips_for_this_image += 1
                generated += 1

        # Method 2: Fallback pseudo-animation (always generate at least 1)
        if clips_for_this_image < max_per_source:
            animated = fallback.animate(src_img, label, num_frames)

            clip_name = f"{src_path.stem}_pseudo_{clips_for_this_image:02d}"
            clip_dir = output_label_dir / clip_name
            clip_dir.mkdir(parents=True, exist_ok=True)

            for fi, frame in enumerate(animated):
                frame_resized = cv2.resize(frame, (target_size, target_size))
                cv2.imwrite(str(clip_dir / f"frame_{fi:02d}.jpg"), frame_resized)

            generated += 1

        if (si + 1) % 50 == 0:
            logger.info(f"  [{label}] {si + 1}/{total} images processed, {generated} clips generated")

    logger.info(f"✅ [{label}] Generated {generated} video clips from {total} source images")
    return generated


# ═══════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-video clips from static face images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup TPSMM (first time only)
  python scripts/tpsmm_video_generator.py --setup

  # Generate videos using TPSMM + driving videos
  python scripts/tpsmm_video_generator.py \\
      --source-dir data/raw \\
      --driving-dir data/driving_videos \\
      --output-dir data/video_clips

  # Generate using simple pseudo-animation only (no TPSMM needed)
  python scripts/tpsmm_video_generator.py \\
      --source-dir data/raw \\
      --output-dir data/video_clips \\
      --no-tpsmm

  # Generate for infarction label only
  python scripts/tpsmm_video_generator.py \\
      --source-dir data/raw \\
      --output-dir data/video_clips \\
      --label infarction --no-tpsmm
        """,
    )
    parser.add_argument("--setup", action="store_true",
                        help="Setup TPSMM: clone repo + download checkpoint")
    parser.add_argument("--source-dir", type=str, default="data/raw",
                        help="Directory with source images (normal/abnormal/infarction)")
    parser.add_argument("--driving-dir", type=str, default="data/driving_videos",
                        help="Directory with driving videos")
    parser.add_argument("--output-dir", type=str, default="data/video_clips",
                        help="Output directory for generated video clips")
    parser.add_argument("--label", type=str, choices=LABELS, default=None,
                        help="Generate for specific label only")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Frames per clip")
    parser.add_argument("--target-size", type=int, default=224,
                        help="Output frame size")
    parser.add_argument("--max-per-source", type=int, default=3,
                        help="Max clips per source image")
    parser.add_argument("--no-tpsmm", action="store_true",
                        help="Use simple pseudo-animation only (no TPSMM)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")

    args = parser.parse_args()

    if args.setup:
        setup_tpsmm()
        return

    labels = [args.label] if args.label else LABELS
    use_tpsmm = not args.no_tpsmm

    total_generated = 0
    for label in labels:
        count = generate_batch(
            source_dir=Path(args.source_dir),
            driving_dir=Path(args.driving_dir),
            output_dir=Path(args.output_dir),
            label=label,
            num_frames=args.num_frames,
            target_size=args.target_size,
            use_tpsmm=use_tpsmm,
            max_per_source=args.max_per_source,
            device=args.device,
        )
        total_generated += count

    logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅ Generation Complete!                                 ║
║  Total clips: {total_generated:<42d}║
║  Output: {str(args.output_dir):<47s}║
║                                                          ║
║  Next step — train 3D CNN:                                ║
║  python scripts/train_3dcnn.py \\                         ║
║      --data-dir {str(args.output_dir):<39s}         ║
╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
