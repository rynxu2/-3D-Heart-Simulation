"""Interactive face data collection script.

Usage:
    # Capture images (2D CNN)
    python scripts/collect_faces.py --label normal --mode image --num 50

    # Capture video clips (3D CNN)
    python scripts/collect_faces.py --label infarction --mode video --num 20

    # Generate pseudo-videos from SynPAIN images
    python scripts/collect_faces.py --label normal --mode generate --source-dir path/to/synpain/normal

    # Generate for all labels at once
    python scripts/collect_faces.py --mode generate --source-dir data/raw

    # Import from folder
    python scripts/collect_faces.py --label abnormal --import-dir path/to/images

    # Check dataset stats
    python scripts/collect_faces.py --stats
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_collector import DataCollector
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Collect face data for heart condition classification")
    parser.add_argument("--label", choices=["normal", "abnormal", "infarction"],
                        help="Target label for collection")
    parser.add_argument("--mode", choices=["image", "video", "generate"], default="image",
                        help="'image' for 2D CNN, 'video' for webcam 3D clips, 'generate' for pseudo-videos from images")
    parser.add_argument("--source-dir", type=str,
                        help="Source folder with images for pseudo-video generation (generate mode)")
    parser.add_argument("--num", type=int, default=50,
                        help="Number of images/clips to capture")
    parser.add_argument("--duration", type=float, default=2.5,
                        help="Video clip duration in seconds (video mode only)")
    parser.add_argument("--frames", type=int, default=16,
                        help="Frames per clip (video mode only)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Output directory for collected data")
    parser.add_argument("--import-dir", type=str,
                        help="Import images from existing directory")
    parser.add_argument("--stats", action="store_true",
                        help="Show dataset statistics and exit")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Target image size (pixels)")

    args = parser.parse_args()

    collector = DataCollector(output_dir=args.output_dir, target_size=args.image_size)

    if args.stats:
        stats = collector.get_dataset_stats()
        print("\n📊 Dataset Statistics:")
        print("=" * 40)
        for label in ["normal", "abnormal", "infarction"]:
            info = stats.get(label, {"images": 0, "clips": 0})
            print(f"  {label:15s} | Images: {info['images']:4d} | Clips: {info['clips']:4d}")
        print("=" * 40)
        print(f"  Total: {stats['total']}")
        print(f"  Balanced: {'✅' if stats['balanced'] else '❌'}")
        return

    if args.import_dir:
        if not args.label:
            parser.error("--label is required when importing")
        count = collector.import_from_folder(args.import_dir, args.label)
        logger.info(f"Imported {count} images for '{args.label}'")
        return

    if args.mode == "generate":
        # Pseudo-video generation from static images
        source = Path(args.source_dir) if args.source_dir else Path(args.output_dir)
        labels = [args.label] if args.label else ["normal", "abnormal", "infarction"]
        total = 0
        for label in labels:
            src = source / label
            if not src.exists():
                logger.warning(f"Source dir not found: {src}")
                continue
            count = collector.generate_pseudo_videos(
                source_dir=src, label=label, num_frames=args.frames,
            )
            total += count
        logger.info(f"Total: {total} pseudo-video clips generated")
        return

    if not args.label:
        parser.error("--label is required for capture mode")

    if args.mode == "video":
        count = collector.capture_video_clip(
            label=args.label,
            duration_sec=args.duration,
            num_frames=args.frames,
            camera_id=args.camera,
        )
        logger.info(f"Captured {count} video clips for '{args.label}'")
    else:
        count = collector.capture_from_webcam(
            label=args.label,
            num_images=args.num,
            camera_id=args.camera,
        )
        logger.info(f"Captured {count} images for '{args.label}'")


if __name__ == "__main__":
    main()
