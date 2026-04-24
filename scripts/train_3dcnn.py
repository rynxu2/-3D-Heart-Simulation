"""3D CNN Training Script — train ResNet3D-18 on face video clips.

Usage:
    python scripts/train_3dcnn.py
    python scripts/train_3dcnn.py --config configs/train_3dcnn_config.yaml
    python scripts/train_3dcnn.py --epochs 5 --batch-size 4  # quick test
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger

from src.config import Config, CONFIGS_DIR, MODELS_DIR
from src.models.classifier_3d import HeartFaceClassifier3D
from src.models.trainer import Trainer
from src.data.video_dataset import create_video_dataloaders


def load_3dcnn_config(config_path: str = "train_3dcnn_config.yaml") -> dict:
    """Load 3D CNN config from YAML."""
    path = CONFIGS_DIR / config_path
    if not path.exists():
        logger.warning(f"Config not found: {path}, using defaults")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train 3D CNN for heart condition classification")
    parser.add_argument("--config", type=str, default="train_3dcnn_config.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    raw = load_3dcnn_config(args.config)
    model_cfg = raw.get("model", {})
    data_cfg = raw.get("data", {})
    train_cfg = raw.get("training", {})

    # Override with CLI args
    data_dir = args.data_dir or data_cfg.get("data_dir", "data/raw")
    epochs = args.epochs or train_cfg.get("epochs", 40)
    batch_size = args.batch_size or data_cfg.get("batch_size", 8)
    lr = args.lr or train_cfg.get("learning_rate", 5e-5)
    num_frames = model_cfg.get("num_frames", 16)
    freeze_epochs = model_cfg.get("freeze_epochs", 5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataloaders
    logger.info(f"Loading video dataset from: {data_dir}")
    train_loader, val_loader, test_loader, info = create_video_dataloaders(
        data_dir=data_dir,
        num_frames=num_frames,
        image_size=data_cfg.get("image_size", 224),
        batch_size=batch_size,
        num_workers=data_cfg.get("num_workers", 2),
    )
    logger.info(f"Dataset: {info}")

    # Create model
    backbone = model_cfg.get("backbone", "r3d_18")
    num_classes = model_cfg.get("num_classes", 3)

    if args.checkpoint:
        model = HeartFaceClassifier3D.from_checkpoint(args.checkpoint, device=device)
    else:
        model = HeartFaceClassifier3D(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
            dropout=model_cfg.get("dropout", 0.3),
            num_frames=num_frames,
        )

    # Create config for trainer
    config = Config(
        backbone=backbone,
        num_classes=num_classes,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 10),
        mixed_precision=train_cfg.get("mixed_precision", True),
        batch_size=batch_size,
    )

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        class_weights=info.get("class_weights"),
    )

    logger.info(f"Starting 2-phase training: {freeze_epochs} frozen + {epochs - freeze_epochs} fine-tune")
    history = trainer.train_phases(
        train_loader=train_loader,
        val_loader=val_loader,
        freeze_epochs=freeze_epochs,
        phase2_lr=lr,
    )

    # Save final checkpoint
    ckpt_path = MODELS_DIR / "checkpoints" / "3dcnn_best.pth"
    logger.info(f"Best checkpoint saved at: {ckpt_path}")

    # Quick evaluation
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader, epochs)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.1f}%")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
