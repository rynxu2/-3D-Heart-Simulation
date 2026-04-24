"""SynPAIN training script — 2-phase transfer learning on Pain/NoPain classification."""

import sys
import json
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.config import SynPainConfig, MODELS_DIR
from src.data.synpain_loader import create_synpain_loaders
from src.models.classifier_2d import create_classifier
from src.models.trainer import Trainer


def main():
    logger.info("=" * 60)
    logger.info("SynPAIN Training — 2-Phase Transfer Learning")
    logger.info("=" * 60)

    # 1. Load config
    config = SynPainConfig.from_yaml("synpain_config.yaml")
    logger.info(f"Config: backbone={config.backbone}, classes={config.num_classes}, "
                f"strategy={config.label_strategy}, epochs={config.epochs}")

    # 2. Load dataset
    train_loader, val_loader, test_loader, dataset_info = create_synpain_loaders(
        cache_dir=config.cache_dir,
        strategy=config.label_strategy,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
    )

    logger.info(f"Dataset: {dataset_info}")

    # 3. Create model
    model = create_classifier(
        backbone=config.backbone,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout=config.dropout,
    )

    # 4. Train with 2-phase approach
    trainer = Trainer(
        model=model,
        config=config,
        class_weights=dataset_info.get("class_weights"),
    )

    history = trainer.train_phases(
        train_loader=train_loader,
        val_loader=val_loader,
        freeze_epochs=config.freeze_epochs,
        phase2_lr=config.learning_rate,
    )

    # 5. Save training history
    results_dir = MODELS_DIR / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete! Checkpoint: models/checkpoints/best_model.pth")
    logger.info(f"Run evaluation: python scripts/synpain_eval.py")


if __name__ == "__main__":
    main()
