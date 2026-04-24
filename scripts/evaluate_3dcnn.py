"""Evaluate 3D CNN model and compare with 2D CNN."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from loguru import logger

from src.config import Config, MODELS_DIR
from src.data.video_dataset import create_video_dataloaders, LABEL_NAMES
from src.models.classifier_3d import HeartFaceClassifier3D


def evaluate(model, loader, device, label_names):
    """Run evaluation and collect predictions."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=[label_names[i] for i in sorted(label_names.keys())],
    )
    logger.info(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"Confusion Matrix:\n{cm}")

    # ROC-AUC (multi-class)
    try:
        if len(label_names) > 2:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        else:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        logger.info(f"ROC-AUC: {auc:.4f}")
    except ValueError as e:
        logger.warning(f"Cannot compute ROC-AUC: {e}")

    accuracy = np.mean(all_preds == all_labels) * 100
    logger.info(f"Accuracy: {accuracy:.1f}%")

    return {"accuracy": accuracy, "report": report, "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D CNN model")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/3dcnn_best.pth")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    model = HeartFaceClassifier3D.from_checkpoint(ckpt_path, device=device)
    logger.info(f"Model loaded from {ckpt_path}")

    # Load test data
    _, _, test_loader, info = create_video_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
    )
    logger.info(f"Test set: {info['test_size']} clips")

    # Evaluate
    results = evaluate(model, test_loader, device, LABEL_NAMES)

    # Save results
    output_dir = MODELS_DIR / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "3dcnn_report.txt", "w") as f:
        f.write(results["report"])

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
