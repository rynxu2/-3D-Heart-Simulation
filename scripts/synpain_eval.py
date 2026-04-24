"""SynPAIN evaluation script — metrics, visualizations, and demographic bias analysis."""

import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.config import SynPainConfig, MODELS_DIR
from src.data.synpain_loader import create_synpain_loaders, SYNPAIN_BINARY_NAMES, SYNPAIN_3CLASS_NAMES
from src.models.classifier_2d import HeartFaceClassifier2D
from src.evaluation.metrics import ClassificationMetrics


def load_model(checkpoint_path: str, config: SynPainConfig, device: str) -> HeartFaceClassifier2D:
    model = HeartFaceClassifier2D(
        backbone=config.backbone,
        num_classes=config.num_classes,
        pretrained=False,
        dropout=config.dropout,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {checkpoint_path} (val_acc={ckpt.get('val_accuracy', 'N/A')})")
    return model


def evaluate(model, test_loader, device: str):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    return all_labels, all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(description="Evaluate SynPAIN model")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="synpain_config.yaml")
    args = parser.parse_args()

    config = SynPainConfig.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load test data
    _, _, test_loader, dataset_info = create_synpain_loaders(
        cache_dir=config.cache_dir,
        strategy=config.label_strategy,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Evaluate
    y_true, y_pred, y_probs = evaluate(model, test_loader, device)

    # Metrics
    metrics = ClassificationMetrics(num_classes=config.num_classes)
    metrics.label_names = list(
        (SYNPAIN_BINARY_NAMES if config.label_strategy == "binary" else SYNPAIN_3CLASS_NAMES).values()
    )

    results = metrics.compute_all(y_true, y_pred, y_probs)
    logger.info(f"\n{results['classification_report']}")

    # Save outputs
    results_dir = MODELS_DIR / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(results["classification_report"])

    metrics.plot_confusion_matrix(y_true, y_pred, save_path=str(results_dir / "confusion_matrix.png"))
    metrics.plot_roc_curves(y_true, y_probs, save_path=str(results_dir / "roc_curves.png"))

    # Load and plot training history
    history_path = results_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        metrics.plot_training_history(history, save_path=str(results_dir / "training_history.png"))

    # Demographic bias analysis
    test_ds = test_loader.dataset
    if hasattr(test_ds, "get_demographics"):
        demographics = test_ds.get_demographics()
        demo_results = metrics.compute_demographic_metrics(y_true, y_pred, demographics)

        with open(results_dir / "demographic_bias_report.json", "w") as f:
            json.dump(demo_results, f, indent=2)

        metrics.plot_demographic_bias(demo_results, save_path=str(results_dir / "demographic_bias_chart.png"))
        logger.info(f"Demographic gaps — Gender: {demo_results.get('gender_gap', 0):.2%}, "
                     f"Age: {demo_results.get('age_gap', 0):.2%}, "
                     f"Ethnicity: {demo_results.get('ethnicity_gap', 0):.2%}")

    logger.info(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
