"""Evaluation metrics for heart condition classification."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
from loguru import logger

from src.data.dataset import LABEL_NAMES


class ClassificationMetrics:
    """Compute and visualize classification metrics."""

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.label_names = [LABEL_NAMES[i] for i in range(num_classes)]

    def compute_all(self, y_true: List[int], y_pred: List[int], y_probs: Optional[np.ndarray] = None) -> dict:
        """Compute all classification metrics."""
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, target_names=self.label_names, zero_division=0
            ),
        }

        if y_probs is not None:
            results["roc_auc"] = self._compute_roc_auc(y_true, y_probs)

        logger.info(f"Accuracy: {results['accuracy']:.4f} | F1: {results['f1_macro']:.4f}")
        return results

    def _compute_roc_auc(self, y_true: List[int], y_probs: np.ndarray) -> dict:
        y_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        roc_data = {}
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc_val = auc(fpr, tpr)
            roc_data[self.label_names[i]] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": roc_auc_val,
            }
        return roc_data

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], save_path: Optional[str] = None):
        """Plot and optionally save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.label_names, yticklabels=self.label_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix - Heart Condition Classification")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Confusion matrix saved: {save_path}")

        return fig

    def plot_roc_curves(self, y_true: List[int], y_probs: np.ndarray, save_path: Optional[str] = None):
        """Plot ROC curves for each class."""
        y_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        for i, (name, color) in enumerate(zip(self.label_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves - Heart Condition Classification")
        ax.legend(loc="lower right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"ROC curves saved: {save_path}")

        return fig

    def plot_training_history(self, history: dict, save_path: Optional[str] = None):
        """Plot training loss and accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history["train_loss"], label="Train", color="#3498db")
        ax1.plot(history["val_loss"], label="Validation", color="#e74c3c")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history["train_acc"], label="Train", color="#3498db")
        ax2.plot(history["val_acc"], label="Validation", color="#e74c3c")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Training history saved: {save_path}")

        return fig
