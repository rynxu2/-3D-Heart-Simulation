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

    def compute_demographic_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        demographics: List[dict],
    ) -> dict:
        """Compute accuracy per demographic subgroup for bias analysis."""
        groups = {"gender": {}, "age": {}, "ethnicity": {}}

        for i, meta in enumerate(demographics):
            if not meta.get("valid", False):
                continue

            for key in ["gender", "age", "ethnicity"]:
                group_val = meta.get(key, "Unknown")
                if group_val not in groups[key]:
                    groups[key][group_val] = {"correct": 0, "total": 0}
                groups[key][group_val]["total"] += 1
                if y_true[i] == y_pred[i]:
                    groups[key][group_val]["correct"] += 1

        results = {}
        for key, subgroups in groups.items():
            results[key] = {}
            for group_name, counts in subgroups.items():
                acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
                results[key][group_name] = {
                    "accuracy": round(acc, 4),
                    "total": counts["total"],
                }

        # Compute max accuracy gap per dimension
        for key in results:
            accs = [v["accuracy"] for v in results[key].values()]
            results[f"{key}_gap"] = round(max(accs) - min(accs), 4) if accs else 0

        return results

    def plot_demographic_bias(self, demographic_results: dict, save_path: Optional[str] = None):
        """Plot bar chart of accuracy per demographic subgroup."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

        for ax, key in zip(axes, ["gender", "age", "ethnicity"]):
            if key not in demographic_results:
                continue
            data = demographic_results[key]
            names = list(data.keys())
            accs = [data[n]["accuracy"] * 100 for n in names]
            totals = [data[n]["total"] for n in names]

            bars = ax.bar(names, accs, color=colors[: len(names)], edgecolor="white", linewidth=1.5)
            ax.set_title(f"Accuracy by {key.title()}", fontsize=14, fontweight="bold")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 105)
            ax.axhline(y=np.mean(accs), color="gray", linestyle="--", alpha=0.5, label=f"Mean: {np.mean(accs):.1f}%")
            ax.legend()

            for bar, total in zip(bars, totals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"n={total}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Demographic bias chart saved: {save_path}")
        return fig
