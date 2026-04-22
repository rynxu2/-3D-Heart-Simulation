"""Evaluation visualizer — generate charts and reports for thesis."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class EvaluationVisualizer:
    """Generate publication-ready evaluation charts for graduation thesis."""

    def __init__(self, output_dir: str | Path = "data/outputs/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style
        plt.style.use("seaborn-v0_8-whitegrid")
        self.label_names = ["Bình thường", "Bất thường", "Nhồi máu cơ tim"]
        self.colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    def plot_class_distribution(self, class_counts: Dict[str, int], title: str = "Phân bố dữ liệu") -> Path:
        """Bar chart of dataset class distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = list(class_counts.keys())
        counts = list(class_counts.values())
        label_vi = {"normal": "Bình thường", "abnormal": "Bất thường", "infarction": "Nhồi máu cơ tim"}
        display_labels = [label_vi.get(l, l) for l in labels]

        bars = ax.bar(display_labels, counts, color=self.colors[:len(labels)], edgecolor="white", linewidth=1.5)

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(count), ha="center", va="bottom", fontweight="bold", fontsize=12)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Số lượng ảnh", fontsize=11)
        plt.tight_layout()

        path = self.output_dir / "class_distribution.png"
        plt.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict],
        metrics: List[str] = None,
    ) -> Path:
        """Grouped bar chart comparing 2D CNN vs 3D CNN performance."""
        if metrics is None:
            metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

        metric_labels = {
            "accuracy": "Accuracy",
            "precision_macro": "Precision",
            "recall_macro": "Recall",
            "f1_macro": "F1-Score",
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metrics))
        width = 0.35
        model_names = list(model_results.keys())

        colors_models = ["#3498db", "#e74c3c"]
        for i, (model_name, results) in enumerate(model_results.items()):
            values = [results.get(m, 0) for m in metrics]
            bars = ax.bar(x + i * width, values, width, label=model_name,
                          color=colors_models[i % 2], edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Metrics", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("So sánh hiệu suất: 2D CNN vs 3D CNN", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([metric_labels.get(m, m) for m in metrics])
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / "model_comparison.png"
        plt.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_per_class_metrics(self, metrics_result: Dict) -> Path:
        """Heatmap of per-class precision/recall/F1."""
        labels = self.label_names
        metric_names = ["Precision", "Recall", "F1-Score"]

        data = np.array([
            metrics_result.get("precision_per_class", [0, 0, 0]),
            metrics_result.get("recall_per_class", [0, 0, 0]),
            metrics_result.get("f1_per_class", [0, 0, 0]),
        ])

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            data, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=labels, yticklabels=metric_names,
            ax=ax, vmin=0, vmax=1, linewidths=1, linecolor="white",
        )
        ax.set_title("Metrics theo từng lớp", fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = self.output_dir / "per_class_metrics.png"
        plt.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_inference_time(self, times: Dict[str, float]) -> Path:
        """Bar chart of inference time per model/stage."""
        fig, ax = plt.subplots(figsize=(8, 5))

        stages = list(times.keys())
        values = list(times.values())

        bars = ax.barh(stages, values, color="#3498db", edgecolor="white", height=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}ms", va="center", fontsize=10)

        ax.set_xlabel("Thời gian (ms)", fontsize=11)
        ax.set_title("Thời gian Inference", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()

        path = self.output_dir / "inference_time.png"
        plt.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def generate_full_report(
        self,
        metrics_2d: Dict,
        metrics_3d: Dict,
        class_counts: Dict[str, int],
        inference_times: Dict[str, float],
    ) -> List[Path]:
        """Generate all evaluation charts for thesis."""
        paths = []

        paths.append(self.plot_class_distribution(class_counts))

        paths.append(self.plot_model_comparison({
            "2D CNN (EfficientNet-B4)": metrics_2d,
            "3D CNN (ResNet3D-18)": metrics_3d,
        }))

        paths.append(self.plot_per_class_metrics(metrics_2d))
        paths.append(self.plot_inference_time(inference_times))

        logger.info(f"Full report generated: {len(paths)} charts in {self.output_dir}")
        return paths
