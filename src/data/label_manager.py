"""Label manager — organize, validate, and export dataset labels."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from loguru import logger


VALID_LABELS = {"normal", "abnormal", "infarction"}


class LabelManager:
    """Manage dataset labels: create CSV, validate, balance stats."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.labels_csv = self.data_dir / "labels.csv"
        self.labels: List[Dict] = []

    def scan_directory(self) -> List[Dict]:
        """Scan data directory and build label list from folder structure.

        Expected: data_dir/normal/*.jpg, data_dir/abnormal/*.jpg, etc.
        """
        self.labels = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for label_name in VALID_LABELS:
            label_dir = self.data_dir / label_name
            if not label_dir.exists():
                logger.warning(f"Directory not found: {label_dir}")
                continue

            for img_file in sorted(label_dir.iterdir()):
                if img_file.suffix.lower() in exts:
                    self.labels.append({
                        "filename": str(img_file.relative_to(self.data_dir)),
                        "filepath": str(img_file),
                        "label": label_name,
                    })

        logger.info(f"Scanned {len(self.labels)} images")
        return self.labels

    def export_csv(self, output_path: Optional[str | Path] = None) -> Path:
        """Export labels to CSV file."""
        if not self.labels:
            self.scan_directory()

        output = Path(output_path) if output_path else self.labels_csv
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "filepath", "label"])
            writer.writeheader()
            writer.writerows(self.labels)

        logger.info(f"Labels exported: {output} ({len(self.labels)} entries)")
        return output

    def load_csv(self, csv_path: Optional[str | Path] = None) -> List[Dict]:
        """Load labels from CSV."""
        path = Path(csv_path) if csv_path else self.labels_csv
        self.labels = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.labels = list(reader)

        logger.info(f"Loaded {len(self.labels)} labels from {path}")
        return self.labels

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if not self.labels:
            self.scan_directory()

        counts = Counter(entry["label"] for entry in self.labels)
        total = len(self.labels)

        stats = {
            "total": total,
            "per_class": dict(counts),
            "balance_ratio": min(counts.values()) / max(counts.values()) if counts else 0,
            "is_balanced": len(set(counts.values())) == 1 if counts else False,
        }

        for label in VALID_LABELS:
            if label not in stats["per_class"]:
                stats["per_class"][label] = 0

        return stats

    def validate(self) -> Dict:
        """Validate dataset: check files exist, labels valid, minimum counts."""
        if not self.labels:
            self.scan_directory()

        issues = []
        missing_files = 0
        invalid_labels = 0

        for entry in self.labels:
            if not Path(entry["filepath"]).exists():
                missing_files += 1
                issues.append(f"Missing: {entry['filepath']}")
            if entry["label"] not in VALID_LABELS:
                invalid_labels += 1
                issues.append(f"Invalid label '{entry['label']}' for {entry['filename']}")

        stats = self.get_stats()
        for label, count in stats["per_class"].items():
            if count < 100:
                issues.append(f"Low count for '{label}': {count} (recommend ≥500)")

        result = {
            "valid": len(issues) == 0,
            "total_images": len(self.labels),
            "missing_files": missing_files,
            "invalid_labels": invalid_labels,
            "issues": issues[:20],  # Cap at 20
            "stats": stats,
        }

        if result["valid"]:
            logger.info("Dataset validation passed ✅")
        else:
            logger.warning(f"Dataset has {len(issues)} issues ⚠️")

        return result

    def get_split_indices(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Get stratified train/val/test split indices."""
        from sklearn.model_selection import train_test_split

        if not self.labels:
            self.scan_directory()

        indices = list(range(len(self.labels)))
        labels = [entry["label"] for entry in self.labels]

        train_idx, temp_idx = train_test_split(
            indices, test_size=(1 - train_ratio), stratify=labels, random_state=42
        )
        temp_labels = [labels[i] for i in temp_idx]
        val_size = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1 - val_size), stratify=temp_labels, random_state=42
        )

        return train_idx, val_idx, test_idx
