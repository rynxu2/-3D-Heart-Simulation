"""Training pipeline with mixed precision, early stopping, and TensorBoard logging."""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional
from loguru import logger

from src.config import Config, MODELS_DIR


class Trainer:
    """Training loop with mixed precision, early stopping, checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        class_weights: Optional[list] = None,
    ):
        self.config = config or Config.from_yaml()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Loss with optional class weights
        weight_tensor = None
        if class_weights:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)

        # Mixed precision
        self.scaler = GradScaler(enabled=self.config.mixed_precision)

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Logging
        self.log_dir = MODELS_DIR / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

        # Checkpoint dir
        self.ckpt_dir = MODELS_DIR / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device, enabled=self.config.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/train", accuracy, epoch)

        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        self.writer.add_scalar("Loss/val", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/val", accuracy, epoch)

        return {"loss": avg_loss, "accuracy": accuracy}

    def save_checkpoint(self, epoch: int, val_metrics: dict, filename: str = "best_model.pth"):
        path = self.ckpt_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def train_phases(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        freeze_epochs: int = 5,
        phase2_lr: float = 5e-5,
    ) -> dict:
        """2-phase training: freeze backbone → unfreeze and fine-tune.

        Phase 1: Train only classifier head (frozen backbone)
        Phase 2: Unfreeze all layers, fine-tune with lower LR
        """
        logger.info(f"=== Phase 1: Frozen backbone ({freeze_epochs} epochs) ===")
        if hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, freeze_epochs + 1):
            import time
            start = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            self.scheduler.step()
            elapsed = time.time() - start

            logger.info(
                f"[P1] Epoch {epoch}/{freeze_epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.1f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.1f}%"
            )
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint(epoch, val_metrics)

        # Phase 2: Unfreeze
        logger.info(f"=== Phase 2: Full fine-tuning ({self.config.epochs - freeze_epochs} epochs) ===")
        if hasattr(self.model, "unfreeze_backbone"):
            self.model.unfreeze_backbone()

        # Reset optimizer with lower LR
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = phase2_lr

        self.patience_counter = 0
        remaining = self.config.epochs - freeze_epochs

        for epoch in range(freeze_epochs + 1, self.config.epochs + 1):
            import time
            start = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            self.scheduler.step()
            elapsed = time.time() - start

            logger.info(
                f"[P2] Epoch {epoch}/{self.config.epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.1f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.1f}%"
            )
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.warning(f"Early stopping at epoch {epoch}")
                break

        self.writer.close()
        logger.info(f"2-phase training complete. Best val loss: {self.best_val_loss:.4f}")
        return history

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Full training loop."""
        logger.info(f"Training on {self.device} for {self.config.epochs} epochs")
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.config.epochs + 1):
            start = time.time()

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            self.scheduler.step()

            elapsed = time.time() - start

            logger.info(
                f"Epoch {epoch}/{self.config.epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.1f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.1f}%"
            )

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            # Best model checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.warning(f"Early stopping at epoch {epoch} (patience={self.config.early_stopping_patience})")
                break

        self.writer.close()
        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        return history
