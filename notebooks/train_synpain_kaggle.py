# %% [markdown]
# # 🫀 SynPAIN Training — Heart Condition Detection
# **Strategy C: Transfer Learning** — Pre-train on Pain/NoPain, fine-tune later for 3-class heart conditions.
#
# Dataset: [TaatiTeam/SynPAIN](https://huggingface.co/datasets/TaatiTeam/SynPAIN) (10,710 synthetic facial images)
#
# **How to use on Kaggle:**
# 1. Create new Kaggle Notebook
# 2. Enable GPU (Settings → Accelerator → GPU T4 x2)
# 3. Copy-paste each cell or upload this file
# 4. Run all cells

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
# Install required packages
import subprocess
subprocess.run(["pip", "install", "-q", "datasets", "huggingface-hub", "albumentations", "loguru", "torchmetrics"], check=True)

# %%
import os
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torchvision import models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
from loguru import logger

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## 2. Configuration

# %%
CONFIG = {
    "dataset_name": "TaatiTeam/SynPAIN",
    "backbone": "efficientnet_b4",
    "num_classes": 2,
    "image_size": 224,
    "batch_size": 16,
    "num_workers": 2,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "total_epochs": 30,
    "freeze_epochs": 5,
    "phase1_lr": 1e-3,
    "phase2_lr": 5e-5,
    "weight_decay": 1e-5,
    "early_stopping_patience": 8,
    "dropout": 0.3,
    "mixed_precision": True,
}

LABEL_MAP = {"NoPain": 0, "Pain": 1}
LABEL_NAMES = {0: "NoPain", 1: "Pain"}
ETHNICITY_MAP = {"0": "Black", "1": "South Asian", "2": "Middle Eastern", "3": "Caucasian", "4": "East Asian"}

OUTPUT_DIR = Path("/kaggle/working" if os.path.exists("/kaggle") else "outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Config: {json.dumps(CONFIG, indent=2)}")

# %% [markdown]
# ## 3. Load SynPAIN Dataset

# %%
from datasets import load_dataset

logger.info("Downloading SynPAIN dataset...")
ds = load_dataset("TaatiTeam/SynPAIN", split="train")
print(f"Raw dataset: {len(ds)} entries")
print(f"Columns: {ds.column_names}")
print(f"Features: {ds.features}")
print(f"Label classes: {ds.features['label'].names}")

# %% [markdown]
# ## 4. Parse Filenames & Extract Labels

# %%
def parse_synpain_filename(filename: str) -> dict:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return {"valid": False, "filename": filename}
    id_str, expression, gender, age = parts[0], parts[1], parts[2], parts[3]
    ethnicity = ETHNICITY_MAP.get(id_str[4], "Unknown") if len(id_str) >= 5 else "Unknown"
    return {
        "valid": True, "filename": filename, "id": id_str,
        "expression": expression, "gender": gender, "age": age,
        "ethnicity": ethnicity, "is_pain": expression == "Pain",
    }

# Filter images only (skip videos)
label_names = ds.features["label"].names
images_idx = label_names.index("Images") if "Images" in label_names else 0
ds_images = ds.filter(lambda x: x["label"] == images_idx)
print(f"After filtering 'Images': {len(ds_images)} entries")

# Extract images and metadata
all_images = []
all_metadata = []
all_labels = []

for i, example in enumerate(ds_images):
    img = example["image"]
    # Try to get filename
    img_path = getattr(img, "filename", None) or ""
    if img_path:
        meta = parse_synpain_filename(Path(img_path).name)
    else:
        meta = {"valid": False, "filename": f"unknown_{i}"}

    if meta.get("valid") and meta["expression"] in LABEL_MAP:
        all_images.append(img)
        all_metadata.append(meta)
        all_labels.append(LABEL_MAP[meta["expression"]])

print(f"\nValid images: {len(all_images)}")
counts = Counter(all_labels)
for lid, count in sorted(counts.items()):
    print(f"  {LABEL_NAMES[lid]}: {count}")

# %% [markdown]
# ## 5. Exploratory Data Analysis

# %%
# Class distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Label distribution
label_counts = Counter(all_labels)
axes[0].bar([LABEL_NAMES[k] for k in sorted(label_counts)],
            [label_counts[k] for k in sorted(label_counts)],
            color=["#2ecc71", "#e74c3c"])
axes[0].set_title("Expression Distribution", fontweight="bold")
axes[0].set_ylabel("Count")

# Gender distribution
genders = Counter(m["gender"] for m in all_metadata if m.get("valid"))
axes[1].bar(genders.keys(), genders.values(), color=["#3498db", "#e91e63"])
axes[1].set_title("Gender Distribution", fontweight="bold")

# Age distribution
ages = Counter(m["age"] for m in all_metadata if m.get("valid"))
axes[2].bar(ages.keys(), ages.values(), color=["#f39c12", "#9b59b6"])
axes[2].set_title("Age Distribution", fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_distributions.png", dpi=150)
plt.show()

# %%
# Show sample images
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Sample SynPAIN Images", fontsize=16, fontweight="bold")

pain_imgs = [(img, m) for img, m, l in zip(all_images, all_metadata, all_labels) if l == 1]
nopain_imgs = [(img, m) for img, m, l in zip(all_images, all_metadata, all_labels) if l == 0]

for i in range(5):
    if i < len(nopain_imgs):
        axes[0][i].imshow(nopain_imgs[i][0])
        m = nopain_imgs[i][1]
        axes[0][i].set_title(f"NoPain\n{m['gender']}, {m['age']}", fontsize=9)
    axes[0][i].axis("off")

    if i < len(pain_imgs):
        axes[1][i].imshow(pain_imgs[i][0])
        m = pain_imgs[i][1]
        axes[1][i].set_title(f"Pain\n{m['gender']}, {m['age']}", fontsize=9)
    axes[1][i].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_images.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Data Augmentation & Dataset

# %%
def get_train_transforms(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.3),
        A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class SynPainDataset(Dataset):
    def __init__(self, images, labels, metadata, transform=None):
        self.images = images
        self.labels = labels
        self.metadata = metadata
        self.transform = transform or get_val_transforms()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = np.array(self.images[idx].convert("RGB"))
        tensor = self.transform(image=img_np)["image"]
        return tensor, self.labels[idx]

    def get_class_weights(self):
        counts = Counter(self.labels)
        total = len(self.labels)
        return [total / (len(counts) * counts[i]) for i in range(len(counts))]

# %% [markdown]
# ## 7. Create DataLoaders

# %%
indices = list(range(len(all_images)))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, stratify=all_labels, random_state=42
)
temp_labels = [all_labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
)

def subset(idx_list):
    return ([all_images[i] for i in idx_list],
            [all_labels[i] for i in idx_list],
            [all_metadata[i] for i in idx_list])

train_imgs, train_labels, train_meta = subset(train_idx)
val_imgs, val_labels, val_meta = subset(val_idx)
test_imgs, test_labels, test_meta = subset(test_idx)

BS = CONFIG["batch_size"]
NW = CONFIG["num_workers"]

train_ds = SynPainDataset(train_imgs, train_labels, train_meta, get_train_transforms())
val_ds = SynPainDataset(val_imgs, val_labels, val_meta, get_val_transforms())
test_ds = SynPainDataset(test_imgs, test_labels, test_meta, get_val_transforms())

train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=NW, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=NW, pin_memory=True)

class_weights = train_ds.get_class_weights()
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Class weights: {class_weights}")

# %% [markdown]
# ## 8. Model Architecture

# %%
class HeartFaceClassifier(nn.Module):
    BACKBONES = {
        "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT, 1792),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    }

    def __init__(self, backbone="efficientnet_b4", num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        model_fn, weights, feat_dim = self.BACKBONES[backbone]
        base = model_fn(weights=weights if pretrained else None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def freeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = False
        print("🔒 Backbone frozen")

    def unfreeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = True
        print("🔓 Backbone unfrozen")

model = HeartFaceClassifier(CONFIG["backbone"], CONFIG["num_classes"], True, CONFIG["dropout"]).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {CONFIG['backbone']} | Params: {total_params:,} | Device: {DEVICE}")

# %% [markdown]
# ## 9. Training Loop

# %%
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
)
scaler = GradScaler(enabled=CONFIG["mixed_precision"])
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None


def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=CONFIG["mixed_precision"]):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return {"loss": total_loss / len(loader), "accuracy": 100.0 * correct / total}


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return {"loss": total_loss / len(loader), "accuracy": 100.0 * correct / total}


# === PHASE 1: Frozen backbone ===
print("=" * 60)
print(f"PHASE 1: Frozen backbone ({CONFIG['freeze_epochs']} epochs)")
print("=" * 60)

model.freeze_backbone()
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=CONFIG["phase1_lr"], weight_decay=CONFIG["weight_decay"])
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["freeze_epochs"])

for epoch in range(1, CONFIG["freeze_epochs"] + 1):
    start = time.time()
    t = train_epoch(model, train_loader, optimizer, epoch)
    v = validate(model, val_loader)
    scheduler.step()
    elapsed = time.time() - start

    history["train_loss"].append(t["loss"])
    history["train_acc"].append(t["accuracy"])
    history["val_loss"].append(v["loss"])
    history["val_acc"].append(v["accuracy"])

    if v["loss"] < best_val_loss:
        best_val_loss = v["loss"]
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"[P1] Epoch {epoch}/{CONFIG['freeze_epochs']} ({elapsed:.1f}s) | "
          f"Train Loss: {t['loss']:.4f} Acc: {t['accuracy']:.1f}% | "
          f"Val Loss: {v['loss']:.4f} Acc: {v['accuracy']:.1f}%")


# === PHASE 2: Full fine-tuning ===
remaining = CONFIG["total_epochs"] - CONFIG["freeze_epochs"]
print("\n" + "=" * 60)
print(f"PHASE 2: Full fine-tuning ({remaining} epochs)")
print("=" * 60)

model.unfreeze_backbone()
optimizer = AdamW(model.parameters(), lr=CONFIG["phase2_lr"], weight_decay=CONFIG["weight_decay"])
scheduler = CosineAnnealingLR(optimizer, T_max=remaining)
patience_counter = 0

for epoch in range(CONFIG["freeze_epochs"] + 1, CONFIG["total_epochs"] + 1):
    start = time.time()
    t = train_epoch(model, train_loader, optimizer, epoch)
    v = validate(model, val_loader)
    scheduler.step()
    elapsed = time.time() - start

    history["train_loss"].append(t["loss"])
    history["train_acc"].append(t["accuracy"])
    history["val_loss"].append(v["loss"])
    history["val_acc"].append(v["accuracy"])

    improved = ""
    if v["loss"] < best_val_loss:
        best_val_loss = v["loss"]
        patience_counter = 0
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        improved = " ✅ BEST"
    else:
        patience_counter += 1

    print(f"[P2] Epoch {epoch}/{CONFIG['total_epochs']} ({elapsed:.1f}s) | "
          f"Train Loss: {t['loss']:.4f} Acc: {t['accuracy']:.1f}% | "
          f"Val Loss: {v['loss']:.4f} Acc: {v['accuracy']:.1f}%{improved}")

    if patience_counter >= CONFIG["early_stopping_patience"]:
        print(f"⚠️ Early stopping at epoch {epoch}")
        break

print(f"\n🏆 Best val loss: {best_val_loss:.4f}")

# %% [markdown]
# ## 10. Evaluation on Test Set

# %%
# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)

model.eval()
all_preds, all_true, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_true.extend(labels.tolist())
        all_probs.append(probs)

all_probs = np.concatenate(all_probs, axis=0)

# Classification report
label_names_list = [LABEL_NAMES[i] for i in range(CONFIG["num_classes"])]
print("\n📊 Classification Report:")
print(classification_report(all_true, all_preds, target_names=label_names_list))

acc = accuracy_score(all_true, all_preds)
f1 = f1_score(all_true, all_preds, average="macro")
print(f"Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

# %% [markdown]
# ## 11. Visualizations

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training curves
axes[0][0].plot(history["train_loss"], label="Train", color="#3498db")
axes[0][0].plot(history["val_loss"], label="Val", color="#e74c3c")
axes[0][0].axvline(x=CONFIG["freeze_epochs"]-1, color="gray", linestyle="--", alpha=0.5, label="Unfreeze")
axes[0][0].set_title("Loss", fontweight="bold"); axes[0][0].legend(); axes[0][0].grid(alpha=0.3)

axes[0][1].plot(history["train_acc"], label="Train", color="#3498db")
axes[0][1].plot(history["val_acc"], label="Val", color="#e74c3c")
axes[0][1].axvline(x=CONFIG["freeze_epochs"]-1, color="gray", linestyle="--", alpha=0.5, label="Unfreeze")
axes[0][1].set_title("Accuracy (%)", fontweight="bold"); axes[0][1].legend(); axes[0][1].grid(alpha=0.3)

# Confusion matrix
cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names_list,
            yticklabels=label_names_list, ax=axes[1][0])
axes[1][0].set_title("Confusion Matrix", fontweight="bold")
axes[1][0].set_xlabel("Predicted"); axes[1][0].set_ylabel("Actual")

# ROC curves
y_bin = label_binarize(all_true, classes=[0, 1])
if y_bin.shape[1] == 1:
    y_bin = np.hstack([1 - y_bin, y_bin])
colors = ["#2ecc71", "#e74c3c"]
for i, (name, color) in enumerate(zip(label_names_list, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    axes[1][1].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
axes[1][1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1][1].set_title("ROC Curves", fontweight="bold")
axes[1][1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_results.png", dpi=150)
plt.show()

# %% [markdown]
# ## 12. Demographic Bias Analysis

# %%
def compute_demographic_metrics(y_true, y_pred, metadata):
    groups = {"gender": {}, "age": {}, "ethnicity": {}}
    for i, meta in enumerate(metadata):
        if not meta.get("valid"): continue
        for key in groups:
            val = meta.get(key, "Unknown")
            if val not in groups[key]:
                groups[key][val] = {"correct": 0, "total": 0}
            groups[key][val]["total"] += 1
            if y_true[i] == y_pred[i]:
                groups[key][val]["correct"] += 1

    results = {}
    for key, subs in groups.items():
        results[key] = {n: {"accuracy": round(c["correct"]/c["total"], 4) if c["total"] else 0,
                            "total": c["total"]} for n, c in subs.items()}
        accs = [v["accuracy"] for v in results[key].values()]
        results[f"{key}_gap"] = round(max(accs) - min(accs), 4) if accs else 0
    return results

demo_results = compute_demographic_metrics(all_true, all_preds, test_meta)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
palette = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

for ax, key in zip(axes, ["gender", "age", "ethnicity"]):
    if key not in demo_results: continue
    data = demo_results[key]
    names = list(data.keys())
    accs = [data[n]["accuracy"] * 100 for n in names]
    bars = ax.bar(names, accs, color=palette[:len(names)])
    ax.set_title(f"Accuracy by {key.title()}", fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    ax.axhline(y=np.mean(accs), color="gray", linestyle="--", alpha=0.5)
    for bar, n in zip(bars, names):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                f"n={data[n]['total']}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "demographic_bias.png", dpi=150)
plt.show()

print(f"\n📊 Demographic Gaps:")
for key in ["gender", "age", "ethnicity"]:
    print(f"  {key.title()}: {demo_results.get(f'{key}_gap', 0):.2%}")

# %% [markdown]
# ## 13. Save Model

# %%
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": CONFIG,
    "history": history,
    "test_accuracy": acc,
    "test_f1": f1,
    "demographic_bias": demo_results,
    "label_map": LABEL_MAP,
}

save_path = OUTPUT_DIR / "synpain_best_model.pth"
torch.save(checkpoint, save_path)
print(f"✅ Model saved: {save_path}")
print(f"   Test Accuracy: {acc:.4f}")
print(f"   Test F1: {f1:.4f}")
print(f"   File size: {save_path.stat().st_size / 1e6:.1f} MB")

# Save metrics JSON
metrics_json = {
    "accuracy": acc, "f1_macro": f1,
    "demographic_bias": demo_results,
    "config": CONFIG, "history": history,
}
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics_json, f, indent=2)

print(f"\n🎉 Training complete! Download from: {OUTPUT_DIR}")
