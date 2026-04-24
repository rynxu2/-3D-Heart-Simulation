"""PyTorch Dataset for video-based heart condition classification (3D CNN).

Loads 16-frame video clips from organized directory structure:
    data_dir/
        normal/      ← video clips (.mp4/.avi) or frame folders
        abnormal/
        infarction/
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from loguru import logger

from src.data.augmentation import get_train_transforms, get_val_transforms

LABEL_MAP = {"normal": 0, "abnormal": 1, "infarction": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class VideoClipDataset(Dataset):
    """Dataset for 16-frame video clips → 3D CNN input.

    Input format: (B, 3, T, H, W) where T=num_frames.
    Supports two modes:
    - Video files: extract equally-spaced frames from .mp4/.avi
    - Frame folders: load sequential images from a subdirectory
    """

    def __init__(
        self,
        clip_paths: List[str],
        labels: List[int],
        num_frames: int = 16,
        image_size: int = 224,
        transform=None,
        is_train: bool = False,
    ):
        self.clip_paths = clip_paths
        self.labels = labels
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.clip_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        clip_path = Path(self.clip_paths[idx])
        label = self.labels[idx]

        if clip_path.is_dir():
            frames = self._load_frame_folder(clip_path)
        elif clip_path.suffix.lower() in VIDEO_EXTS:
            frames = self._load_video_file(clip_path)
        else:
            frames = self._generate_pseudo_frames(clip_path)

        # Apply spatial transforms to each frame
        processed = []
        for frame in frames:
            if self.transform:
                augmented = self.transform(image=frame)
                processed.append(augmented["image"])
            else:
                resized = cv2.resize(frame, (self.image_size, self.image_size))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                processed.append(tensor)

        # Stack: (T, C, H, W) → (C, T, H, W) for 3D CNN
        video_tensor = torch.stack(processed, dim=1)  # (C, T, H, W)
        return video_tensor, label

    def _load_video_file(self, path: Path) -> List[np.ndarray]:
        """Extract equally-spaced frames from video file."""
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * self.num_frames

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)
            else:
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        cap.release()

        # Pad if not enough frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        return frames[:self.num_frames]

    def _load_frame_folder(self, folder: Path) -> List[np.ndarray]:
        """Load sequential frames from a directory of images."""
        files = sorted([f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS])

        if not files:
            return [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * self.num_frames

        # Sample equally spaced
        indices = np.linspace(0, len(files) - 1, self.num_frames, dtype=int)
        frames = []

        for i in indices:
            img = cv2.imread(str(files[i]))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.image_size, self.image_size))
                frames.append(img)
            else:
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        return frames[:self.num_frames]

    def _generate_pseudo_frames(self, image_path: Path) -> List[np.ndarray]:
        """Generate pseudo video from single image with small augmentations.

        Useful when only still images are available — simulates
        micro-movements by applying slight affine transforms per frame.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * self.num_frames

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        frames = []
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        for i in range(self.num_frames):
            # Small random affine: ±2° rotation, ±2px translation
            angle = np.random.uniform(-2, 2) if self.is_train else (i - self.num_frames // 2) * 0.3
            tx = np.random.uniform(-2, 2) if self.is_train else 0
            ty = np.random.uniform(-2, 2) if self.is_train else 0

            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty

            warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            frames.append(warped)

        return frames

    def get_class_weights(self) -> List[float]:
        """Compute inverse-frequency class weights."""
        counts = Counter(self.labels)
        total = len(self.labels)
        num_classes = max(counts.keys()) + 1
        return [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]


def build_video_dataset_from_directory(
    data_dir: str | Path,
) -> Tuple[List[str], List[int]]:
    """Scan directory for video clips or frame folders.

    Supports:
    - Video files directly in label dirs: data_dir/normal/clip001.mp4
    - Frame folders: data_dir/normal/clip001/frame_00.jpg
    - Single images (converted to pseudo-video): data_dir/normal/face_001.jpg
    """
    data_dir = Path(data_dir)
    clip_paths = []
    labels = []

    for label_name, label_id in LABEL_MAP.items():
        label_dir = data_dir / label_name
        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue

        for item in sorted(label_dir.iterdir()):
            if item.is_dir():
                # Frame folder
                has_images = any(f.suffix.lower() in IMAGE_EXTS for f in item.iterdir())
                if has_images:
                    clip_paths.append(str(item))
                    labels.append(label_id)
            elif item.suffix.lower() in VIDEO_EXTS:
                clip_paths.append(str(item))
                labels.append(label_id)
            elif item.suffix.lower() in IMAGE_EXTS:
                # Single image → pseudo-video
                clip_paths.append(str(item))
                labels.append(label_id)

    logger.info(f"Found {len(clip_paths)} clips across {len(LABEL_MAP)} classes")
    for name, lid in LABEL_MAP.items():
        count = labels.count(lid)
        logger.info(f"  {name}: {count} clips")

    return clip_paths, labels


def create_video_dataloaders(
    data_dir: str | Path,
    num_frames: int = 16,
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 2,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Create train/val/test DataLoaders for video clips."""
    clip_paths, labels = build_video_dataset_from_directory(data_dir)

    if not clip_paths:
        raise ValueError(f"No video clips found in {data_dir}")

    # Stratified split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        clip_paths, labels,
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=42,
    )
    val_ratio = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=42,
    )

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_ds = VideoClipDataset(train_paths, train_labels, num_frames, image_size, train_transform, is_train=True)
    val_ds = VideoClipDataset(val_paths, val_labels, num_frames, image_size, val_transform)
    test_ds = VideoClipDataset(test_paths, test_labels, num_frames, image_size, val_transform)

    logger.info(f"Video split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    info = {
        "total_clips": len(clip_paths),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "class_weights": train_ds.get_class_weights(),
        "num_frames": num_frames,
    }

    return train_loader, val_loader, test_loader, info
