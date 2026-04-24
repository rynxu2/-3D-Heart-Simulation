"""Tests for 3D CNN pipeline — video dataset, classifier, and predictor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np


class TestVideoDataset:
    """Test VideoClipDataset class."""

    def test_pseudo_frames_from_single_image(self, tmp_path):
        """Single image should generate num_frames pseudo-frames."""
        import cv2
        from src.data.video_dataset import VideoClipDataset

        # Create a dummy image
        img_path = tmp_path / "face.jpg"
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        dataset = VideoClipDataset(
            clip_paths=[str(img_path)],
            labels=[0],
            num_frames=16,
            image_size=224,
        )

        tensor, label = dataset[0]
        assert tensor.shape == (3, 16, 224, 224), f"Expected (3,16,224,224), got {tensor.shape}"
        assert label == 0

    def test_frame_folder_loading(self, tmp_path):
        """Frame folder should load correctly."""
        import cv2
        from src.data.video_dataset import VideoClipDataset

        clip_dir = tmp_path / "clip_00001"
        clip_dir.mkdir()
        for i in range(20):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(clip_dir / f"frame_{i:02d}.jpg"), img)

        dataset = VideoClipDataset(
            clip_paths=[str(clip_dir)],
            labels=[1],
            num_frames=16,
            image_size=224,
        )

        tensor, label = dataset[0]
        assert tensor.shape == (3, 16, 224, 224)
        assert label == 1

    def test_dataset_length(self, tmp_path):
        """Dataset length should match number of clips."""
        import cv2
        from src.data.video_dataset import VideoClipDataset

        paths = []
        for i in range(5):
            img_path = tmp_path / f"face_{i}.jpg"
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            paths.append(str(img_path))

        dataset = VideoClipDataset(paths, [0, 1, 2, 0, 1], num_frames=8, image_size=112)
        assert len(dataset) == 5


class TestClassifier3D:
    """Test HeartFaceClassifier3D."""

    def test_forward_pass_shape(self):
        """Model output should be (B, num_classes)."""
        from src.models.classifier_3d import HeartFaceClassifier3D

        model = HeartFaceClassifier3D(num_classes=3, pretrained=False)
        x = torch.randn(2, 3, 16, 224, 224)
        out = model(x)
        assert out.shape == (2, 3), f"Expected (2,3), got {out.shape}"

    def test_freeze_unfreeze(self):
        """Freeze/unfreeze should toggle requires_grad."""
        from src.models.classifier_3d import HeartFaceClassifier3D

        model = HeartFaceClassifier3D(num_classes=3, pretrained=False)

        model.freeze_backbone()
        for p in model.features.parameters():
            assert not p.requires_grad

        model.unfreeze_backbone()
        for p in model.features.parameters():
            assert p.requires_grad

    def test_predict_method(self):
        """predict() should return class indices and probabilities."""
        from src.models.classifier_3d import HeartFaceClassifier3D

        model = HeartFaceClassifier3D(num_classes=3, pretrained=False)
        x = torch.randn(1, 3, 16, 224, 224)
        pred_class, probs = model.predict(x)

        assert pred_class.shape == (1,)
        assert probs.shape == (1, 3)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_num_classes_2(self):
        """Binary classifier should output (B, 2)."""
        from src.models.classifier_3d import HeartFaceClassifier3D

        model = HeartFaceClassifier3D(num_classes=2, pretrained=False)
        x = torch.randn(1, 3, 16, 224, 224)
        out = model(x)
        assert out.shape == (1, 2)


class TestHeartbeatEngine:
    """Test HeartbeatEngine label-based creation."""

    def test_from_label_normal(self):
        from src.heart_simulation.heartbeat_engine import HeartbeatEngine

        engine = HeartbeatEngine.from_label("normal")
        assert engine.bpm == 72
        assert engine.pattern == "regular"

    def test_from_label_abnormal(self):
        from src.heart_simulation.heartbeat_engine import HeartbeatEngine

        engine = HeartbeatEngine.from_label("abnormal")
        assert engine.bpm == 90
        assert engine.pattern == "irregular"

    def test_from_label_infarction(self):
        from src.heart_simulation.heartbeat_engine import HeartbeatEngine

        engine = HeartbeatEngine.from_label("infarction")
        assert engine.bpm == 110
        assert engine.pattern == "rapid_irregular"

    def test_from_label_unknown_defaults_to_normal(self):
        from src.heart_simulation.heartbeat_engine import HeartbeatEngine

        engine = HeartbeatEngine.from_label("unknown_label")
        assert engine.bpm == 72
        assert engine.pattern == "regular"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
