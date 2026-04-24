"""Unit tests for SynPAIN data loading and training pipeline."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.synpain_loader import (
    parse_synpain_filename,
    create_labels_from_metadata,
    SynPainDataset,
    SYNPAIN_BINARY_MAP,
    SYNPAIN_3CLASS_MAP,
)
from src.models.classifier_2d import HeartFaceClassifier2D, create_classifier
from src.data.augmentation import get_val_transforms


class TestFilenameParser:
    """Test SynPAIN filename parsing logic."""

    def test_parse_pain_man_young(self):
        result = parse_synpain_filename("1100300001_Pain_Man_Young.jpg")
        assert result["valid"] is True
        assert result["expression"] == "Pain"
        assert result["gender"] == "Man"
        assert result["age"] == "Young"
        assert result["is_pain"] is True
        assert result["ethnicity"] == "Caucasian"

    def test_parse_nopain_woman_old(self):
        result = parse_synpain_filename("1010100005_NoPain_Woman_Old.jpg")
        assert result["valid"] is True
        assert result["expression"] == "NoPain"
        assert result["gender"] == "Woman"
        assert result["age"] == "Old"
        assert result["is_pain"] is False

    def test_parse_ethnicity_black(self):
        result = parse_synpain_filename("1100000001_Pain_Man_Young.jpg")
        assert result["ethnicity"] == "Black"

    def test_parse_ethnicity_south_asian(self):
        result = parse_synpain_filename("1100100001_Pain_Man_Young.jpg")
        assert result["ethnicity"] == "South Asian"

    def test_parse_ethnicity_middle_eastern(self):
        result = parse_synpain_filename("1100200001_Pain_Man_Young.jpg")
        assert result["ethnicity"] == "Middle Eastern"

    def test_parse_ethnicity_east_asian(self):
        result = parse_synpain_filename("1100400001_Pain_Man_Young.jpg")
        assert result["ethnicity"] == "East Asian"

    def test_parse_invalid_filename(self):
        result = parse_synpain_filename("invalid.jpg")
        assert result["valid"] is False

    def test_parse_short_filename(self):
        result = parse_synpain_filename("ab.jpg")
        assert result["valid"] is False


class TestLabelMapping:
    """Test label creation from parsed metadata."""

    @pytest.fixture
    def sample_metadata(self):
        return [
            {"valid": True, "expression": "Pain", "gender": "Man", "age": "Young", "ethnicity": "Caucasian"},
            {"valid": True, "expression": "NoPain", "gender": "Woman", "age": "Old", "ethnicity": "Black"},
            {"valid": True, "expression": "Pain", "gender": "Woman", "age": "Young", "ethnicity": "East Asian"},
            {"valid": False, "filename": "invalid"},
            {"valid": True, "expression": "NoPain", "gender": "Man", "age": "Old", "ethnicity": "South Asian"},
        ]

    def test_binary_mapping(self, sample_metadata):
        labels, valid_indices = create_labels_from_metadata(sample_metadata, "binary")
        assert len(labels) == 4  # 4 valid entries
        assert labels[0] == 1   # Pain → 1
        assert labels[1] == 0   # NoPain → 0
        assert labels[2] == 1   # Pain → 1
        assert labels[3] == 0   # NoPain → 0

    def test_3class_mapping(self, sample_metadata):
        labels, valid_indices = create_labels_from_metadata(sample_metadata, "3class")
        assert labels[0] == 2   # Pain → infarction (2)
        assert labels[1] == 0   # NoPain → normal (0)

    def test_invalid_entries_filtered(self, sample_metadata):
        labels, valid_indices = create_labels_from_metadata(sample_metadata, "binary")
        assert 3 not in valid_indices  # Index 3 (invalid) should be excluded


class TestSynPainDataset:
    """Test PyTorch Dataset wrapper."""

    @pytest.fixture
    def mock_dataset(self):
        from PIL import Image
        images = [Image.new("RGB", (64, 64), color=(i * 50, 100, 150)) for i in range(5)]
        labels = [0, 1, 0, 1, 0]
        metadata = [
            {"valid": True, "gender": "Man", "age": "Young", "ethnicity": "Caucasian"},
            {"valid": True, "gender": "Woman", "age": "Old", "ethnicity": "Black"},
            {"valid": True, "gender": "Man", "age": "Young", "ethnicity": "East Asian"},
            {"valid": True, "gender": "Woman", "age": "Old", "ethnicity": "South Asian"},
            {"valid": True, "gender": "Man", "age": "Young", "ethnicity": "Middle Eastern"},
        ]
        return SynPainDataset(images, labels, metadata, get_val_transforms(224), 224)

    def test_dataset_length(self, mock_dataset):
        assert len(mock_dataset) == 5

    def test_getitem_returns_tensor_and_label(self, mock_dataset):
        tensor, label = mock_dataset[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert label in {0, 1}

    def test_class_weights(self, mock_dataset):
        weights = mock_dataset.get_class_weights()
        assert len(weights) == 2
        assert all(w > 0 for w in weights)

    def test_demographics(self, mock_dataset):
        demos = mock_dataset.get_demographics()
        assert len(demos) == 5
        assert demos[0]["gender"] == "Man"


class TestModelBinary:
    """Test model with binary (2-class) classification."""

    def test_forward_pass_2class(self):
        model = HeartFaceClassifier2D(
            backbone="resnet34",
            num_classes=2,
            pretrained=False,
            dropout=0.1,
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 2)

    def test_freeze_unfreeze(self):
        model = HeartFaceClassifier2D(
            backbone="resnet34",
            num_classes=2,
            pretrained=False,
        )
        model.freeze_backbone()
        for param in model.features.parameters():
            assert param.requires_grad is False

        model.unfreeze_backbone()
        for param in model.features.parameters():
            assert param.requires_grad is True

    def test_predict_binary(self):
        model = HeartFaceClassifier2D(
            backbone="resnet34",
            num_classes=2,
            pretrained=False,
        )
        x = torch.randn(1, 3, 224, 224)
        pred_class, probs = model.predict(x)
        assert pred_class.shape == (1,)
        assert probs.shape == (1, 2)
        assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
