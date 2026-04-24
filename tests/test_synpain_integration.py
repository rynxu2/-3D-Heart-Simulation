"""Integration tests for SynPAIN pipeline (requires dataset download)."""

import pytest
import torch

from src.config import SynPainConfig


@pytest.mark.slow
class TestSynPainE2E:
    """End-to-end tests — require SynPAIN download (~5GB)."""

    @pytest.fixture(scope="class")
    def loaders(self):
        from src.data.synpain_loader import create_synpain_loaders
        train_loader, val_loader, test_loader, info = create_synpain_loaders(
            strategy="binary",
            batch_size=4,
            num_workers=0,
        )
        return train_loader, val_loader, test_loader, info

    def test_dataset_loads(self, loaders):
        _, _, _, info = loaders
        assert info["total_images"] > 0
        assert info["train_size"] > 0

    def test_batch_shape(self, loaders):
        train_loader = loaders[0]
        images, labels = next(iter(train_loader))
        assert images.shape[1:] == (3, 224, 224)
        assert labels.shape[0] == images.shape[0]
        assert all(l in {0, 1} for l in labels.tolist())

    def test_train_one_epoch(self, loaders):
        from src.models.classifier_2d import create_classifier
        from src.models.trainer import Trainer

        train_loader, val_loader = loaders[0], loaders[1]
        config = SynPainConfig(epochs=1, num_classes=2, backbone="resnet34")

        model = create_classifier(backbone="resnet34", num_classes=2, pretrained=False)
        trainer = Trainer(model=model, config=config)
        history = trainer.train(train_loader, val_loader)

        assert len(history["train_loss"]) == 1
        assert history["train_loss"][0] > 0

    def test_demographic_balance(self, loaders):
        test_loader = loaders[2]
        test_ds = test_loader.dataset
        demographics = test_ds.get_demographics()

        genders = {m.get("gender") for m in demographics if m.get("valid")}
        ages = {m.get("age") for m in demographics if m.get("valid")}

        assert "Man" in genders or "Woman" in genders
        assert "Young" in ages or "Old" in ages
