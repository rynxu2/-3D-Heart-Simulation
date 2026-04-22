"""Synthetic face image generator using Stable Diffusion or simple augmentation."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from loguru import logger


# Prompt templates for each heart condition
CONDITION_PROMPTS = {
    "normal": [
        "portrait photo of a healthy person, relaxed expression, calm face, natural skin color, studio lighting",
        "face photo of a person at rest, peaceful expression, normal complexion, front view",
        "headshot of a healthy adult, neutral expression, clear skin, natural lighting",
    ],
    "abnormal": [
        "portrait of a person experiencing chest discomfort, slight grimace, mild sweating, worried expression",
        "face photo of person with heart palpitations, uneasy expression, slightly pale, concerned look",
        "headshot of person feeling dizzy, uncomfortable expression, light perspiration, front view",
    ],
    "infarction": [
        "portrait of person in severe chest pain, distressed expression, pale face, heavy sweating",
        "face photo of person having heart attack, agonized expression, ashen skin, clenched jaw",
        "headshot of person in acute cardiac distress, pain expression, grey pallor, sweating profusely",
    ],
}


class SyntheticGenerator:
    """Generate synthetic face images for training data augmentation.

    Methods:
    1. Stable Diffusion (high quality, requires GPU + diffusers)
    2. OpenCV augmentation (fast, no extra deps, lower diversity)
    """

    def __init__(self, output_dir: str | Path, target_size: int = 224):
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self._sd_pipeline = None

        for label in ["normal", "abnormal", "infarction"]:
            (self.output_dir / label).mkdir(parents=True, exist_ok=True)

    def generate_augmented(
        self,
        source_dir: str | Path,
        label: str,
        num_per_image: int = 5,
    ) -> int:
        """Generate synthetic images via heavy augmentation of existing images.

        Applies combinations of: color shift, blur, noise, brightness,
        perspective transform, elastic deformation.
        """
        source_dir = Path(source_dir)
        save_dir = self.output_dir / label
        exts = {".jpg", ".jpeg", ".png"}
        generated = 0
        existing = len(list(save_dir.glob("*.jpg")))

        source_files = [f for f in source_dir.iterdir() if f.suffix.lower() in exts]
        if not source_files:
            logger.warning(f"No source images in {source_dir}")
            return 0

        for img_file in source_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            image = cv2.resize(image, (self.target_size, self.target_size))

            for i in range(num_per_image):
                augmented = self._apply_random_augmentation(image, label)
                filename = f"syn_{label}_{existing + generated:05d}.jpg"
                cv2.imwrite(str(save_dir / filename), augmented)
                generated += 1

        logger.info(f"Generated {generated} augmented images for '{label}'")
        return generated

    def _apply_random_augmentation(self, image: np.ndarray, condition: str) -> np.ndarray:
        """Apply condition-specific random augmentations."""
        result = image.copy()
        rng = np.random.default_rng()

        # Random horizontal flip
        if rng.random() > 0.5:
            result = cv2.flip(result, 1)

        # Random rotation (-15 to 15 degrees)
        angle = rng.uniform(-15, 15)
        h, w = result.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Condition-specific color modifications
        if condition == "abnormal":
            # Slight pallor: reduce red, increase blue slightly
            result = self._adjust_pallor(result, strength=rng.uniform(0.05, 0.15))
            # Add slight sweat shine
            if rng.random() > 0.5:
                result = self._add_shine(result, intensity=rng.uniform(0.1, 0.3))

        elif condition == "infarction":
            # Strong pallor / grey tone
            result = self._adjust_pallor(result, strength=rng.uniform(0.15, 0.35))
            # Heavy sweating effect
            result = self._add_shine(result, intensity=rng.uniform(0.2, 0.5))
            # Reduce saturation (ashen look)
            result = self._reduce_saturation(result, factor=rng.uniform(0.5, 0.8))

        # Random brightness/contrast
        alpha = rng.uniform(0.8, 1.2)  # contrast
        beta = rng.uniform(-20, 20)     # brightness
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

        # Random gaussian noise
        if rng.random() > 0.5:
            noise = rng.normal(0, rng.uniform(5, 15), result.shape).astype(np.int16)
            result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return result

    def _adjust_pallor(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Make face paler — reduce warm tones."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= (1 - strength)  # Reduce saturation
        hsv[:, :, 2] *= (1 + strength * 0.3)  # Slightly increase brightness
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _add_shine(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add specular highlights to simulate sweating."""
        h, w = image.shape[:2]
        rng = np.random.default_rng()

        for _ in range(int(intensity * 20)):
            cx = rng.integers(w // 4, 3 * w // 4)
            cy = rng.integers(h // 4, 3 * h // 4)
            radius = rng.integers(3, 10)
            brightness = rng.integers(180, 255)
            overlay = image.copy()
            cv2.circle(overlay, (cx, cy), radius, (brightness, brightness, brightness), -1)
            image = cv2.addWeighted(image, 0.85, overlay, 0.15, 0)

        return image

    def _reduce_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Reduce color saturation for ashen appearance."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def generate_with_diffusion(
        self,
        label: str,
        num_images: int = 50,
        guidance_scale: float = 7.5,
    ) -> int:
        """Generate images using Stable Diffusion (requires diffusers)."""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
        except ImportError:
            logger.error("Install diffusers: pip install diffusers transformers accelerate")
            return 0

        if self._sd_pipeline is None:
            logger.info("Loading Stable Diffusion pipeline...")
            self._sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
            ).to("cuda" if torch.cuda.is_available() else "cpu")

        save_dir = self.output_dir / label
        existing = len(list(save_dir.glob("*.jpg")))
        prompts = CONDITION_PROMPTS.get(label, CONDITION_PROMPTS["normal"])
        generated = 0

        for i in range(num_images):
            prompt = prompts[i % len(prompts)]
            image = self._sd_pipeline(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=30,
                height=512,
                width=512,
            ).images[0]

            # Resize and save
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            image_np = cv2.resize(image_np, (self.target_size, self.target_size))

            filename = f"sd_{label}_{existing + generated:05d}.jpg"
            cv2.imwrite(str(save_dir / filename), image_np)
            generated += 1

            if (i + 1) % 10 == 0:
                logger.info(f"Generated {generated}/{num_images} for '{label}'")

        logger.info(f"Generated {generated} SD images for '{label}'")
        return generated
