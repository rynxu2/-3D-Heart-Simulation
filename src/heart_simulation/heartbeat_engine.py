"""Heartbeat engine — converts BPM and pattern to animation parameters."""

import math
import time
import random
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger


@dataclass
class HeartbeatFrame:
    """Single frame of heartbeat animation."""
    timestamp: float       # seconds
    scale: float           # 0.85 - 1.0 (contraction to relaxation)
    phase: str             # "systole" or "diastole"
    color_intensity: float # 0.0 - 1.0 (for glow/pulse effect)
    beat_number: int


class HeartbeatEngine:
    """Generate heartbeat animation parameters from BPM and pattern.

    Cardiac cycle:
    - Systole (contraction): ~1/3 of cycle → scale 1.0 → 0.85
    - Diastole (relaxation): ~2/3 of cycle → scale 0.85 → 1.0
    """

    PATTERNS = {
        "regular": {
            "jitter": 0.0,
            "skip_chance": 0.0,
            "description": "Nhịp đều - tim bình thường",
        },
        "irregular": {
            "jitter": 0.15,       # ±15% timing variation
            "skip_chance": 0.10,  # 10% chance of skipped/extra beat
            "description": "Nhịp không đều - rối loạn nhịp tim",
        },
        "rapid_irregular": {
            "jitter": 0.20,       # ±20% timing variation
            "skip_chance": 0.05,
            "description": "Nhịp nhanh không đều - nhồi máu cơ tim",
        },
    }

    def __init__(
        self,
        bpm: int = 72,
        pattern: str = "regular",
        contraction_min: float = 0.85,
        contraction_max: float = 1.0,
        systole_ratio: float = 0.33,
    ):
        self.bpm = bpm
        self.pattern = pattern
        self.contraction_min = contraction_min
        self.contraction_max = contraction_max
        self.systole_ratio = systole_ratio
        self.diastole_ratio = 1.0 - systole_ratio

        self.cycle_duration = 60.0 / bpm  # seconds per beat
        self.pattern_config = self.PATTERNS.get(pattern, self.PATTERNS["regular"])

        logger.info(f"HeartbeatEngine: {bpm} BPM, pattern={pattern}, cycle={self.cycle_duration:.3f}s")

    def get_scale_at_time(self, t: float) -> float:
        """Calculate heart scale at time t (seconds)."""
        jitter = self.pattern_config["jitter"]
        effective_cycle = self.cycle_duration

        if jitter > 0:
            # Add timing jitter for irregular patterns
            random.seed(int(t * 10))  # Deterministic but varied
            effective_cycle *= (1.0 + random.uniform(-jitter, jitter))

        # Position within current cycle (0.0 to 1.0)
        cycle_pos = (t % effective_cycle) / effective_cycle

        if cycle_pos < self.systole_ratio:
            # Systole: contracting (1.0 → 0.85)
            progress = cycle_pos / self.systole_ratio
            # Smooth easing (ease-in-out)
            ease = 0.5 - 0.5 * math.cos(progress * math.pi)
            scale = self.contraction_max - ease * (self.contraction_max - self.contraction_min)
        else:
            # Diastole: relaxing (0.85 → 1.0)
            progress = (cycle_pos - self.systole_ratio) / self.diastole_ratio
            ease = 0.5 - 0.5 * math.cos(progress * math.pi)
            scale = self.contraction_min + ease * (self.contraction_max - self.contraction_min)

        return scale

    def get_color_intensity_at_time(self, t: float) -> float:
        """Color pulse intensity (1.0 at peak contraction, 0.0 at rest)."""
        scale = self.get_scale_at_time(t)
        # Normalize: 0 at max scale, 1 at min scale
        return 1.0 - (scale - self.contraction_min) / (self.contraction_max - self.contraction_min)

    def generate_frames(self, duration: float, fps: int = 60) -> List[HeartbeatFrame]:
        """Generate animation frames for a given duration."""
        frames = []
        total_frames = int(duration * fps)
        beat_count = 0
        last_phase = "diastole"

        for i in range(total_frames):
            t = i / fps
            scale = self.get_scale_at_time(t)
            intensity = self.get_color_intensity_at_time(t)

            cycle_pos = (t % self.cycle_duration) / self.cycle_duration
            phase = "systole" if cycle_pos < self.systole_ratio else "diastole"

            # Count beats
            if phase == "systole" and last_phase == "diastole":
                beat_count += 1
            last_phase = phase

            frames.append(HeartbeatFrame(
                timestamp=t,
                scale=scale,
                phase=phase,
                color_intensity=intensity,
                beat_number=beat_count,
            ))

        return frames

    def get_animation_params(self) -> dict:
        """Get parameters for Three.js animation (JSON-serializable)."""
        return {
            "bpm": self.bpm,
            "pattern": self.pattern,
            "description": self.pattern_config["description"],
            "cycleDuration": self.cycle_duration,
            "systoleRatio": self.systole_ratio,
            "contractionMin": self.contraction_min,
            "contractionMax": self.contraction_max,
            "jitter": self.pattern_config["jitter"],
            "skipChance": self.pattern_config["skip_chance"],
        }

    @classmethod
    def from_prediction(cls, prediction: dict) -> "HeartbeatEngine":
        """Create engine from predictor output."""
        bpm_info = prediction.get("bpm", {"min": 72, "max": 72, "pattern": "regular"})
        avg_bpm = (bpm_info["min"] + bpm_info["max"]) // 2
        return cls(bpm=avg_bpm, pattern=bpm_info["pattern"])

    @classmethod
    def from_label(cls, label: str) -> "HeartbeatEngine":
        """Create engine from a condition label (normal/abnormal/infarction)."""
        LABEL_PRESETS = {
            "normal": {"bpm": 72, "pattern": "regular"},
            "no_pain": {"bpm": 72, "pattern": "regular"},
            "abnormal": {"bpm": 90, "pattern": "irregular"},
            "infarction": {"bpm": 110, "pattern": "rapid_irregular"},
            "pain": {"bpm": 110, "pattern": "rapid_irregular"},
        }
        preset = LABEL_PRESETS.get(label, LABEL_PRESETS["normal"])
        return cls(bpm=preset["bpm"], pattern=preset["pattern"])
