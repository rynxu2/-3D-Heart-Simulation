"""Level B: Anatomical heart model with 4 chambers and valve animation."""

import json
from pathlib import Path
from typing import Optional
from loguru import logger

from src.heart_simulation.heartbeat_engine import HeartbeatEngine
from src.config import load_heart_config


class AnatomicalHeartModel:
    """Anatomical heart with 4 chambers, valves, and realistic animation.

    Uses a pre-made anatomical .glb model from Sketchfab (free license).
    Animation driven by HeartbeatEngine parameters.
    """

    HEART_PARTS = {
        "left_atrium": {"contraction_delay": 0.0, "scale_factor": 0.9},
        "right_atrium": {"contraction_delay": 0.02, "scale_factor": 0.9},
        "left_ventricle": {"contraction_delay": 0.12, "scale_factor": 0.8},
        "right_ventricle": {"contraction_delay": 0.14, "scale_factor": 0.85},
        "aorta": {"contraction_delay": 0.20, "scale_factor": 0.95},
        "pulmonary_artery": {"contraction_delay": 0.22, "scale_factor": 0.95},
    }

    VALVE_STATES = {
        "mitral": {"opens_during": "diastole", "chamber": "left_ventricle"},
        "tricuspid": {"opens_during": "diastole", "chamber": "right_ventricle"},
        "aortic": {"opens_during": "systole", "chamber": "left_ventricle"},
        "pulmonary": {"opens_during": "systole", "chamber": "right_ventricle"},
    }

    def __init__(self, model_path: Optional[str | Path] = None):
        self.config = load_heart_config()
        sim_config = self.config.get("simulation", {})
        self.colors = sim_config.get("colors", {
            "normal": "#e74c3c",
            "abnormal": "#f39c12",
            "infarction": "#8b0000",
            "damaged_zone": "#2c2c2c",
        })

        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = None

        logger.info("AnatomicalHeartModel initialized (Level B)")

    def get_animation_config(self, engine: HeartbeatEngine, condition: str = "normal") -> dict:
        """Generate full animation config for Three.js viewer.

        Args:
            engine: HeartbeatEngine with BPM and pattern
            condition: "normal", "abnormal", or "infarction"

        Returns:
            JSON-serializable config for heart_viewer.js
        """
        base_params = engine.get_animation_params()

        # Per-chamber animation with delays (realistic cardiac conduction)
        chambers = {}
        for part_name, part_config in self.HEART_PARTS.items():
            chambers[part_name] = {
                "contractionDelay": part_config["contraction_delay"],
                "scaleFactor": part_config["scale_factor"],
                "enabled": True,
            }

        # Valve animation
        valves = {}
        for valve_name, valve_config in self.VALVE_STATES.items():
            valves[valve_name] = {
                "opensDuring": valve_config["opens_during"],
                "linkedChamber": valve_config["chamber"],
            }

        # Condition-specific effects
        effects = self._get_condition_effects(condition)

        return {
            "level": "anatomy",
            "modelPath": str(self.model_path) if self.model_path else None,
            "heartbeat": base_params,
            "chambers": chambers,
            "valves": valves,
            "color": self.colors.get(condition, self.colors["normal"]),
            "effects": effects,
            "viewer": self.config.get("viewer", {}),
        }

    def _get_condition_effects(self, condition: str) -> dict:
        """Visual effects based on heart condition."""
        if condition == "normal":
            return {
                "glow": True,
                "glowColor": self.colors["normal"],
                "glowIntensity": 0.3,
                "damageZones": [],
                "bloodFlow": "normal",
            }
        elif condition == "abnormal":
            return {
                "glow": True,
                "glowColor": self.colors["abnormal"],
                "glowIntensity": 0.5,
                "damageZones": [],
                "bloodFlow": "irregular",
                "flickerEffect": True,  # Pulsing glow for arrhythmia
            }
        else:  # infarction
            return {
                "glow": True,
                "glowColor": self.colors["infarction"],
                "glowIntensity": 0.8,
                "damageZones": [
                    {
                        "position": [0.3, -0.2, 0.1],  # Left ventricle anterior wall
                        "radius": 0.15,
                        "color": self.colors["damaged_zone"],
                        "label": "Vùng nhồi máu",
                    }
                ],
                "bloodFlow": "restricted",
                "warningPulse": True,
            }

    def export_config_json(self, engine: HeartbeatEngine, condition: str, output_path: str | Path):
        """Export animation config to JSON file."""
        config = self.get_animation_config(engine, condition)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Animation config exported: {output_path}")
