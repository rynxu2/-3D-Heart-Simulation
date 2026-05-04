"""Heart renderer — export 3D heart animations to video/GIF/frames."""

import math
import json
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from loguru import logger

from src.heart_simulation.heartbeat_engine import HeartbeatEngine, HeartbeatFrame


class HeartRenderer:
    """Render heart animation to video, GIF, or frame sequence.

    Modes:
    - JSON export: animation config for Three.js (web)
    - Frame sequence: PNG frames for offline rendering
    - Video: MP4 via OpenCV
    - ECG waveform: simulated ECG trace image
    """

    def __init__(self, output_dir: Union[str, Path] = "data/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_animation_json(
        self,
        engine: HeartbeatEngine,
        condition: str,
        duration: float = 5.0,
        fps: int = 60,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Export animation keyframes as JSON for Three.js consumption."""
        frames = engine.generate_frames(duration, fps)

        keyframes = []
        for frame in frames:
            keyframes.append({
                "t": round(frame.timestamp, 4),
                "scale": round(frame.scale, 4),
                "phase": frame.phase,
                "intensity": round(frame.color_intensity, 4),
                "beat": frame.beat_number,
            })

        data = {
            "condition": condition,
            "bpm": engine.bpm,
            "pattern": engine.pattern,
            "duration": duration,
            "fps": fps,
            "totalFrames": len(keyframes),
            "keyframes": keyframes,
        }

        path = Path(output_path) if output_path else self.output_dir / f"heart_anim_{condition}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Animation JSON exported: {path} ({len(keyframes)} frames)")
        return path

    def render_ecg_waveform(
        self,
        engine: HeartbeatEngine,
        condition: str,
        duration: float = 5.0,
        width: int = 800,
        height: int = 200,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Render a simulated ECG waveform image."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib required: pip install matplotlib")
            raise

        fps = 200  # High res for smooth ECG
        frames = engine.generate_frames(duration, fps)

        times = [f.timestamp for f in frames]
        ecg_signal = self._generate_ecg_signal(frames, engine)

        # Color based on condition
        colors = {"normal": "#2ecc71", "abnormal": "#f39c12", "infarction": "#e74c3c"}
        color = colors.get(condition, "#2ecc71")

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.plot(times, ecg_signal, color=color, linewidth=1.5)
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xlabel("Time (s)", color="white", fontsize=9)
        ax.set_ylabel("mV", color="white", fontsize=9)

        label_vi = {"normal": "Bình thường", "abnormal": "Bất thường", "infarction": "Nhồi máu cơ tim"}
        ax.set_title(
            f"ECG - {label_vi.get(condition, condition)} ({engine.bpm} BPM)",
            color="white", fontsize=11, fontweight="bold",
        )

        # Grid
        ax.grid(True, alpha=0.2, color="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        path = Path(output_path) if output_path else self.output_dir / f"ecg_{condition}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(path), dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info(f"ECG waveform saved: {path}")
        return path

    def _generate_ecg_signal(self, frames: List[HeartbeatFrame], engine: HeartbeatEngine) -> List[float]:
        """Generate synthetic ECG-like signal from heartbeat frames."""
        signal = []
        cycle = engine.cycle_duration

        for frame in frames:
            t = frame.timestamp
            pos = (t % cycle) / cycle

            # PQRST complex simulation
            val = 0.0

            # P wave (atrial depolarization)
            if 0.0 < pos < 0.1:
                val = 0.15 * math.sin(pos / 0.1 * math.pi)

            # QRS complex (ventricular depolarization)
            elif 0.12 < pos < 0.2:
                qrs_pos = (pos - 0.12) / 0.08
                if qrs_pos < 0.2:
                    val = -0.1 * math.sin(qrs_pos / 0.2 * math.pi)  # Q
                elif qrs_pos < 0.5:
                    val = 1.2 * math.sin((qrs_pos - 0.2) / 0.3 * math.pi)  # R
                else:
                    val = -0.3 * math.sin((qrs_pos - 0.5) / 0.5 * math.pi)  # S

            # T wave (ventricular repolarization)
            elif 0.25 < pos < 0.45:
                val = 0.3 * math.sin((pos - 0.25) / 0.2 * math.pi)

            # Add jitter for irregular patterns
            if engine.pattern != "regular":
                jitter = engine.pattern_config["jitter"]
                noise = np.random.normal(0, jitter * 0.1)
                val += noise

            # ST elevation for infarction
            if engine.pattern == "rapid_irregular" and 0.2 < pos < 0.3:
                val += 0.25  # ST segment elevation

            signal.append(val)

        return signal

    def render_comparison(
        self,
        duration: float = 5.0,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Render side-by-side ECG comparison of all 3 conditions."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise

        conditions = [
            ("normal", 72, "regular", "Bình thường", "#2ecc71"),
            ("abnormal", 110, "irregular", "Bất thường", "#f39c12"),
            ("infarction", 120, "rapid_irregular", "Nhồi máu cơ tim", "#e74c3c"),
        ]

        fig, axes = plt.subplots(3, 1, figsize=(10, 6), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")

        for ax, (cond, bpm, pattern, label, color) in zip(axes, conditions):
            engine = HeartbeatEngine(bpm=bpm, pattern=pattern)
            frames = engine.generate_frames(duration, fps=200)
            times = [f.timestamp for f in frames]
            ecg = self._generate_ecg_signal(frames, engine)

            ax.plot(times, ecg, color=color, linewidth=1.2)
            ax.set_facecolor("#1a1a2e")
            ax.set_ylabel("mV", color="white", fontsize=8)
            ax.set_title(f"{label} ({bpm} BPM)", color=color, fontsize=10, fontweight="bold")
            ax.tick_params(colors="white", labelsize=7)
            ax.grid(True, alpha=0.15, color="white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")

        axes[-1].set_xlabel("Time (s)", color="white", fontsize=9)
        plt.tight_layout()

        path = Path(output_path) if output_path else self.output_dir / "ecg_comparison.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(path), dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info(f"ECG comparison saved: {path}")
        return path
