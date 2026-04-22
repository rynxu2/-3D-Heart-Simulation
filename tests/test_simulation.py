import pytest
from src.heart_simulation.heartbeat_engine import HeartbeatEngine

class TestHeartbeatEngine:
    def test_regular_pattern_timing(self):
        # Arrange
        engine = HeartbeatEngine(bpm=60, pattern="regular", contraction_min=0.85, contraction_max=1.0)
        
        # Act & Assert
        assert engine.cycle_duration == 1.0  # 60 BPM = 1 beat per second
        assert engine.pattern_config["jitter"] == 0.0
        
    def test_scale_bounds(self):
        # Arrange
        engine = HeartbeatEngine(bpm=80, pattern="regular", contraction_min=0.85, contraction_max=1.0)
        
        # Act & Assert
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5]:
            scale = engine.get_scale_at_time(t)
            assert 0.85 <= scale <= 1.0, f"Scale {scale} is out of bounds at time {t}"

    def test_irregular_pattern_config(self):
        # Arrange & Act
        engine = HeartbeatEngine(bpm=120, pattern="irregular")
        
        # Assert
        assert engine.pattern_config["jitter"] > 0
        assert engine.cycle_duration == 0.5  # 120 BPM = 0.5s per beat
