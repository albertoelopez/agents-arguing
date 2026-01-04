import numpy as np
import pytest

from src.utils.audio import normalize_audio


class TestNormalizeAudio:
    def test_normalizes_loud_audio(self):
        loud_audio = np.ones(1000, dtype=np.float32) * 0.9
        normalized = normalize_audio(loud_audio, target_db=-20.0)

        assert np.max(np.abs(normalized)) < 0.9

    def test_normalizes_quiet_audio(self):
        quiet_audio = np.ones(1000, dtype=np.float32) * 0.01
        normalized = normalize_audio(quiet_audio, target_db=-20.0)

        assert np.max(np.abs(normalized)) > 0.01

    def test_clips_to_valid_range(self):
        audio = np.ones(1000, dtype=np.float32) * 0.001
        normalized = normalize_audio(audio, target_db=0.0)

        assert np.all(normalized >= -1.0)
        assert np.all(normalized <= 1.0)

    def test_handles_silent_audio(self):
        silent = np.zeros(1000, dtype=np.float32)
        normalized = normalize_audio(silent)

        assert np.allclose(normalized, 0.0)
