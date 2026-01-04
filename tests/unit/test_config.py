from pathlib import Path

import pytest

from src.config import Settings


class TestSettings:
    def test_default_values(self):
        settings = Settings(
            anthropic_api_key="test",
            _env_file=None,
        )

        assert settings.stt_model == "faster-whisper"
        assert settings.stt_model_size == "large-v3"
        assert settings.tts_model == "xtts_v2"
        assert settings.video_model == "echomimic_v3"

    def test_output_dir_is_path(self):
        settings = Settings(_env_file=None)
        assert isinstance(settings.output_dir, Path)

    def test_ensure_dirs_creates_directories(self, tmp_path: Path):
        settings = Settings(
            output_dir=tmp_path / "output",
            avatars_dir=tmp_path / "avatars",
            _env_file=None,
        )

        settings.ensure_dirs()

        assert settings.output_dir.exists()
        assert settings.avatars_dir.exists()
