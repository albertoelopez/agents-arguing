import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_assets_dir(tmp_path: Path) -> Path:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "avatars").mkdir()
    (assets / "audio").mkdir()
    (assets / "output").mkdir()
    return assets


@pytest.fixture
def sample_audio() -> np.ndarray:
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def mock_tts_service():
    service = AsyncMock()
    service.initialize = AsyncMock()
    service.shutdown = AsyncMock()
    service.synthesize = AsyncMock(return_value=MagicMock(
        audio=np.zeros(22050, dtype=np.float32),
        sample_rate=22050,
        duration=1.0,
    ))
    return service


@pytest.fixture
def mock_stt_service():
    service = AsyncMock()
    service.initialize = AsyncMock()
    service.shutdown = AsyncMock()
    service.transcribe = AsyncMock(return_value=MagicMock(
        text="Hello world",
        segments=[],
        language="en",
        duration=1.0,
    ))
    return service


@pytest.fixture
def mock_video_generator():
    generator = AsyncMock()
    generator.initialize = AsyncMock()
    generator.shutdown = AsyncMock()
    generator.generate = AsyncMock(return_value=MagicMock(
        path=Path("/tmp/test.mp4"),
        duration=1.0,
        width=1280,
        height=720,
        fps=25,
    ))
    return generator


@pytest.fixture
def sample_debate_config():
    return {
        "topic": "AI will benefit humanity",
        "pro_name": "Alex",
        "con_name": "Jordan",
        "num_rounds": 2,
    }
