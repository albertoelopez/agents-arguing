from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.video.composer import VideoComposer, DebateSegment
from src.video.generator import VideoClip


class TestVideoComposer:
    @pytest.fixture
    def composer(self):
        return VideoComposer(resolution=(1280, 720), fps=25)

    @pytest.fixture
    def mock_video_clip(self, tmp_path: Path):
        return VideoClip(
            path=tmp_path / "test.mp4",
            duration=5.0,
            width=640,
            height=480,
            fps=25,
        )

    def test_initializes_with_default_resolution(self):
        composer = VideoComposer()
        assert composer.resolution == (1280, 720)
        assert composer.fps == 25

    def test_initializes_with_custom_resolution(self):
        composer = VideoComposer(resolution=(1920, 1080), fps=30)
        assert composer.resolution == (1920, 1080)
        assert composer.fps == 30

    def test_creates_debate_segment(self, mock_video_clip):
        segment = DebateSegment(
            speaker_name="Alex",
            video_clip=mock_video_clip,
            is_pro=True,
        )

        assert segment.speaker_name == "Alex"
        assert segment.is_pro is True
        assert segment.video_clip == mock_video_clip
