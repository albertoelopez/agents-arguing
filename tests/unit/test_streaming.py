import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.realtime.streaming import (
    StreamingDebateSession,
    StreamEvent,
    StreamEventType,
)


class TestStreamEvent:
    def test_creates_event_with_type_and_data(self):
        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"topic": "Test topic"},
        )

        assert event.type == StreamEventType.DEBATE_START
        assert event.data["topic"] == "Test topic"

    def test_serializes_to_json(self):
        event = StreamEvent(
            type=StreamEventType.TEXT_CHUNK,
            data={"text": "Hello"},
        )

        json_str = event.to_json()
        assert '"type": "text_chunk"' in json_str
        assert '"text": "Hello"' in json_str


class TestStreamingDebateSession:
    @pytest.fixture
    def session(self):
        return StreamingDebateSession(
            topic="Test topic",
            pro_name="Alice",
            con_name="Bob",
            num_rounds=1,
            enable_audio=False,
        )

    def test_initializes_with_config(self, session):
        assert session.topic == "Test topic"
        assert session.pro_config.name == "Alice"
        assert session.con_config.name == "Bob"
        assert session.num_rounds == 1

    def test_is_not_running_initially(self, session):
        assert session.is_running is False

    def test_stop_sets_should_stop_flag(self, session):
        session.stop()
        assert session._should_stop is True

    def test_chunks_text_correctly(self, session):
        text = "This is a test sentence with multiple words that should be chunked"
        chunks = session._chunk_text(text, chunk_size=20)

        assert len(chunks) > 1
        reconstructed = " ".join(chunks)
        assert text in reconstructed or reconstructed.strip() == text


class TestStreamEventTypes:
    def test_all_event_types_have_values(self):
        expected_types = [
            "debate_start",
            "turn_start",
            "text_chunk",
            "turn_end",
            "audio_chunk",
            "audio_complete",
            "debate_end",
            "error",
        ]

        actual_values = [e.value for e in StreamEventType]
        for expected in expected_types:
            assert expected in actual_values
