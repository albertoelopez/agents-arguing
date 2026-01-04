import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable
from enum import Enum

from src.agents.debate_manager_simple import SimpleDebateManager, DebateTurn
from src.agents.debater_simple import DebaterConfig
from src.config import settings


class StreamEventType(Enum):
    DEBATE_START = "debate_start"
    TURN_START = "turn_start"
    TEXT_CHUNK = "text_chunk"
    TURN_END = "turn_end"
    AUDIO_CHUNK = "audio_chunk"
    AUDIO_COMPLETE = "audio_complete"
    DEBATE_END = "debate_end"
    ERROR = "error"


@dataclass
class StreamEvent:
    type: StreamEventType
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
        })


class StreamingDebateSession:
    def __init__(
        self,
        topic: str,
        pro_name: str = "Alex",
        pro_personality: str = "Optimistic, data-driven",
        con_name: str = "Jordan",
        con_personality: str = "Skeptical, philosophical",
        num_rounds: int = 3,
        enable_audio: bool = False,
        llm_provider: str = "ollama",
        llm_model: str = "llama3.1:8b",
    ):
        self.topic = topic
        self.num_rounds = num_rounds
        self.enable_audio = enable_audio
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        self.pro_config = DebaterConfig(
            name=pro_name,
            stance="pro",
            personality=pro_personality,
        )
        self.con_config = DebaterConfig(
            name=con_name,
            stance="con",
            personality=con_personality,
        )

        self._debate_manager: SimpleDebateManager | None = None
        self._is_running = False
        self._should_stop = False

    async def initialize(self) -> None:
        self._debate_manager = SimpleDebateManager(
            topic=self.topic,
            pro_config=self.pro_config,
            con_config=self.con_config,
            num_rounds=self.num_rounds,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
        )

    async def shutdown(self) -> None:
        self._is_running = False

    async def stream_debate(self) -> AsyncIterator[StreamEvent]:
        if not self._debate_manager:
            await self.initialize()

        self._is_running = True
        self._should_stop = False

        yield StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "topic": self.topic,
                "pro": self.pro_config.name,
                "con": self.con_config.name,
                "rounds": self.num_rounds,
            },
        )

        try:
            async for turn in self._debate_manager.run_debate_stream():
                if self._should_stop:
                    break

                yield StreamEvent(
                    type=StreamEventType.TURN_START,
                    data={
                        "speaker": turn.speaker,
                        "stance": turn.stance,
                        "turn_type": turn.turn_type,
                        "round": turn.round_number,
                    },
                )

                chunks = self._chunk_text(turn.content)
                for chunk in chunks:
                    if self._should_stop:
                        break

                    yield StreamEvent(
                        type=StreamEventType.TEXT_CHUNK,
                        data={
                            "speaker": turn.speaker,
                            "text": chunk,
                        },
                    )
                    await asyncio.sleep(0.05)

                yield StreamEvent(
                    type=StreamEventType.TURN_END,
                    data={
                        "speaker": turn.speaker,
                        "full_text": turn.content,
                    },
                )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
            )

        yield StreamEvent(
            type=StreamEventType.DEBATE_END,
            data={"total_turns": len(self._debate_manager.result.turns)},
        )

        self._is_running = False

    async def _stream_audio(self, turn: DebateTurn) -> AsyncIterator[StreamEvent]:
        try:
            result = await self._tts.synthesize(
                text=turn.content,
                language="en",
            )

            chunk_size = result.sample_rate // 4
            audio_data = result.audio

            for i in range(0, len(audio_data), chunk_size):
                if self._should_stop:
                    break

                chunk = audio_data[i:i + chunk_size]
                yield StreamEvent(
                    type=StreamEventType.AUDIO_CHUNK,
                    data={
                        "speaker": turn.speaker,
                        "sample_rate": result.sample_rate,
                        "chunk_index": i // chunk_size,
                        "audio_base64": self._encode_audio(chunk),
                    },
                )
                await asyncio.sleep(0.1)

            yield StreamEvent(
                type=StreamEventType.AUDIO_COMPLETE,
                data={
                    "speaker": turn.speaker,
                    "duration": result.duration,
                },
            )
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": f"Audio generation failed: {e}"},
            )

    def _chunk_text(self, text: str, chunk_size: int = 50) -> list[str]:
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _encode_audio(self, audio_data) -> str:
        import base64
        import numpy as np

        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        return base64.b64encode(audio_bytes).decode("utf-8")

    def stop(self) -> None:
        self._should_stop = True

    @property
    def is_running(self) -> bool:
        return self._is_running
