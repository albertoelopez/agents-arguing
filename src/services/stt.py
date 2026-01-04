from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from src.services.base import BaseService
from src.config import settings


@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptionResult:
    text: str
    segments: list[TranscriptionSegment]
    language: str
    duration: float


class STTService(BaseService):
    @abstractmethod
    async def transcribe(
        self,
        audio: np.ndarray | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionSegment]:
        pass


class FasterWhisperSTT(STTService):
    def __init__(
        self,
        model_size: str = settings.stt_model_size,
        device: str = settings.stt_device,
        compute_type: str = "float16",
    ):
        super().__init__()
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    async def initialize(self) -> None:
        if self._initialized:
            return

        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._initialized = True

    async def shutdown(self) -> None:
        self._model = None
        self._initialized = False

    async def transcribe(
        self,
        audio: np.ndarray | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        if not self._initialized:
            await self.initialize()

        if isinstance(audio, Path):
            audio_input = str(audio)
        else:
            audio_input = audio

        segments_raw, info = self._model.transcribe(
            audio_input,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        segments = []
        full_text_parts = []

        for seg in segments_raw:
            segments.append(
                TranscriptionSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    confidence=seg.avg_logprob,
                )
            )
            full_text_parts.append(seg.text.strip())

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            segments=segments,
            language=info.language,
            duration=info.duration,
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionSegment]:
        if not self._initialized:
            await self.initialize()

        buffer = np.array([], dtype=np.float32)
        chunk_duration = 3.0
        sample_rate = 16000
        chunk_samples = int(chunk_duration * sample_rate)

        async for chunk in audio_stream:
            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= chunk_samples:
                audio_chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                segments, _ = self._model.transcribe(
                    audio_chunk,
                    language=language,
                    beam_size=3,
                    vad_filter=True,
                )

                for seg in segments:
                    yield TranscriptionSegment(
                        text=seg.text.strip(),
                        start=seg.start,
                        end=seg.end,
                        confidence=seg.avg_logprob,
                    )
