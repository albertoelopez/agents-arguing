import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.services.tts import TTSService, XTTSService, SynthesisResult
from src.config import settings


@dataclass
class AudioSegment:
    speaker: str
    text: str
    audio: np.ndarray
    sample_rate: int
    duration: float
    file_path: Path | None = None


class AudioPipeline:
    def __init__(
        self,
        tts_service: TTSService | None = None,
        output_dir: Path | None = None,
    ):
        self.tts = tts_service or XTTSService()
        self.output_dir = output_dir or settings.output_dir / "audio"
        self._voice_samples: dict[str, Path] = {}

    async def initialize(self) -> None:
        await self.tts.initialize()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def shutdown(self) -> None:
        await self.tts.shutdown()

    def register_voice(self, speaker_name: str, voice_sample: Path) -> None:
        self._voice_samples[speaker_name] = voice_sample

    async def synthesize_speech(
        self,
        text: str,
        speaker: str,
        segment_id: str,
        language: str = "en",
    ) -> AudioSegment:
        voice_sample = self._voice_samples.get(speaker)

        result = await self.tts.synthesize(
            text=text,
            speaker_wav=voice_sample,
            language=language,
        )

        file_path = self.output_dir / f"{segment_id}_{speaker}.wav"
        await self._save_audio(result, file_path)

        return AudioSegment(
            speaker=speaker,
            text=text,
            audio=result.audio,
            sample_rate=result.sample_rate,
            duration=result.duration,
            file_path=file_path,
        )

    async def _save_audio(self, result: SynthesisResult, path: Path) -> None:
        import soundfile as sf

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: sf.write(str(path), result.audio, result.sample_rate),
        )

    async def process_debate_turns(
        self,
        turns: list[dict],
        on_segment_ready: Callable[[AudioSegment], None] | None = None,
    ) -> list[AudioSegment]:
        segments = []

        for i, turn in enumerate(turns):
            segment = await self.synthesize_speech(
                text=turn["content"],
                speaker=turn["speaker"],
                segment_id=f"turn_{i:03d}",
            )
            segments.append(segment)

            if on_segment_ready:
                on_segment_ready(segment)

        return segments
