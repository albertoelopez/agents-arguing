from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.services.base import BaseService
from src.config import settings


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration: float


class TTSService(BaseService):
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        speaker_wav: Path | None = None,
        language: str = "en",
    ) -> SynthesisResult:
        pass

    @abstractmethod
    async def clone_voice(
        self,
        speaker_wav: Path,
        speaker_name: str,
    ) -> str:
        pass


class XTTSService(TTSService):
    def __init__(
        self,
        device: str = settings.tts_device,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self._tts = None
        self._speaker_embeddings: dict[str, np.ndarray] = {}

    async def initialize(self) -> None:
        if self._initialized:
            return

        from TTS.api import TTS

        self._tts = TTS(model_name=self.model_name).to(self.device)
        self._initialized = True

    async def shutdown(self) -> None:
        self._tts = None
        self._speaker_embeddings.clear()
        self._initialized = False

    async def synthesize(
        self,
        text: str,
        speaker_wav: Path | None = None,
        language: str = "en",
    ) -> SynthesisResult:
        if not self._initialized:
            await self.initialize()

        if speaker_wav is None:
            audio = self._tts.tts(text=text, language=language)
        else:
            audio = self._tts.tts(
                text=text,
                speaker_wav=str(speaker_wav),
                language=language,
            )

        audio_array = np.array(audio, dtype=np.float32)
        sample_rate = self._tts.synthesizer.output_sample_rate
        duration = len(audio_array) / sample_rate

        return SynthesisResult(
            audio=audio_array,
            sample_rate=sample_rate,
            duration=duration,
        )

    async def clone_voice(
        self,
        speaker_wav: Path,
        speaker_name: str,
    ) -> str:
        if not self._initialized:
            await self.initialize()

        gpt_cond_latent, speaker_embedding = self._tts.synthesizer.tts_model.get_conditioning_latents(
            audio_path=[str(speaker_wav)]
        )

        self._speaker_embeddings[speaker_name] = {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
        }

        return speaker_name

    async def synthesize_with_cloned_voice(
        self,
        text: str,
        speaker_name: str,
        language: str = "en",
    ) -> SynthesisResult:
        if not self._initialized:
            await self.initialize()

        if speaker_name not in self._speaker_embeddings:
            raise ValueError(f"Speaker '{speaker_name}' not found. Clone voice first.")

        embeddings = self._speaker_embeddings[speaker_name]

        audio = self._tts.synthesizer.tts_model.inference(
            text=text,
            language=language,
            gpt_cond_latent=embeddings["gpt_cond_latent"],
            speaker_embedding=embeddings["speaker_embedding"],
        )

        audio_array = np.array(audio["wav"], dtype=np.float32)
        sample_rate = self._tts.synthesizer.output_sample_rate
        duration = len(audio_array) / sample_rate

        return SynthesisResult(
            audio=audio_array,
            sample_rate=sample_rate,
            duration=duration,
        )
