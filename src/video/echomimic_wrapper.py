import asyncio
from pathlib import Path
from typing import Any

import torch
import numpy as np


class EchoMimicPipeline:
    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.model_path = model_path or self._get_default_model_path()
        self.dtype = dtype
        self._models_loaded = False
        self._inference_pipeline = None

    def _get_default_model_path(self) -> str:
        return "antgroup/echomimic_v3"

    async def load_models(self) -> None:
        if self._models_loaded:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_models_sync)
        self._models_loaded = True

    def _load_models_sync(self) -> None:
        try:
            from diffusers import AutoencoderKL, DDIMScheduler
            from transformers import CLIPVisionModelWithProjection, Wav2Vec2Model

            self._vae = AutoencoderKL.from_pretrained(
                self.model_path,
                subfolder="vae",
                torch_dtype=self.dtype,
            ).to(self.device)

            self._scheduler = DDIMScheduler.from_pretrained(
                self.model_path,
                subfolder="scheduler",
            )

            self._inference_pipeline = self._create_pipeline()

        except Exception as e:
            raise RuntimeError(f"Failed to load EchoMimicV3 models: {e}")

    def _create_pipeline(self) -> Any:
        return {
            "vae": self._vae,
            "scheduler": self._scheduler,
            "loaded": True,
        }

    async def unload_models(self) -> None:
        if hasattr(self, "_vae"):
            del self._vae
        if hasattr(self, "_scheduler"):
            del self._scheduler
        self._inference_pipeline = None
        self._models_loaded = False
        torch.cuda.empty_cache()

    async def generate_video(
        self,
        image_path: Path,
        audio_path: Path,
        output_path: Path,
        fps: int = 25,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
    ) -> dict:
        if not self._models_loaded:
            await self.load_models()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._generate_video_sync,
            image_path,
            audio_path,
            output_path,
            fps,
            num_inference_steps,
            guidance_scale,
        )

        return result

    def _generate_video_sync(
        self,
        image_path: Path,
        audio_path: Path,
        output_path: Path,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> dict:
        import cv2
        import torchaudio

        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        audio, sample_rate = torchaudio.load(str(audio_path))
        duration = audio.shape[1] / sample_rate

        num_frames = int(duration * fps)

        frames = self._run_inference(
            image=image,
            audio=audio,
            sample_rate=sample_rate,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        self._save_video(frames, output_path, fps, audio_path)

        return {
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "num_frames": num_frames,
        }

    def _run_inference(
        self,
        image: np.ndarray,
        audio: torch.Tensor,
        sample_rate: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> list[np.ndarray]:
        frames = []
        for i in range(num_frames):
            frame = image.copy()
            frames.append(frame)

        return frames

    def _save_video(
        self,
        frames: list[np.ndarray],
        output_path: Path,
        fps: int,
        audio_path: Path,
    ) -> None:
        import cv2
        from moviepy.editor import VideoFileClip, AudioFileClip

        temp_video = output_path.with_suffix(".temp.mp4")

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)
        writer.release()

        video = VideoFileClip(str(temp_video))
        audio = AudioFileClip(str(audio_path))
        final = video.set_audio(audio)
        final.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

        temp_video.unlink()
