import asyncio
import sys
import math
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image


ECHOMIMIC_V3_PATH = Path(__file__).parent.parent.parent / "external" / "echomimic_v3"


class EchoMimicV3Pipeline:
    def __init__(
        self,
        device: str = "cuda",
        models_path: Path | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.models_path = models_path or ECHOMIMIC_V3_PATH / "models"
        self.dtype = dtype
        self._models_loaded = False
        self._pipeline = None
        self._wav2vec_processor = None
        self._wav2vec_model = None
        self._vae = None

    async def load_models(self) -> None:
        if self._models_loaded:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_models_sync)
        self._models_loaded = True

    def _load_models_sync(self) -> None:
        if str(ECHOMIMIC_V3_PATH) not in sys.path:
            sys.path.insert(0, str(ECHOMIMIC_V3_PATH))

        try:
            from omegaconf import OmegaConf
            from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
            from diffusers import FlowMatchEulerDiscreteScheduler

            from src.wan_vae import AutoencoderKLWan
            from src.wan_image_encoder import CLIPModel
            from src.wan_text_encoder import WanT5EncoderModel
            from src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
            from src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline
            from src.utils import filter_kwargs

            cfg_path = ECHOMIMIC_V3_PATH / "config" / "config.yaml"
            cfg = OmegaConf.load(str(cfg_path))

            model_name = self.models_path / "Wan2.1-Fun-V1.1-1.3B-InP"
            transformer_path = self.models_path / "transformer" / "transformer" / "diffusion_pytorch_model.safetensors"

            transformer = WanTransformerAudioMask3DModel.from_pretrained(
                str(model_name / cfg['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
                transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
                torch_dtype=self.dtype,
            )

            if transformer_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(str(transformer_path))
                state_dict = state_dict.get("state_dict", state_dict)
                transformer.load_state_dict(state_dict, strict=False)

            self._vae = AutoencoderKLWan.from_pretrained(
                str(model_name / cfg['vae_kwargs'].get('vae_subpath', 'vae')),
                additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
            ).to(self.dtype)

            tokenizer = AutoTokenizer.from_pretrained(
                str(model_name / cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
            )

            text_encoder = WanT5EncoderModel.from_pretrained(
                str(model_name / cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
                torch_dtype=self.dtype,
            ).eval()

            clip_image_encoder = CLIPModel.from_pretrained(
                str(model_name / cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            ).to(self.dtype).eval()

            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(cfg['scheduler_kwargs']))
            )

            self._pipeline = WanFunInpaintAudioPipeline(
                transformer=transformer,
                vae=self._vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scheduler=scheduler,
                clip_image_encoder=clip_image_encoder,
            )
            self._pipeline.to(device=self.device)

            wav2vec_path = self.models_path / "wav2vec2-base-960h"
            self._wav2vec_processor = Wav2Vec2Processor.from_pretrained(str(wav2vec_path))
            self._wav2vec_model = Wav2Vec2Model.from_pretrained(str(wav2vec_path)).eval()
            self._wav2vec_model.requires_grad_(False)

        except Exception as e:
            raise RuntimeError(f"Failed to load EchoMimicV3 models: {e}")

    async def unload_models(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
        if self._wav2vec_model is not None:
            del self._wav2vec_model
        if self._wav2vec_processor is not None:
            del self._wav2vec_processor
        if self._vae is not None:
            del self._vae
        self._pipeline = None
        self._wav2vec_model = None
        self._wav2vec_processor = None
        self._vae = None
        self._models_loaded = False
        torch.cuda.empty_cache()

    async def generate_video(
        self,
        image_path: Path,
        audio_path: Path,
        output_path: Path,
        prompt: str = "A person speaking naturally with clear lip movements.",
        fps: int = 25,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.0,
        audio_guidance_scale: float = 2.9,
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
            prompt,
            fps,
            num_inference_steps,
            guidance_scale,
            audio_guidance_scale,
        )

        return result

    def _extract_audio_features(self, audio_path: Path) -> torch.Tensor:
        import librosa
        sr = 16000
        audio_segment, sample_rate = librosa.load(str(audio_path), sr=sr)
        input_values = self._wav2vec_processor(
            audio_segment, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        features = self._wav2vec_model(input_values).last_hidden_state
        return features.squeeze(0)

    def _get_sample_size(self, image: Image.Image, default_size: list[int]) -> tuple[int, int]:
        width, height = image.size
        original_area = width * height
        default_area = default_size[0] * default_size[1]

        if default_area < original_area:
            ratio = math.sqrt(original_area / default_area)
            width = width / ratio // 16 * 16
            height = height / ratio // 16 * 16
        else:
            width = width // 16 * 16
            height = height // 16 * 16

        return int(height), int(width)

    def _generate_video_sync(
        self,
        image_path: Path,
        audio_path: Path,
        output_path: Path,
        prompt: str,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        audio_guidance_scale: float,
    ) -> dict:
        from moviepy import VideoFileClip, AudioFileClip

        if str(ECHOMIMIC_V3_PATH) not in sys.path:
            sys.path.insert(0, str(ECHOMIMIC_V3_PATH))
        from src.utils import get_image_to_video_latent3, save_videos_grid

        ref_img = Image.open(image_path).convert("RGB")

        audio_clip = AudioFileClip(str(audio_path))
        duration = audio_clip.duration

        audio_features = self._extract_audio_features(audio_path)
        audio_embeds = audio_features.unsqueeze(0).to(device=self.device, dtype=self.dtype)

        video_length = int(duration * fps)
        video_length = (
            int((video_length - 1) // self._vae.config.temporal_compression_ratio * self._vae.config.temporal_compression_ratio) + 1
            if video_length != 1 else 1
        )

        sample_size = [768, 768]
        sample_height, sample_width = self._get_sample_size(ref_img, sample_size)

        negative_prompt = "Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands."

        generator = torch.Generator(device=self.device).manual_seed(42)

        input_video, input_video_mask, clip_image = get_image_to_video_latent3(
            ref_img, None, video_length=video_length, sample_size=[sample_height, sample_width]
        )

        sample = self._pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            audio_embeds=audio_embeds,
            audio_scale=1.0,
            height=sample_height,
            width=sample_width,
            generator=generator,
            guidance_scale=guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            num_inference_steps=num_inference_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
        ).videos

        temp_video = output_path.with_suffix(".temp.mp4")
        save_videos_grid(sample[:, :, :video_length], str(temp_video), fps=fps)

        video_clip = VideoFileClip(str(temp_video))
        audio_clip = audio_clip.subclipped(0, video_length / fps)
        final = video_clip.with_audio(audio_clip)
        final.write_videofile(str(output_path), codec="libx264", audio_codec="aac", threads=2)

        temp_video.unlink(missing_ok=True)
        video_clip.close()
        final.close()

        return {
            "duration": duration,
            "width": sample_width,
            "height": sample_height,
            "fps": fps,
            "num_frames": video_length,
        }


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
        from moviepy import VideoFileClip, AudioFileClip

        temp_video = output_path.with_suffix(".temp.mp4")

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)
        writer.release()

        video = VideoFileClip(str(temp_video))
        audio = AudioFileClip(str(audio_path))
        final = video.with_audio(audio)
        final.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

        temp_video.unlink()
