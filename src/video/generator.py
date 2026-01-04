from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from src.services.base import BaseService
from src.config import settings


@dataclass
class VideoClip:
    path: Path
    duration: float
    width: int
    height: int
    fps: int


class VideoGenerator(BaseService):
    @abstractmethod
    async def generate(
        self,
        avatar_image: Path,
        audio: Path,
        output_path: Path,
    ) -> VideoClip:
        pass


class EchoMimicGenerator(VideoGenerator):
    def __init__(
        self,
        device: str = settings.video_device,
        model_path: str | None = None,
    ):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self._pipeline = None

    async def initialize(self) -> None:
        if self._initialized:
            return

        try:
            from src.video.echomimic_wrapper import EchoMimicPipeline

            self._pipeline = EchoMimicPipeline(
                device=self.device,
                model_path=self.model_path,
            )
            await self._pipeline.load_models()
        except ImportError:
            raise ImportError(
                "EchoMimicV3 not installed. Run: pip install -e '.[echomimic]' "
                "and clone https://github.com/antgroup/echomimic_v3"
            )

        self._initialized = True

    async def shutdown(self) -> None:
        if self._pipeline:
            await self._pipeline.unload_models()
        self._pipeline = None
        self._initialized = False

    async def generate(
        self,
        avatar_image: Path,
        audio: Path,
        output_path: Path,
    ) -> VideoClip:
        if not self._initialized:
            await self.initialize()

        result = await self._pipeline.generate_video(
            image_path=avatar_image,
            audio_path=audio,
            output_path=output_path,
        )

        return VideoClip(
            path=output_path,
            duration=result["duration"],
            width=result["width"],
            height=result["height"],
            fps=result["fps"],
        )
