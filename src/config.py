from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    groq_api_key: str = Field(default="")

    stt_model: Literal["faster-whisper", "parakeet"] = "faster-whisper"
    stt_model_size: str = "large-v3"
    stt_device: Literal["cuda", "cpu"] = "cuda"

    tts_model: Literal["xtts_v2", "piper"] = "xtts_v2"
    tts_device: Literal["cuda", "cpu"] = "cuda"

    video_model: Literal["echomimic_v3"] = "echomimic_v3"
    video_device: Literal["cuda", "cpu"] = "cuda"

    output_dir: Path = Path("./assets/output")
    avatars_dir: Path = Path("./assets/avatars")

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.avatars_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
