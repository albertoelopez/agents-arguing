import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.agents.debate_manager import DebateManager, DebateTurn, DebateResult
from src.agents.debater import DebaterConfig
from src.agents.moderator import ModeratorConfig
from src.pipeline.audio_pipeline import AudioPipeline, AudioSegment
from src.video.generator import EchoMimicGenerator, VideoClip
from src.video.composer import VideoComposer, DebateSegment
from src.config import settings


@dataclass
class OrchestratorConfig:
    topic: str
    pro_name: str
    pro_personality: str
    pro_avatar: Path
    pro_voice: Path | None = None
    con_name: str = "Opponent"
    con_personality: str = "Critical thinker"
    con_avatar: Path | None = None
    con_voice: Path | None = None
    num_rounds: int = 3
    output_dir: Path | None = None


class DebateOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.output_dir = config.output_dir or settings.output_dir
        self.console = Console()

        self._debate_manager: DebateManager | None = None
        self._audio_pipeline: AudioPipeline | None = None
        self._video_generator: EchoMimicGenerator | None = None
        self._video_composer: VideoComposer | None = None

    async def initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pro_config = DebaterConfig(
            name=self.config.pro_name,
            stance="pro",
            personality=self.config.pro_personality,
            avatar_path=self.config.pro_avatar,
            voice_sample_path=self.config.pro_voice,
        )

        con_config = DebaterConfig(
            name=self.config.con_name,
            stance="con",
            personality=self.config.con_personality,
            avatar_path=self.config.con_avatar,
            voice_sample_path=self.config.con_voice,
        )

        self._debate_manager = DebateManager(
            topic=self.config.topic,
            pro_config=pro_config,
            con_config=con_config,
            num_rounds=self.config.num_rounds,
        )

        self._audio_pipeline = AudioPipeline(
            output_dir=self.output_dir / "audio",
        )
        await self._audio_pipeline.initialize()

        if self.config.pro_voice:
            self._audio_pipeline.register_voice(self.config.pro_name, self.config.pro_voice)
        if self.config.con_voice:
            self._audio_pipeline.register_voice(self.config.con_name, self.config.con_voice)

        self._video_generator = EchoMimicGenerator()
        await self._video_generator.initialize()

        self._video_composer = VideoComposer()

    async def shutdown(self) -> None:
        if self._audio_pipeline:
            await self._audio_pipeline.shutdown()
        if self._video_generator:
            await self._video_generator.shutdown()

    async def run(
        self,
        on_turn: Callable[[DebateTurn], None] | None = None,
    ) -> Path:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            debate_task = progress.add_task("Running debate...", total=None)
            debate_result = await self._run_debate(on_turn)
            progress.update(debate_task, completed=True, description="Debate complete")

            audio_task = progress.add_task("Generating audio...", total=len(debate_result.turns))
            audio_segments = await self._generate_audio(debate_result, progress, audio_task)
            progress.update(audio_task, description="Audio complete")

            video_task = progress.add_task("Generating video clips...", total=len(audio_segments))
            video_clips = await self._generate_videos(audio_segments, debate_result, progress, video_task)
            progress.update(video_task, description="Videos complete")

            compose_task = progress.add_task("Composing final video...", total=None)
            final_path = await self._compose_final_video(video_clips, debate_result)
            progress.update(compose_task, completed=True, description="Done!")

        return final_path

    async def _run_debate(
        self,
        on_turn: Callable[[DebateTurn], None] | None,
    ) -> DebateResult:
        async for turn in self._debate_manager.run_debate_stream():
            self.console.print(f"[bold]{turn.speaker}[/bold]: {turn.content[:100]}...")
            if on_turn:
                on_turn(turn)

        return self._debate_manager.result

    async def _generate_audio(
        self,
        debate_result: DebateResult,
        progress: Progress,
        task_id: int,
    ) -> list[AudioSegment]:
        segments = []

        for i, turn in enumerate(debate_result.turns):
            segment = await self._audio_pipeline.synthesize_speech(
                text=turn.content,
                speaker=turn.speaker,
                segment_id=f"turn_{i:03d}",
            )
            segments.append(segment)
            progress.update(task_id, advance=1)

        return segments

    async def _generate_videos(
        self,
        audio_segments: list[AudioSegment],
        debate_result: DebateResult,
        progress: Progress,
        task_id: int,
    ) -> list[tuple[DebateTurn, VideoClip]]:
        video_clips = []
        video_dir = self.output_dir / "video_clips"
        video_dir.mkdir(exist_ok=True)

        for i, (turn, audio) in enumerate(zip(debate_result.turns, audio_segments)):
            if turn.stance == "pro":
                avatar = self.config.pro_avatar
            elif turn.stance == "con":
                avatar = self.config.con_avatar
            else:
                avatar = self.config.pro_avatar

            if avatar and audio.file_path:
                output_path = video_dir / f"clip_{i:03d}_{turn.speaker}.mp4"

                clip = await self._video_generator.generate(
                    avatar_image=avatar,
                    audio=audio.file_path,
                    output_path=output_path,
                )
                video_clips.append((turn, clip))

            progress.update(task_id, advance=1)

        return video_clips

    async def _compose_final_video(
        self,
        video_clips: list[tuple[DebateTurn, VideoClip]],
        debate_result: DebateResult,
    ) -> Path:
        segments = []

        for turn, clip in video_clips:
            segment = DebateSegment(
                speaker_name=turn.speaker,
                video_clip=clip,
                is_pro=(turn.stance == "pro"),
            )
            segments.append(segment)

        output_path = self.output_dir / "final_debate.mp4"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._video_composer.compose_debate(
                segments=segments,
                output_path=output_path,
                title=debate_result.topic,
            ),
        )

        return output_path
