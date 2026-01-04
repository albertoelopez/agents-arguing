import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    AudioRawFrame,
    TranscriptionFrame,
    EndFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.transports.base_transport import TransportParams

from src.config import settings


@dataclass
class DebaterVoice:
    name: str
    stance: str
    personality: str
    voice_id: str | None = None


class DebateContextProcessor(FrameProcessor):
    def __init__(
        self,
        pro_debater: DebaterVoice,
        con_debater: DebaterVoice,
        topic: str,
    ):
        super().__init__()
        self.pro = pro_debater
        self.con = con_debater
        self.topic = topic
        self.current_speaker = "pro"
        self.conversation_history: list[dict] = []
        self.turn_count = 0

    async def process_frame(self, frame: Frame, direction: str) -> AsyncIterator[Frame]:
        if isinstance(frame, TranscriptionFrame):
            user_input = frame.text

            if self._is_switch_command(user_input):
                self._switch_speaker()
                yield TextFrame(text=f"Switching to {self._get_current_debater().name}...")
            else:
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input,
                })

            yield frame
        else:
            yield frame

    def _is_switch_command(self, text: str) -> bool:
        switch_phrases = ["next speaker", "switch", "your turn", "respond"]
        return any(phrase in text.lower() for phrase in switch_phrases)

    def _switch_speaker(self) -> None:
        self.current_speaker = "con" if self.current_speaker == "pro" else "pro"
        self.turn_count += 1

    def _get_current_debater(self) -> DebaterVoice:
        return self.pro if self.current_speaker == "pro" else self.con

    def get_system_prompt(self) -> str:
        debater = self._get_current_debater()
        stance_text = "in favor of" if debater.stance == "pro" else "against"

        return f"""You are {debater.name}, debating {stance_text} the topic: "{self.topic}"

Your personality: {debater.personality}

Guidelines:
- Present clear, logical arguments
- Respond directly to opponent's points when provided
- Keep responses conversational (2-3 sentences for real-time)
- Be persuasive but respectful
- You're in a live debate, so be engaging

Previous context will be provided. Respond naturally as {debater.name}."""


class VoiceDebatePipeline:
    def __init__(
        self,
        topic: str,
        pro_name: str = "Alex",
        pro_personality: str = "Optimistic, data-driven",
        con_name: str = "Jordan",
        con_personality: str = "Skeptical, philosophical",
        on_transcription: Callable[[str, str], None] | None = None,
        on_response: Callable[[str, str], None] | None = None,
    ):
        self.topic = topic
        self.on_transcription = on_transcription
        self.on_response = on_response

        self.pro_debater = DebaterVoice(
            name=pro_name,
            stance="pro",
            personality=pro_personality,
        )
        self.con_debater = DebaterVoice(
            name=con_name,
            stance="con",
            personality=con_personality,
        )

        self._pipeline: Pipeline | None = None
        self._task: PipelineTask | None = None
        self._context_processor: DebateContextProcessor | None = None

    async def create_pipeline(
        self,
        transport,
        stt_service,
        tts_service,
    ) -> Pipeline:
        self._context_processor = DebateContextProcessor(
            pro_debater=self.pro_debater,
            con_debater=self.con_debater,
            topic=self.topic,
        )

        llm = AnthropicLLMService(
            api_key=settings.anthropic_api_key,
            model="claude-sonnet-4-20250514",
        )

        pipeline = Pipeline([
            transport.input(),
            stt_service,
            self._context_processor,
            llm,
            tts_service,
            transport.output(),
        ])

        self._pipeline = pipeline
        return pipeline

    async def run(self, transport, stt_service, tts_service) -> None:
        pipeline = await self.create_pipeline(transport, stt_service, tts_service)

        self._task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

        runner = PipelineRunner()
        await runner.run(self._task)

    async def stop(self) -> None:
        if self._task:
            await self._task.cancel()

    def switch_speaker(self) -> str:
        if self._context_processor:
            self._context_processor._switch_speaker()
            return self._context_processor._get_current_debater().name
        return ""

    def get_current_speaker(self) -> str:
        if self._context_processor:
            return self._context_processor._get_current_debater().name
        return self.pro_debater.name
