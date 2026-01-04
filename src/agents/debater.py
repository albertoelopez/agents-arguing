from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ChatCompletionClient


@dataclass
class DebaterConfig:
    name: str
    stance: Literal["pro", "con"]
    personality: str
    avatar_path: Path | None = None
    voice_sample_path: Path | None = None
    system_prompt: str = field(default="")

    def __post_init__(self) -> None:
        if not self.system_prompt:
            self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        stance_desc = "in favor of" if self.stance == "pro" else "against"
        return f"""You are {self.name}, a skilled debater arguing {stance_desc} the given topic.

Your personality: {self.personality}

Guidelines:
- Present clear, logical arguments with evidence when possible
- Directly address and counter your opponent's points
- Stay focused on the topic
- Be persuasive but respectful
- Keep responses concise (2-3 paragraphs max)
- Use rhetorical techniques effectively

You are currently in a formal debate. Respond only with your argument, no meta-commentary."""


class DebaterAgent:
    def __init__(
        self,
        config: DebaterConfig,
        model_client: ChatCompletionClient,
    ):
        self.config = config
        self.model_client = model_client
        self._agent = self._create_agent()
        self._conversation_history: list[dict] = []

    def _create_agent(self) -> AssistantAgent:
        return AssistantAgent(
            name=self.config.name,
            model_client=self.model_client,
            system_message=self.config.system_prompt,
        )

    async def generate_argument(
        self,
        topic: str,
        opponent_argument: str | None = None,
        round_number: int = 1,
    ) -> str:
        if round_number == 1 and opponent_argument is None:
            prompt = f"""Topic: "{topic}"

This is your opening statement. Present your strongest argument {"in favor of" if self.config.stance == "pro" else "against"} the topic."""
        else:
            prompt = f"""Topic: "{topic}"

Your opponent just argued:
"{opponent_argument}"

Respond to their argument and strengthen your position. This is round {round_number}."""

        self._conversation_history.append({"role": "user", "content": prompt})

        response = await self._agent.on_messages(
            [TextMessage(content=prompt, source="moderator")],
            cancellation_token=None,
        )

        argument = response.chat_message.content
        self._conversation_history.append({"role": "assistant", "content": argument})

        return argument

    async def generate_closing(self, topic: str) -> str:
        prompt = f"""Topic: "{topic}"

Deliver your closing statement. Summarize your key points and make a final compelling case for your position."""

        response = await self._agent.on_messages(
            [TextMessage(content=prompt, source="moderator")],
            cancellation_token=None,
        )

        return response.chat_message.content

    def reset(self) -> None:
        self._conversation_history.clear()
        self._agent = self._create_agent()
