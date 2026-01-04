from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.agents.llm_client import LLMClient, ChatMessage


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


class SimpleDebaterAgent:
    def __init__(self, config: DebaterConfig, llm_client: LLMClient):
        self.config = config
        self.llm = llm_client
        self._conversation_history: list[ChatMessage] = []

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

        self._conversation_history.append(ChatMessage(role="user", content=prompt))

        response = await self.llm.chat(
            messages=self._conversation_history,
            system=self.config.system_prompt,
        )

        self._conversation_history.append(ChatMessage(role="assistant", content=response))

        return response

    async def generate_closing(self, topic: str) -> str:
        prompt = f"""Topic: "{topic}"

Deliver your closing statement. Summarize your key points and make a final compelling case for your position."""

        self._conversation_history.append(ChatMessage(role="user", content=prompt))

        response = await self.llm.chat(
            messages=self._conversation_history,
            system=self.config.system_prompt,
        )

        return response

    def reset(self) -> None:
        self._conversation_history.clear()
