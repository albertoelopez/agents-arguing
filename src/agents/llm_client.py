from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio

import ollama

from src.config import settings


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: list[ChatMessage], system: str | None = None) -> str:
        pass


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self._client = ollama.AsyncClient()

    async def chat(self, messages: list[ChatMessage], system: str | None = None) -> str:
        ollama_messages = []

        if system:
            ollama_messages.append({"role": "system", "content": system})

        for msg in messages:
            ollama_messages.append({"role": msg.role, "content": msg.content})

        response = await self._client.chat(
            model=self.model,
            messages=ollama_messages,
        )

        return response["message"]["content"]


class AnthropicClientWrapper(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    async def chat(self, messages: list[ChatMessage], system: str | None = None) -> str:
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

            api_messages = [{"role": m.role, "content": m.content} for m in messages]

            response = await client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system or "",
                messages=api_messages,
            )

            return response.content[0].text
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")


class GroqClient(LLMClient):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model

    async def chat(self, messages: list[ChatMessage], system: str | None = None) -> str:
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=settings.groq_api_key)

            api_messages = []
            if system:
                api_messages.append({"role": "system", "content": system})

            for msg in messages:
                api_messages.append({"role": msg.role, "content": msg.content})

            response = await client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                max_tokens=1024,
            )

            return response.choices[0].message.content
        except ImportError:
            raise ImportError("Install groq: pip install groq")


def get_llm_client(provider: str = "ollama", model: str | None = None) -> LLMClient:
    if provider == "ollama":
        return OllamaClient(model=model or "llama3.1:8b")
    elif provider == "anthropic":
        return AnthropicClientWrapper(model=model or "claude-sonnet-4-20250514")
    elif provider == "groq":
        return GroqClient(model=model or "llama-3.3-70b-versatile")
    else:
        raise ValueError(f"Unknown provider: {provider}")
