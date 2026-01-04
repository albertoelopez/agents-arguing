import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.debater import DebaterAgent, DebaterConfig


class TestDebaterConfig:
    def test_creates_system_prompt_for_pro_stance(self):
        config = DebaterConfig(
            name="Alex",
            stance="pro",
            personality="Optimistic",
        )

        assert "in favor of" in config.system_prompt
        assert "Alex" in config.system_prompt
        assert "Optimistic" in config.system_prompt

    def test_creates_system_prompt_for_con_stance(self):
        config = DebaterConfig(
            name="Jordan",
            stance="con",
            personality="Skeptical",
        )

        assert "against" in config.system_prompt
        assert "Jordan" in config.system_prompt

    def test_uses_custom_system_prompt_when_provided(self):
        custom_prompt = "You are a custom debater."
        config = DebaterConfig(
            name="Test",
            stance="pro",
            personality="Test",
            system_prompt=custom_prompt,
        )

        assert config.system_prompt == custom_prompt


class TestDebaterAgent:
    @pytest.fixture
    def mock_model_client(self):
        client = MagicMock()
        return client

    @pytest.fixture
    def debater_config(self):
        return DebaterConfig(
            name="TestDebater",
            stance="pro",
            personality="Analytical",
        )

    def test_creates_agent_with_config(self, debater_config, mock_model_client):
        agent = DebaterAgent(debater_config, mock_model_client)

        assert agent.config == debater_config
        assert agent._agent is not None

    def test_reset_clears_conversation_history(self, debater_config, mock_model_client):
        agent = DebaterAgent(debater_config, mock_model_client)
        agent._conversation_history.append({"role": "user", "content": "test"})

        agent.reset()

        assert len(agent._conversation_history) == 0


class TestModeratorAgent:
    def test_introduce_debate_includes_topic_and_names(self):
        from src.agents.moderator import ModeratorAgent

        moderator = ModeratorAgent()
        intro = moderator.introduce_debate(
            topic="Test Topic",
            pro_name="Alice",
            con_name="Bob",
        )

        assert "Test Topic" in intro
        assert "Alice" in intro
        assert "Bob" in intro

    def test_transition_mentions_speakers(self):
        from src.agents.moderator import ModeratorAgent

        moderator = ModeratorAgent()
        transition = moderator.transition_to_opponent("Alice", "Bob", 1)

        assert "Alice" in transition
        assert "Bob" in transition

    def test_conclude_thanks_both_debaters(self):
        from src.agents.moderator import ModeratorAgent

        moderator = ModeratorAgent()
        conclusion = moderator.conclude_debate("Alice", "Bob")

        assert "Alice" in conclusion
        assert "Bob" in conclusion
        assert "Thank you" in conclusion
