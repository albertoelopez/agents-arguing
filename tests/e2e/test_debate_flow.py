import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.debate_manager import DebateManager, DebateTurn
from src.agents.debater import DebaterConfig


class TestDebateFlow:
    @pytest.fixture
    def pro_config(self):
        return DebaterConfig(
            name="Alex",
            stance="pro",
            personality="Optimistic",
        )

    @pytest.fixture
    def con_config(self):
        return DebaterConfig(
            name="Jordan",
            stance="con",
            personality="Skeptical",
        )

    @pytest.mark.asyncio
    async def test_debate_produces_expected_turn_structure(
        self,
        pro_config,
        con_config,
    ):
        with patch("src.agents.debate_manager.AnthropicClient") as mock_client:
            mock_response = MagicMock()
            mock_response.chat_message.content = "Test argument"
            mock_client.return_value = MagicMock()

            with patch.object(DebateManager, "_create_model_client", return_value=mock_client.return_value):
                manager = DebateManager(
                    topic="Test topic",
                    pro_config=pro_config,
                    con_config=con_config,
                    num_rounds=1,
                )

                turns = []
                with patch.object(manager.pro_agent, "generate_argument", new_callable=AsyncMock) as mock_pro:
                    with patch.object(manager.con_agent, "generate_argument", new_callable=AsyncMock) as mock_con:
                        with patch.object(manager.pro_agent, "generate_closing", new_callable=AsyncMock) as mock_pro_close:
                            with patch.object(manager.con_agent, "generate_closing", new_callable=AsyncMock) as mock_con_close:
                                mock_pro.return_value = "Pro argument"
                                mock_con.return_value = "Con argument"
                                mock_pro_close.return_value = "Pro closing"
                                mock_con_close.return_value = "Con closing"

                                async for turn in manager.run_debate_stream():
                                    turns.append(turn)

                assert len(turns) > 0

                turn_types = [t.turn_type for t in turns]
                assert "introduction" in turn_types
                assert "argument" in turn_types
                assert "closing" in turn_types
                assert "conclusion" in turn_types

    @pytest.mark.asyncio
    async def test_debate_result_contains_both_stances(
        self,
        pro_config,
        con_config,
    ):
        with patch("src.agents.debate_manager.AnthropicClient"):
            with patch.object(DebateManager, "_create_model_client", return_value=MagicMock()):
                manager = DebateManager(
                    topic="Test topic",
                    pro_config=pro_config,
                    con_config=con_config,
                    num_rounds=1,
                )

                with patch.object(manager.pro_agent, "generate_argument", new_callable=AsyncMock, return_value="Pro"):
                    with patch.object(manager.con_agent, "generate_argument", new_callable=AsyncMock, return_value="Con"):
                        with patch.object(manager.pro_agent, "generate_closing", new_callable=AsyncMock, return_value="Pro close"):
                            with patch.object(manager.con_agent, "generate_closing", new_callable=AsyncMock, return_value="Con close"):
                                result = await manager.run_debate()

                stances = {t.stance for t in result.turns}
                assert "pro" in stances
                assert "con" in stances
                assert "neutral" in stances

    def test_debate_result_generates_transcript(self, pro_config, con_config):
        with patch("src.agents.debate_manager.AnthropicClient"):
            with patch.object(DebateManager, "_create_model_client", return_value=MagicMock()):
                manager = DebateManager(
                    topic="Test topic",
                    pro_config=pro_config,
                    con_config=con_config,
                    num_rounds=1,
                )

                manager.result.turns.append(
                    DebateTurn(
                        speaker="Alex",
                        stance="pro",
                        content="Test content",
                        round_number=1,
                        turn_type="argument",
                    )
                )

                transcript = manager.result.get_transcript()

                assert "Test topic" in transcript
                assert "Alex" in transcript
                assert "Test content" in transcript
