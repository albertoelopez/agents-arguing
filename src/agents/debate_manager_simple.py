from dataclasses import dataclass, field
from typing import AsyncIterator

from src.agents.debater_simple import SimpleDebaterAgent, DebaterConfig
from src.agents.moderator import ModeratorAgent, ModeratorConfig
from src.agents.llm_client import get_llm_client, LLMClient


@dataclass
class DebateTurn:
    speaker: str
    stance: str
    content: str
    round_number: int
    turn_type: str


@dataclass
class DebateResult:
    topic: str
    turns: list[DebateTurn] = field(default_factory=list)
    pro_name: str = ""
    con_name: str = ""

    def get_transcript(self) -> str:
        lines = [f"Debate Topic: {self.topic}\n", "=" * 50, ""]

        for turn in self.turns:
            lines.append(f"[{turn.turn_type.upper()}] {turn.speaker} ({turn.stance}):")
            lines.append(turn.content)
            lines.append("")

        return "\n".join(lines)


class SimpleDebateManager:
    def __init__(
        self,
        topic: str,
        pro_config: DebaterConfig,
        con_config: DebaterConfig,
        moderator_config: ModeratorConfig | None = None,
        num_rounds: int = 3,
        llm_provider: str = "ollama",
        llm_model: str | None = None,
    ):
        self.topic = topic
        self.num_rounds = num_rounds

        self._llm_client = get_llm_client(llm_provider, llm_model)

        self.pro_agent = SimpleDebaterAgent(pro_config, self._llm_client)
        self.con_agent = SimpleDebaterAgent(con_config, self._llm_client)
        self.moderator = ModeratorAgent(moderator_config)

        self.result = DebateResult(
            topic=topic,
            pro_name=pro_config.name,
            con_name=con_config.name,
        )

    async def run_debate(self) -> DebateResult:
        async for _ in self.run_debate_stream():
            pass
        return self.result

    async def run_debate_stream(self) -> AsyncIterator[DebateTurn]:
        intro = self.moderator.introduce_debate(
            self.topic,
            self.pro_agent.config.name,
            self.con_agent.config.name,
        )

        intro_turn = DebateTurn(
            speaker=self.moderator.config.name,
            stance="neutral",
            content=intro,
            round_number=0,
            turn_type="introduction",
        )
        self.result.turns.append(intro_turn)
        yield intro_turn

        for round_num in range(1, self.num_rounds + 1):
            round_announce = self.moderator.announce_round(round_num, self.num_rounds)
            announce_turn = DebateTurn(
                speaker=self.moderator.config.name,
                stance="neutral",
                content=round_announce,
                round_number=round_num,
                turn_type="announcement",
            )
            self.result.turns.append(announce_turn)
            yield announce_turn

            last_con_argument = None
            if round_num > 1:
                con_turns = [t for t in self.result.turns if t.stance == "con" and t.turn_type == "argument"]
                if con_turns:
                    last_con_argument = con_turns[-1].content

            pro_argument = await self.pro_agent.generate_argument(
                self.topic,
                opponent_argument=last_con_argument,
                round_number=round_num,
            )

            pro_turn = DebateTurn(
                speaker=self.pro_agent.config.name,
                stance="pro",
                content=pro_argument,
                round_number=round_num,
                turn_type="argument",
            )
            self.result.turns.append(pro_turn)
            yield pro_turn

            transition = self.moderator.transition_to_opponent(
                self.pro_agent.config.name,
                self.con_agent.config.name,
                round_num,
            )
            trans_turn = DebateTurn(
                speaker=self.moderator.config.name,
                stance="neutral",
                content=transition,
                round_number=round_num,
                turn_type="transition",
            )
            self.result.turns.append(trans_turn)
            yield trans_turn

            con_argument = await self.con_agent.generate_argument(
                self.topic,
                opponent_argument=pro_argument,
                round_number=round_num,
            )

            con_turn = DebateTurn(
                speaker=self.con_agent.config.name,
                stance="con",
                content=con_argument,
                round_number=round_num,
                turn_type="argument",
            )
            self.result.turns.append(con_turn)
            yield con_turn

        closing_intro = self.moderator.introduce_closing()
        closing_intro_turn = DebateTurn(
            speaker=self.moderator.config.name,
            stance="neutral",
            content=closing_intro,
            round_number=self.num_rounds + 1,
            turn_type="transition",
        )
        self.result.turns.append(closing_intro_turn)
        yield closing_intro_turn

        pro_closing = await self.pro_agent.generate_closing(self.topic)
        pro_close_turn = DebateTurn(
            speaker=self.pro_agent.config.name,
            stance="pro",
            content=pro_closing,
            round_number=self.num_rounds + 1,
            turn_type="closing",
        )
        self.result.turns.append(pro_close_turn)
        yield pro_close_turn

        con_closing = await self.con_agent.generate_closing(self.topic)
        con_close_turn = DebateTurn(
            speaker=self.con_agent.config.name,
            stance="con",
            content=con_closing,
            round_number=self.num_rounds + 1,
            turn_type="closing",
        )
        self.result.turns.append(con_close_turn)
        yield con_close_turn

        conclusion = self.moderator.conclude_debate(
            self.pro_agent.config.name,
            self.con_agent.config.name,
        )
        conclusion_turn = DebateTurn(
            speaker=self.moderator.config.name,
            stance="neutral",
            content=conclusion,
            round_number=self.num_rounds + 1,
            turn_type="conclusion",
        )
        self.result.turns.append(conclusion_turn)
        yield conclusion_turn
