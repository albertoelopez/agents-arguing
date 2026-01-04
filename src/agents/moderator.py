from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModeratorConfig:
    name: str = "Moderator"
    avatar_path: Path | None = None
    voice_sample_path: Path | None = None


class ModeratorAgent:
    def __init__(self, config: ModeratorConfig | None = None):
        self.config = config or ModeratorConfig()

    def introduce_debate(self, topic: str, pro_name: str, con_name: str) -> str:
        return f"""Welcome to today's debate. Our topic is: "{topic}"

Arguing in favor, we have {pro_name}.
Arguing against, we have {con_name}.

Each debater will have multiple rounds to present their arguments.
Let's begin with opening statements. {pro_name}, you have the floor."""

    def transition_to_opponent(
        self,
        current_speaker: str,
        next_speaker: str,
        round_number: int,
    ) -> str:
        if round_number == 1:
            return f"Thank you, {current_speaker}. Now, {next_speaker}, your opening statement please."
        return f"Thank you, {current_speaker}. {next_speaker}, your response."

    def announce_round(self, round_number: int, total_rounds: int) -> str:
        return f"Round {round_number} of {total_rounds}."

    def introduce_closing(self) -> str:
        return "We now move to closing statements."

    def conclude_debate(self, pro_name: str, con_name: str) -> str:
        return f"""Thank you to both {pro_name} and {con_name} for an engaging debate.

We've heard compelling arguments from both sides. The audience can now reflect on the points presented.

This concludes our debate. Thank you for watching."""
