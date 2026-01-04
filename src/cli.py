import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from src.pipeline.orchestrator import DebateOrchestrator, OrchestratorConfig
from src.config import settings

app = typer.Typer(
    name="debate",
    help="Generate AI agent debates with voice and video output",
)
console = Console()


@app.command()
def run(
    topic: str = typer.Argument(..., help="The debate topic"),
    pro_name: str = typer.Option("Alex", "--pro-name", "-p", help="Name of the pro debater"),
    con_name: str = typer.Option("Jordan", "--con-name", "-c", help="Name of the con debater"),
    pro_avatar: Optional[Path] = typer.Option(None, "--pro-avatar", help="Avatar image for pro debater"),
    con_avatar: Optional[Path] = typer.Option(None, "--con-avatar", help="Avatar image for con debater"),
    pro_voice: Optional[Path] = typer.Option(None, "--pro-voice", help="Voice sample for pro debater"),
    con_voice: Optional[Path] = typer.Option(None, "--con-voice", help="Voice sample for con debater"),
    rounds: int = typer.Option(3, "--rounds", "-r", help="Number of debate rounds"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    console.print(Panel.fit(
        f"[bold blue]AI Debate Generator[/bold blue]\n"
        f"Topic: {topic}\n"
        f"Debaters: {pro_name} (PRO) vs {con_name} (CON)\n"
        f"Rounds: {rounds}",
        title="Configuration",
    ))

    if pro_avatar is None:
        pro_avatar = settings.avatars_dir / "default_pro.png"
    if con_avatar is None:
        con_avatar = settings.avatars_dir / "default_con.png"

    config = OrchestratorConfig(
        topic=topic,
        pro_name=pro_name,
        pro_personality="Optimistic, data-driven, focuses on progress and innovation",
        pro_avatar=pro_avatar,
        pro_voice=pro_voice,
        con_name=con_name,
        con_personality="Skeptical, philosophical, emphasizes risks and ethical concerns",
        con_avatar=con_avatar,
        con_voice=con_voice,
        num_rounds=rounds,
        output_dir=output_dir,
    )

    asyncio.run(_run_debate(config))


async def _run_debate(config: OrchestratorConfig) -> None:
    orchestrator = DebateOrchestrator(config)

    try:
        await orchestrator.initialize()
        final_video = await orchestrator.run()

        console.print(Panel.fit(
            f"[bold green]Debate video generated successfully![/bold green]\n"
            f"Output: {final_video}",
            title="Complete",
        ))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise
    finally:
        await orchestrator.shutdown()


@app.command()
def text_only(
    topic: str = typer.Argument(..., help="The debate topic"),
    pro_name: str = typer.Option("Alex", "--pro-name", "-p"),
    con_name: str = typer.Option("Jordan", "--con-name", "-c"),
    rounds: int = typer.Option(3, "--rounds", "-r"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save transcript to file"),
) -> None:
    from src.agents.debate_manager import DebateManager
    from src.agents.debater import DebaterConfig

    console.print(f"[bold]Starting text-only debate on:[/bold] {topic}\n")

    pro_config = DebaterConfig(
        name=pro_name,
        stance="pro",
        personality="Optimistic, data-driven",
    )
    con_config = DebaterConfig(
        name=con_name,
        stance="con",
        personality="Skeptical, philosophical",
    )

    manager = DebateManager(
        topic=topic,
        pro_config=pro_config,
        con_config=con_config,
        num_rounds=rounds,
    )

    async def run() -> None:
        async for turn in manager.run_debate_stream():
            if turn.stance == "pro":
                color = "green"
            elif turn.stance == "con":
                color = "red"
            else:
                color = "blue"

            console.print(f"\n[bold {color}]{turn.speaker}[/bold {color}] ({turn.turn_type}):")
            console.print(turn.content)

        if output:
            transcript = manager.result.get_transcript()
            output.write_text(transcript)
            console.print(f"\n[dim]Transcript saved to {output}[/dim]")

    asyncio.run(run())


if __name__ == "__main__":
    app()
