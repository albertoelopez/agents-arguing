import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Generator

import gradio as gr

from src.agents.debate_manager_simple import SimpleDebateManager, DebateTurn
from src.agents.debater_simple import DebaterConfig
from src.realtime.streaming import StreamingDebateSession
from src.config import settings

DEBATE_TURNS_STORE: dict[str, list[DebateTurn]] = {}


async def synthesize_audio_edge_tts(text: str, voice: str, output_path: Path) -> Path:
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return output_path
    except ImportError:
        raise ImportError("Install edge-tts: pip install edge-tts")


async def generate_debate_audio(
    turns: list[DebateTurn],
    pro_voice: str = "en-US-GuyNeural",
    con_voice: str = "en-US-JennyNeural",
    moderator_voice: str = "en-US-AriaNeural",
) -> list[Path]:
    audio_files = []
    output_dir = settings.output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, turn in enumerate(turns):
        if turn.stance == "pro":
            voice = pro_voice
        elif turn.stance == "con":
            voice = con_voice
        else:
            voice = moderator_voice

        audio_path = output_dir / f"turn_{i:03d}_{turn.speaker}.mp3"
        await synthesize_audio_edge_tts(turn.content, voice, audio_path)
        audio_files.append(audio_path)

    return audio_files


def combine_audio_files(audio_files: list[Path], output_path: Path) -> Path:
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=500)

    for audio_file in audio_files:
        segment = AudioSegment.from_mp3(str(audio_file))
        combined += segment + pause

    combined.export(str(output_path), format="mp3")
    return output_path


async def generate_video_for_debate(
    turns: list[DebateTurn],
    pro_avatar: Path | None,
    con_avatar: Path | None,
    audio_files: list[Path],
) -> Path:
    from src.video.composer import VideoComposer

    output_dir = settings.output_dir / "video"
    output_dir.mkdir(parents=True, exist_ok=True)

    composer = VideoComposer()

    segments = []
    for i, (turn, audio_path) in enumerate(zip(turns, audio_files)):
        if turn.stance == "pro" and pro_avatar:
            avatar = pro_avatar
        elif turn.stance == "con" and con_avatar:
            avatar = con_avatar
        else:
            avatar = None

        segments.append({
            "turn": turn,
            "audio": audio_path,
            "avatar": avatar,
        })

    output_path = output_dir / f"debate_{uuid.uuid4().hex[:8]}.mp4"
    await composer.compose_debate_video(segments, output_path)

    return output_path


CSS = """
.debate-container { max-width: 900px; margin: 0 auto; }
.pro-message { background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); border-left: 4px solid #4CAF50; }
.con-message { background: linear-gradient(135deg, #4a1a1a 0%, #5a2d2d 100%); border-left: 4px solid #f44336; }
.moderator-message { background: linear-gradient(135deg, #1a1a4a 0%, #2d2d5a 100%); border-left: 4px solid #2196F3; }
.speaker-name { font-weight: bold; font-size: 1.1em; margin-bottom: 8px; }
.turn-content { line-height: 1.6; }
"""


def format_turn(turn: DebateTurn) -> str:
    if turn.stance == "pro":
        color = "#4CAF50"
        icon = "🟢"
    elif turn.stance == "con":
        color = "#f44336"
        icon = "🔴"
    else:
        color = "#2196F3"
        icon = "🔵"

    return f"""
<div style="margin: 10px 0; padding: 15px; border-radius: 8px; border-left: 4px solid {color}; background: rgba(255,255,255,0.05);">
    <div style="color: {color}; font-weight: bold; margin-bottom: 8px;">
        {icon} {turn.speaker} <span style="font-size: 0.8em; opacity: 0.7;">({turn.turn_type})</span>
    </div>
    <div style="line-height: 1.6;">{turn.content}</div>
</div>
"""


async def run_debate_async(
    topic: str,
    pro_name: str,
    pro_personality: str,
    con_name: str,
    con_personality: str,
    num_rounds: int,
    llm_provider: str = "groq",
    llm_model: str = "llama-3.3-70b-versatile",
    progress=gr.Progress(),
) -> Generator[tuple[str, str], None, None]:
    debate_id = uuid.uuid4().hex[:8]

    pro_config = DebaterConfig(
        name=pro_name,
        stance="pro",
        personality=pro_personality,
    )
    con_config = DebaterConfig(
        name=con_name,
        stance="con",
        personality=con_personality,
    )

    manager = SimpleDebateManager(
        topic=topic,
        pro_config=pro_config,
        con_config=con_config,
        num_rounds=num_rounds,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    output_html = f"""
<div style="text-align: center; padding: 20px; margin-bottom: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px;">
    <h2 style="color: #fff; margin: 0;">📣 {topic}</h2>
    <p style="color: #aaa; margin-top: 10px;">{pro_name} (PRO) vs {con_name} (CON) • {num_rounds} rounds</p>
</div>
"""
    yield output_html, debate_id

    turn_count = 0
    total_turns = 2 + (num_rounds * 2) + 2 + 1
    collected_turns = []

    async for turn in manager.run_debate_stream():
        output_html += format_turn(turn)
        turn_count += 1
        collected_turns.append(turn)
        progress(turn_count / total_turns, desc=f"{turn.speaker} speaking...")
        yield output_html, debate_id

    DEBATE_TURNS_STORE[debate_id] = collected_turns

    output_html += f"""
<div style="text-align: center; padding: 20px; margin-top: 20px; background: #1a1a2e; border-radius: 12px;">
    <h3 style="color: #4CAF50;">✅ Debate Complete</h3>
    <p style="color: #aaa;">Debate ID: {debate_id} • Click "Generate Audio" to create audio version</p>
</div>
"""
    yield output_html, debate_id


def run_debate_sync(
    topic: str,
    pro_name: str,
    pro_personality: str,
    con_name: str,
    con_personality: str,
    num_rounds: int,
    llm_provider: str = "groq",
    llm_model: str = "llama-3.3-70b-versatile",
    progress=gr.Progress(),
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def collect():
        result = ""
        debate_id = ""
        async for html, d_id in run_debate_async(
            topic, pro_name, pro_personality,
            con_name, con_personality, num_rounds, llm_provider, llm_model, progress
        ):
            result = html
            debate_id = d_id
            yield result, debate_id

    gen = collect()

    while True:
        try:
            result, debate_id = loop.run_until_complete(gen.__anext__())
            yield result, debate_id
        except StopAsyncIteration:
            break


def generate_audio_sync(debate_id: str, pro_voice: str, con_voice: str, progress=gr.Progress()):
    if not debate_id or debate_id not in DEBATE_TURNS_STORE:
        return None, "No debate found. Run a debate first."

    turns = DEBATE_TURNS_STORE[debate_id]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        progress(0.1, desc="Generating audio for debate turns...")
        audio_files = loop.run_until_complete(
            generate_debate_audio(turns, pro_voice, con_voice)
        )

        progress(0.9, desc="Combining audio files...")
        output_dir = settings.output_dir / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_path = output_dir / f"debate_{debate_id}_full.mp3"
        combine_audio_files(audio_files, combined_path)

        progress(1.0, desc="Done!")
        return str(combined_path), f"Audio generated: {len(turns)} segments combined"

    except Exception as e:
        return None, f"Error generating audio: {str(e)}"


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="Agents Arguing",
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="red",
            neutral_hue="slate",
        ),
    ) as app:
        gr.Markdown("""
# 🎭 Agents Arguing
### Watch AI agents debate any topic in real-time
        """)

        with gr.Row():
            with gr.Column(scale=1):
                topic_input = gr.Textbox(
                    label="Debate Topic",
                    placeholder="e.g., AI will benefit humanity more than harm it",
                    value="Artificial Intelligence will create more jobs than it destroys",
                    lines=2,
                )

                with gr.Row():
                    num_rounds = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        label="Number of Rounds",
                    )

                llm_provider = gr.Dropdown(
                    label="LLM Provider",
                    choices=["groq", "ollama"],
                    value="groq",
                )

                llm_model = gr.Dropdown(
                    label="LLM Model",
                    choices=[
                        "llama-3.3-70b-versatile",
                        "llama-3.1-8b-instant",
                        "mixtral-8x7b-32768",
                        "gemma2-9b-it",
                    ],
                    value="llama-3.3-70b-versatile",
                )

                def update_models(provider):
                    if provider == "groq":
                        return gr.Dropdown(
                            choices=[
                                "llama-3.3-70b-versatile",
                                "llama-3.1-8b-instant",
                                "mixtral-8x7b-32768",
                                "gemma2-9b-it",
                            ],
                            value="llama-3.3-70b-versatile",
                        )
                    else:
                        return gr.Dropdown(
                            choices=[
                                "llama3.1:8b",
                                "llama3.2:latest",
                                "mistral:latest",
                                "qwen3:latest",
                            ],
                            value="llama3.1:8b",
                        )

                llm_provider.change(fn=update_models, inputs=llm_provider, outputs=llm_model)

                gr.Markdown("### 🟢 Pro Debater")
                with gr.Group():
                    pro_name = gr.Textbox(label="Name", value="Alex")
                    pro_personality = gr.Textbox(
                        label="Personality",
                        value="Optimistic, data-driven, focuses on technological progress and innovation",
                        lines=2,
                    )

                gr.Markdown("### 🔴 Con Debater")
                with gr.Group():
                    con_name = gr.Textbox(label="Name", value="Jordan")
                    con_personality = gr.Textbox(
                        label="Personality",
                        value="Skeptical, philosophical, emphasizes ethical concerns and potential risks",
                        lines=2,
                    )

                start_btn = gr.Button("🎬 Start Debate", variant="primary", size="lg")

            with gr.Column(scale=2):
                debate_output = gr.HTML(
                    value="""
<div style="text-align: center; padding: 60px; color: #666;">
    <h3>👆 Configure your debate and click "Start Debate"</h3>
    <p>The debate will stream in real-time here</p>
</div>
                    """,
                    label="Debate",
                )

                debate_id_state = gr.State(value="")

                gr.Markdown("### 🎙️ Audio Generation")
                with gr.Row():
                    pro_voice = gr.Dropdown(
                        label="Pro Voice",
                        choices=[
                            "en-US-GuyNeural",
                            "en-US-ChristopherNeural",
                            "en-US-EricNeural",
                            "en-GB-RyanNeural",
                        ],
                        value="en-US-GuyNeural",
                    )
                    con_voice = gr.Dropdown(
                        label="Con Voice",
                        choices=[
                            "en-US-JennyNeural",
                            "en-US-AriaNeural",
                            "en-US-SaraNeural",
                            "en-GB-SoniaNeural",
                        ],
                        value="en-US-JennyNeural",
                    )

                with gr.Row():
                    generate_audio_btn = gr.Button("🔊 Generate Audio", variant="secondary")

                audio_status = gr.Textbox(label="Status", interactive=False)
                audio_output = gr.Audio(label="Debate Audio", type="filepath")

        start_btn.click(
            fn=run_debate_sync,
            inputs=[
                topic_input,
                pro_name,
                pro_personality,
                con_name,
                con_personality,
                num_rounds,
                llm_provider,
                llm_model,
            ],
            outputs=[debate_output, debate_id_state],
        )

        generate_audio_btn.click(
            fn=generate_audio_sync,
            inputs=[debate_id_state, pro_voice, con_voice],
            outputs=[audio_output, audio_status],
        )

        gr.Markdown("""
---
### How it works
1. **Set your topic** - Any debatable subject
2. **Select your LLM** - Choose Groq (cloud) or Ollama (local)
3. **Customize debaters** - Give them names and personalities
4. **Watch the debate** - AI agents argue back and forth in real-time
5. **Generate Audio** - Convert the debate to speech with different voices

Built with Groq/Ollama + Edge-TTS
        """)

    return app


def main():
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
