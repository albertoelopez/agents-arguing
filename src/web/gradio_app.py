import asyncio
from pathlib import Path
from typing import Generator

import gradio as gr

from src.agents.debate_manager_simple import SimpleDebateManager, DebateTurn
from src.agents.debater_simple import DebaterConfig
from src.realtime.streaming import StreamingDebateSession
from src.config import settings


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
) -> Generator[str, None, None]:

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
    yield output_html

    turn_count = 0
    total_turns = 2 + (num_rounds * 2) + 2 + 1

    async for turn in manager.run_debate_stream():
        output_html += format_turn(turn)
        turn_count += 1
        progress(turn_count / total_turns, desc=f"{turn.speaker} speaking...")
        yield output_html

    output_html += """
<div style="text-align: center; padding: 20px; margin-top: 20px; background: #1a1a2e; border-radius: 12px;">
    <h3 style="color: #4CAF50;">✅ Debate Complete</h3>
</div>
"""
    yield output_html


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
        async for html in run_debate_async(
            topic, pro_name, pro_personality,
            con_name, con_personality, num_rounds, llm_provider, llm_model, progress
        ):
            result = html
            yield result

    gen = collect()

    while True:
        try:
            result = loop.run_until_complete(gen.__anext__())
            yield result
        except StopAsyncIteration:
            break


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
            outputs=debate_output,
        )

        gr.Markdown("""
---
### How it works
1. **Set your topic** - Any debatable subject
2. **Select your LLM** - Choose from your local Ollama models
3. **Customize debaters** - Give them names and personalities
4. **Watch the debate** - AI agents argue back and forth in real-time

Built with Ollama + Pipecat 🚀
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
