# Agents Arguing

AI agents debating each other with voice and video output.

## Features

- **AI-Powered Debates**: Two AI agents argue opposing sides of any topic
- **Text-to-Speech**: Generate realistic voice output using XTTS v2
- **Lip-Sync Video**: Create talking avatar videos with EchoMimic V3
- **Streaming Support**: Real-time debate streaming via WebSocket
- **Web Interface**: Gradio-based UI for easy interaction
- **Customizable Avatars**: Use custom images and voice samples for debaters

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for video generation)

## Installation

```bash
git clone https://github.com/yourusername/agents-arguing.git
cd agents-arguing
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For web interface:
```bash
pip install -e ".[web]"
```

For video generation with EchoMimic:
```bash
pip install -e ".[echomimic]"
```

## Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Required environment variables:
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` - For LLM access
- `STT_MODEL` - Speech-to-text model (default: faster-whisper)
- `TTS_MODEL` - Text-to-speech model (default: xtts_v2)
- `VIDEO_MODEL` - Video generation model (default: echomimic_v3)

## Usage

### CLI - Full Video Debate

```bash
debate run "Should AI be regulated?" --rounds 3 --pro-name Alex --con-name Jordan
```

### CLI - Text-Only Debate

```bash
debate text-only "Is remote work better than office work?" --rounds 2
```

### Web Server

```bash
debate-server
```

### Gradio UI

```bash
debate-ui
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--pro-name, -p` | Name of the pro debater (default: Alex) |
| `--con-name, -c` | Name of the con debater (default: Jordan) |
| `--pro-avatar` | Avatar image for pro debater |
| `--con-avatar` | Avatar image for con debater |
| `--pro-voice` | Voice sample for pro debater |
| `--con-voice` | Voice sample for con debater |
| `--rounds, -r` | Number of debate rounds (default: 3) |
| `--output, -o` | Output directory |

## Project Structure

```
agents_arguing/
├── src/
│   ├── agents/          # Debate agents (debater, moderator, manager)
│   ├── pipeline/        # Audio and orchestration pipelines
│   ├── services/        # STT, TTS, and base services
│   ├── video/           # Video generation
│   ├── web/             # Web server and WebSocket handlers
│   └── cli.py           # Command-line interface
├── tests/
│   ├── unit/            # Unit tests
│   └── e2e/             # End-to-end tests
├── assets/
│   ├── avatars/         # Default avatar images
│   └── output/          # Generated output files
└── configs/             # Configuration files
```

## Development

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

## License

MIT
