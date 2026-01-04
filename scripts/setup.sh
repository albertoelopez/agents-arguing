#!/bin/bash
set -e

echo "🎭 Setting up Agents Arguing..."

if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "📦 Creating virtual environment and installing dependencies..."
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,web]"

echo "🎬 Installing EchoMimicV3 dependencies..."
uv pip install -e ".[echomimic,pipecat]"

if [ ! -d "external/echomimic_v3" ]; then
    echo "📥 Cloning EchoMimicV3..."
    mkdir -p external
    git clone https://github.com/antgroup/echomimic_v3.git external/echomimic_v3
fi

echo "📁 Creating asset directories..."
mkdir -p assets/{avatars,audio,output}

if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Add avatar images to assets/avatars/"
echo "  3. (Optional) Add voice samples to assets/audio/"
echo ""
echo "Run options:"
echo "  debate 'Topic'          # CLI text-only debate"
echo "  debate-server           # FastAPI server (http://localhost:8000)"
echo "  debate-ui               # Gradio UI (http://localhost:7860)"
echo ""
echo "Web endpoints:"
echo "  http://localhost:8000/live   # Real-time WebSocket client"
echo "  http://localhost:8000/docs   # API documentation"
echo "  http://localhost:7860        # Gradio UI"
