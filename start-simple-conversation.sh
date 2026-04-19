#!/bin/bash
# Start Simple Conversation Server with Female Voice

echo "========================================"
echo "Simple Conversation Server"
echo "========================================"
echo "Mode: Non-streaming LLM + Female Voice TTS"
echo "Port: 8772"
echo "========================================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if vibevoice is installed
uv run python -c "import vibevoice" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: VibeVoice not installed"
    echo "Run: pip install vibevoice[streamingtts]"
    exit 1
fi

# Run the server
uv run python conversation_simple_server.py
