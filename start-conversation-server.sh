#!/bin/bash
# Conversation Server Startup Script

echo "=========================================="
echo "Conversation Server with VibeVoice TTS"
echo "=========================================="

# Check if LLM API is running
if ! curl -s http://127.0.0.1:12345/v1/models > /dev/null 2>&1; then
    echo "⚠️  Warning: LLM API at http://127.0.0.1:12345/v1 is not responding"
    echo "   Make sure Qwen3.5-2B-MLX-8bit server is running"
fi

# Check if captions.db exists
if [ ! -f "captions.db" ]; then
    echo "⚠️  Warning: captions.db not found"
    echo "   The server will start but won't have conversation context"
fi

# Set default environment variables if not set
export LLM_API_BASE="${LLM_API_BASE:-http://127.0.0.1:12345/v1}"
export LLM_API_KEY="${LLM_API_KEY:-1234}"
export LLM_MODEL="${LLM_MODEL:-Qwen3.5-2B-MLX-8bit}"
export VIBEVOICE_DEVICE="${VIBEVOICE_DEVICE:-mps}"
export VIBEVOICE_INFERENCE_STEPS="${VIBEVOICE_INFERENCE_STEPS:-6}"
export VIBEVOICE_CFG_SCALE="${VIBEVOICE_CFG_SCALE:-1.5}"
export VIBEVOICE_SPEED="${VIBEVOICE_SPEED:-1.5}"
export CONTEXT_LIMIT="${CONTEXT_LIMIT:-20}"
export CONTEXT_TIME_WINDOW="${CONTEXT_TIME_WINDOW:-3600}"

echo "Configuration:"
echo "  LLM API: $LLM_API_BASE"
echo "  LLM Model: $LLM_MODEL"
echo "  VibeVoice Device: $VIBEVOICE_DEVICE"
echo "  VibeVoice: steps=$VIBEVOICE_INFERENCE_STEPS, cfg=$VIBEVOICE_CFG_SCALE, speed=$VIBEVOICE_SPEED"
echo "  Context Limit: $CONTEXT_LIMIT messages"
echo "=========================================="

# Run the server with uv
uv run conversation_server.py
