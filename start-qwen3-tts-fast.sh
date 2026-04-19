#!/bin/bash
# Qwen3-TTS 快速响应服务启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Qwen3-TTS 快速响应服务"
echo "========================================"

# 检查 uv
if ! command -v uv &> /dev/null; then
    echo "错误: uv 未安装"
    echo "请安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 检查 LLM API
LLM_API="${LLM_API_BASE:-http://127.0.0.1:12345/v1}"
echo "检查 LLM API: $LLM_API"

if ! curl -s -o /dev/null -w "%{http_code}" "$LLM_API" | grep -q "200\|404"; then
    echo "警告: LLM API 可能未运行"
    echo "请确保 LLM 服务已启动"
fi

# 使用 uv 运行
echo ""
echo "启动服务器..."
echo "WebSocket: ws://localhost:8770"
echo "打开 qwen3-tts-fast.html"
echo ""

uv run --with websockets --with requests --with numpy python qwen3_tts_fast_server.py
