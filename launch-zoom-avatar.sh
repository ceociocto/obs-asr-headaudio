#!/bin/bash
# 一键启动 Zoom 虚拟头像（外部浏览器模式）

cd "$(dirname "$0")"

echo "🎭 HeadAudio Zoom 虚拟头像启动（外部浏览器模式）"
echo "================================"
echo ""

# 检查服务器是否已运行
if lsof -i :8080 > /dev/null 2>&1; then
    echo "✅ 本地服务器已在运行"
else
    echo "📡 启动本地服务器..."
    python3 -m http.server 8080 > /dev/null 2>&1 &
    SERVER_PID=$!
    sleep 1

    if ! lsof -i :8080 > /dev/null 2>&1; then
        echo "❌ 服务器启动失败"
        exit 1
    fi
    echo "✅ 服务器已启动 (PID: $!)"
fi

echo ""
echo "🌐 打开浏览器..."
# 使用默认浏览器打开自动模式
open "http://localhost:8080/glm-tts.html?mode=auto"

echo ""
echo "📋 后续步骤:"
echo ""
echo "1. ✅ 浏览器已打开，允许麦克风权限"
echo "2. 📺 打开 OBS Studio"
echo "3. ➕ 在 OBS 中添加 '窗口捕获' 来源"
echo "4. 🖼️ 选择浏览器窗口 (glm-tts.html)"
echo "5. 📹 启动虚拟摄像机 (控制 → 启动虚拟摄像机)"
echo "6. 💻 在 Zoom 中选择 'OBS Virtual Camera'"
echo ""
echo "----------------------------------------"
echo "按 Ctrl+C 停止服务器（如果需要）"
echo ""

# 等待用户中断
trap "echo ''; echo '🛑 停止服务器...'; kill $SERVER_PID 2>/dev/null; exit 0" INT

wait
