#!/bin/bash

# 1. 检查 OBS 是否正在运行，如果在运行则提示并退出
if pgrep -x "obs" > /dev/null; then
    echo "⚠️  错误: OBS 已经在运行中。请先完全关闭 (Command+Q) 现有的 OBS 进程。"
    exit 1
fi

echo "🚀 正在启动 AI 专用版 OBS (已解除音视频权限限制)..."

# 2. 执行带参数的启动命令
/Applications/OBS.app/Contents/MacOS/OBS \
  --use-fake-ui-for-media-stream \
  --autoplay-policy=no-user-gesture-required \
  --allow-file-access-from-files \
  --disable-gesture-requirement-for-media-playback \
  --disable-features=PreloadMediaEngagementData,MediaEngagementBypassAutoplayPolicies &

echo "✅ 启动指令已发送。"