#!/usr/bin/env python3
"""
语音识别 WebSocket 服务器
Qwen3-ASR (本地) → 文字 → 返回浏览器

用法: python voice-asr-server.py
浏览器录音 → 发送音频数据 → 服务器识别 → 返回文字
"""

import asyncio
import websockets
import json
import base64
import logging
import tempfile
import os
import wave
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("asr-server")

# --- ASR ---
from qwen3_asr_wrapper import get_asr
asr = get_asr()
ASR_OK = asr is not None and asr.ensure_model_loaded()

executor = ThreadPoolExecutor(max_workers=1)


def do_transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> str | None:
    """在线程中调用 ASR"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        return asr.transcribe_file(path)
    finally:
        os.unlink(path)


# --- WebSocket ---
async def handler(ws):
    log.info("客户端连接")
    await ws.send(json.dumps({"type": "ready", "asr": ASR_OK}))

    async for msg in ws:
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue

        if data.get("type") == "audio":
            audio_b64 = data.get("data", "")
            sample_rate = data.get("sampleRate", 16000)
            if not audio_b64 or not ASR_OK:
                continue

            log.info(f"收到音频 {len(audio_b64)} chars, 识别中...")
            audio_bytes = base64.b64decode(audio_b64)

            # 在线程池中运行 ASR（避免阻塞事件循环）
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                executor, do_transcribe, audio_bytes, sample_rate
            )

            if text:
                log.info(f"识别结果: {text}")
                await ws.send(json.dumps({"type": "transcript", "text": text}))
            else:
                await ws.send(json.dumps({"type": "transcript", "text": ""}))

        elif data.get("type") == "ping":
            await ws.send(json.dumps({"type": "pong"}))


async def main():
    log.info("=" * 40)
    log.info("Qwen3-ASR 服务器  ws://localhost:8765")
    log.info(f"ASR: {'就绪' if ASR_OK else '不可用'}")
    log.info("=" * 40)

    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
