#!/usr/bin/env python3
"""
Parakeet-MLX ASR + Knowledge Base RAG WebSocket 服务器
Mic → ASR → Knowledge Base RAG → Answer → TTS

用法: python parakeet_asr_server.py
浏览器录音 → 发送音频数据 → 服务器 ASR 识别 → RAG 检索知识库 → LLM 生成回答 → 返回浏览器 TTS
"""

import asyncio
import websockets
import json
import base64
import logging
import tempfile
import os
import wave
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("parakeet-asr-server")

# --- Parakeet ASR ---
try:
    from parakeet_mlx import from_pretrained
    PARAKEET_AVAILABLE = True
    log.info("parakeet-mlx available, loading model...")
    model = from_pretrained(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3"))
    PARAKEET_OK = True
    log.info("Parakeet model loaded successfully")
except ImportError:
    PARAKEET_AVAILABLE = False
    PARAKEET_OK = False
    model = None
    log.warning("parakeet-mlx not available. Install with: pip install parakeet-mlx")
except Exception as e:
    PARAKEET_OK = False
    model = None
    log.error(f"Failed to load Parakeet model: {e}")

# --- Knowledge Base RAG ---
try:
    from knowledge_base import KnowledgeBase
    from knowledge_base_demo import seed_meetings
    kb = KnowledgeBase()
    KB_OK = True
    log.info("Knowledge base loaded")
    # Seed demo data if empty
    if kb.stats()["meeting_turns"] == 0:
        seed_meetings(kb)
        log.info("Seeded demo meeting data")
except Exception as e:
    KB_OK = False
    kb = None
    log.warning(f"Knowledge base not available: {e}")

# 过滤 ASR 误识别的填充词/短句，避免触发无意义的 RAG 查询
FILLER_WORDS = {"yeah", "yes", "no", "ok", "okay", "hmm", "um", "uh", "oh", "wow", "hey", "right", "sure", "thanks", "hello", "hi"}
MIN_QUERY_LEN = 4  # 少于这个字符数的也不查询

executor = ThreadPoolExecutor(max_workers=2)


def should_query(text: str) -> bool:
    """判断 ASR 文本是否值得查询知识库"""
    if not text or len(text) < MIN_QUERY_LEN:
        return False
    words = text.lower().strip().rstrip(".!?,;").split()
    if len(words) <= 2 and all(w in FILLER_WORDS for w in words):
        return False
    return True


def do_transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> str | None:
    """在线程中调用 Parakeet ASR"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        result = model.transcribe(path)
        text = result.text if hasattr(result, "text") else str(result)
        return text.strip() if text else None
    except Exception as e:
        log.error(f"Transcription error: {e}")
        return None
    finally:
        os.unlink(path)


def do_rag_query(text: str) -> str | None:
    """在线程中调用知识库 RAG"""
    if not kb:
        return None
    try:
        return kb.query(text)
    except Exception as e:
        log.error(f"RAG query error: {e}")
        return None


# --- WebSocket ---
async def handler(ws):
    log.info("客户端连接")
    await ws.send(json.dumps({
        "type": "ready",
        "asr": PARAKEET_OK,
        "kb": KB_OK,
    }))

    async for msg in ws:
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue

        if data.get("type") == "audio":
            audio_b64 = data.get("data", "")
            sample_rate = data.get("sampleRate", 16000)
            if not audio_b64 or not PARAKEET_OK:
                continue

            log.info(f"收到音频 {len(audio_b64)} chars, 识别中...")
            audio_bytes = base64.b64decode(audio_b64)

            # Step 1: ASR 语音识别
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                executor, do_transcribe, audio_bytes, sample_rate
            )

            if not text:
                await ws.send(json.dumps({"type": "transcript", "text": ""}))
                continue

            log.info(f"ASR: {text}")
            # 发送原始识别结果给前端显示
            await ws.send(json.dumps({"type": "transcript", "text": text}))

            # Step 1.5: 实时写入知识库（积累会议内容供后续检索）
            if KB_OK and should_query(text):
                kb.add_meeting_turn(role="speaker", content=text)

            # Step 2: RAG 知识库查询（过滤填充词）
            if KB_OK and should_query(text):
                log.info(f"RAG 查询中...")
                answer = await loop.run_in_executor(executor, do_rag_query, text)
                if answer:
                    log.info(f"RAG 回答: {answer[:80]}...")
                    try:
                        await ws.send(json.dumps({"type": "answer", "text": answer}))
                    except websockets.exceptions.ConnectionClosed:
                        log.info("客户端已断开，跳过发送")
                else:
                    log.warning("RAG 未返回结果")

        elif data.get("type") == "text":
            # 支持纯文本查询（不需要 ASR）
            text = data.get("text", "").strip()
            if text and KB_OK:
                log.info(f"文本查询: {text}")
                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(executor, do_rag_query, text)
                if answer:
                    await ws.send(json.dumps({"type": "answer", "text": answer}))

        elif data.get("type") == "ping":
            await ws.send(json.dumps({"type": "pong"}))


async def main():
    log.info("=" * 50)
    log.info("Parakeet ASR + Knowledge Base RAG 服务器")
    log.info("  ws://localhost:8766")
    log.info(f"  ASR: {'就绪' if PARAKEET_OK else '不可用'}")
    log.info(f"  KB:  {'就绪' if KB_OK else '不可用'}")
    if KB_OK:
        s = kb.stats()
        log.info(f"  知识库: {s['meeting_turns']} 条会议记录, {s['documents']} 个文档")
    log.info("=" * 50)

    async with websockets.serve(handler, "localhost", 8766):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
