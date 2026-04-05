#!/usr/bin/env python3
"""
WebSocket 语音识别 + Qwen3-TTS 服务器
"""

import asyncio
import websockets
import json
import logging
import concurrent.futures
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
clients = []
result_queue = asyncio.Queue()
tts_queue = asyncio.Queue()

# 导入 Qwen3-TTS wrapper
try:
    from qwen3_tts_wrapper import get_tts
    TTS_AVAILABLE = True
    logger.info("Qwen3-TTS available")
except ImportError as e:
    TTS_AVAILABLE = False
    logger.warning(f"Qwen3-TTS not available: {e}")

# 语音识别支持（已禁用）
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = False  # 强制禁用
    logger.info("Speech recognition module found but DISABLED")
except ImportError:
    HAS_SPEECH_RECOGNITION = False
    logger.warning("Speech recognition NOT available")


async def tts_worker():
    """TTS 工作协程"""
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    while True:
        try:
            # 获取 TTS 请求
            request = await tts_queue.get()
            text = request.get("text", "")

            if not text:
                continue

            logger.info(f"Generating TTS: '{text[:50]}...'")

            # 在线程池中生成音频
            def generate():
                tts = get_tts()
                if tts.model:
                    return tts.synthesize(
                        text=text,
                        voice=request.get("voice", "Vivian"),
                        speed=request.get("speed", 1.0),
                        emotion=request.get("emotion", "Normal tone")
                    )
                return None

            audio_data = await loop.run_in_executor(executor, generate)

            if audio_data:
                # 编码为 base64
                import base64
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                # 广播到所有客户端
                msg = json.dumps({
                    "type": "audio",
                    "audio": audio_b64,
                    "format": "wav",
                    "sample_rate": 24000
                })
                await broadcast(msg)
                logger.info("✓ TTS audio sent")
            else:
                # TTS 失败
                msg = json.dumps({
                    "type": "tts_error",
                    "message": "TTS generation failed"
                })
                await broadcast(msg)

        except Exception as e:
            logger.error(f"TTS worker error: {e}")


def sync_speech_recognition(queue):
    """在单独线程中运行同步语音识别"""
    recognizer = sr.Recognizer()

    # 列出可用麦克风
    mics = sr.Microphone.list_microphone_names()
    logger.info(f"[Speech] Available microphones: {len(mics)}")
    for i, name in enumerate(mics[:5]):
        logger.info(f"[Speech]   Mic {i}: {name}")

    # 尝试使用默认麦克风
    try:
        mic = sr.Microphone()
    except Exception as e:
        logger.error(f"[Speech] Failed to create microphone: {e}")
        return

    # 校准
    logger.info("[Speech] Calibrating microphone...")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        logger.info("[Speech] Microphone calibrated successfully")
    except Exception as e:
        logger.error(f"[Speech] Calibration failed: {e}")
        return

    logger.info("[Speech] Ready to listen!")

    consecutive_errors = 0
    max_errors = 5

    while True:
        try:
            with mic as source:
                logger.info("[Speech] Listening...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)

                try:
                    # DISABLED: Google Speech API
                    # text = recognizer.recognize_google(audio, language="en-US")
                    # logger.info(f"[Speech] ✓ Recognized: '{text}'")
                    # queue.put_nowait(text)

                    # 模拟识别（用于测试 TTS）
                    logger.debug("[Speech] Audio captured (recognition disabled)")
                    consecutive_errors = 0

                except sr.UnknownValueError:
                    logger.debug("[Speech] No speech detected")
                    consecutive_errors = 0
                except sr.RequestError as e:
                    logger.error(f"[Speech] API error: {e}")
                    consecutive_errors += 1
                except Exception as e:
                    logger.error(f"[Speech] Recognition error: {e}")
                    consecutive_errors += 1

        except sr.WaitTimeoutError:
            logger.debug("[Speech] Listen timeout")
        except Exception as e:
            logger.error(f"[Speech] Error in listen loop: {e}")
            consecutive_errors += 1

        # 如果连续错误太多，等待一段时间
        if consecutive_errors > max_errors:
            logger.warning(f"[Speech] Too many errors ({consecutive_errors}), waiting...")
            import time
            time.sleep(5)
            consecutive_errors = 0


async def process_results():
    """处理语音识别结果"""
    while True:
        try:
            text = await result_queue.get()

            # 发送识别结果
            msg = json.dumps({
                "type": "transcript",
                "text": text,
                "is_final": True
            })
            await broadcast(msg)

            # 同时发送到 TTS
            if TTS_AVAILABLE:
                await tts_queue.put({"text": text})

        except Exception as e:
            logger.error(f"Process error: {e}")


async def broadcast(message):
    """向所有客户端广播消息"""
    if not clients:
        logger.debug(f"No clients: {message[:50]}")
        return

    logger.info(f"Broadcasting to {len(clients)} client(s)")
    for client in clients:
        try:
            await client.send(message)
        except Exception as e:
            logger.error(f"Send error: {e}")


async def handler(websocket):
    """处理 WebSocket 连接"""
    addr = websocket.remote_address
    logger.info(f"=== CONNECTED: {addr} ===")

    clients.append(websocket)

    try:
        # 发送欢迎消息
        welcome = json.dumps({
            "type": "connected",
            "message": "Connected to voice server",
            "has_speech": HAS_SPEECH_RECOGNITION,
            "has_tts": TTS_AVAILABLE
        })
        await websocket.send(welcome)

        # 保持连接
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
                elif data.get("type") == "tts":
                    # 手动触发 TTS
                    text = data.get("text", "")
                    if text:
                        await tts_queue.put({
                            "text": text,
                            "voice": data.get("voice", "Vivian"),
                            "speed": data.get("speed", 1.0)
                        })
            except Exception as e:
                logger.error(f"Message error: {e}")

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"=== DISCONNECTED: {addr} ===")
    finally:
        if websocket in clients:
            clients.remove(websocket)


async def main():
    logger.info("=" * 50)
    logger.info("Qwen3-TTS Server (Speech Recognition Disabled)")
    logger.info("=" * 50)

    # 启动 WebSocket 服务器
    server = await websockets.serve(handler, "localhost", 8765)
    logger.info("Server listening on ws://localhost:8765")
    logger.info(f"Speech recognition: {'ON' if HAS_SPEECH_RECOGNITION else 'OFF'}")
    logger.info(f"TTS: {'ON' if TTS_AVAILABLE else 'OFF'}")
    logger.info("=" * 50)

    # 启动任务
    tasks = []

    # DISABLED: 语音识别（使用 Google API，已禁用）
    # if HAS_SPEECH_RECOGNITION:
    #     # 语音识别线程
    #     loop = asyncio.get_event_loop()
    #     executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    #     loop.run_in_executor(executor, sync_speech_recognition, result_queue)
    #
    #     # 结果处理
    #     tasks.append(asyncio.create_task(process_results()))

    # TTS 工作协程
    tasks.append(asyncio.create_task(tts_worker()))

    # 等待
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")
