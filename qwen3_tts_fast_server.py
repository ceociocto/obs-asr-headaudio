#!/usr/bin/env python3
"""
Qwen3-TTS Fast Response Server
LLM streaming + TTS pipeline for minimum latency
"""

import asyncio
import websockets
import json
import logging
import time
import os
import sys
import base64
import tempfile
import sqlite3
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("qwen3-tts-fast")

# --- Config ---
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:12345/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "1234")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.5-2B-MLX-8bit")

QWEN3_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"

MLX_ENV_PATH = Path(__file__).parent / "mlx_audio_env"
QWEN_TTS_PATH = Path(__file__).parent / "qwen3-tts-apple-silicon"
CAPTIONS_DB_PATH = Path(__file__).parent / "captions.db"

if MLX_ENV_PATH.exists() and (MLX_ENV_PATH / "bin" / "python").exists():
    TTS_PYTHON = str(MLX_ENV_PATH / "bin" / "python")
else:
    TTS_PYTHON = sys.executable

_meeting_context_cache = None


def load_meeting_context(force_reload: bool = False) -> str:
    """Load recent meeting context from database"""
    global _meeting_context_cache

    if _meeting_context_cache is not None and not force_reload:
        return _meeting_context_cache

    if not CAPTIONS_DB_PATH.exists():
        return ""

    try:
        conn = sqlite3.connect(str(CAPTIONS_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all entries
        cursor.execute("""
            SELECT c.speaker, c.text
            FROM captions c
            ORDER BY c.received_at ASC
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return ""

        # Build compact context
        lines = []
        for row in reversed(rows):
            speaker = row["speaker"] or "Unknown"
            text = row["text"]
            lines.append(f"{speaker}: {text}")

        context = "\n".join(lines)
        log.info(f"[DB] Loaded {len(rows)} entries")
        _meeting_context_cache = context
        return context

    except Exception as e:
        log.error(f"[DB] Error: {e}")
        return ""


def create_tts_script():
    """Create persistent TTS worker script"""
    script_path = tempfile.mktemp(suffix="_qwen3_tts_persistent.py")

    script_content = f'''
import sys, json, base64, io, wave, numpy as np, re
sys.path.insert(0, "{str(QWEN_TTS_PATH)}")
from mlx_audio.tts.utils import load_model

_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_model("{QWEN3_TTS_MODEL}")
    return _model

def audio_to_wav_base64(audio, rate):
    audio = np.array(audio, dtype=np.float32)
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes(audio_int16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()

def split_text(text):
    # Clean TTS-unfriendly characters
    text = re.sub(r'[*#@&%$~^|<>\\[\\]]+', '', text)  # Remove special chars
    text = re.sub(r'[*_]+', '', text)  # Remove markdown bold/italic
    text = re.sub(r'\\s+', ' ', text)  # Normalize whitespace

    chunks, has_zh = [], any('\\u4e00'<=c<='\\u9fff' for c in text)
    if has_zh:
        parts, cur, first = re.split(r'([。！？，,、；;])', text), "", False
        for i, p in enumerate(parts):
            cur += p
            thresh = 10 if not first else 25
            if len(cur) >= thresh or (i < len(parts)-1 and parts[i+1] in '。！？'):
                if cur.strip(): chunks.append(cur.strip()); first = True
                cur = ""
    else:
        sents, cur, first = re.split(r'([.!?]\\s)', text), "", False
        for s in sents:
            cur += s
            if len(cur.split()) >= (6 if not first else 18):
                chunks.append(cur.strip()); first = True; cur = ""
    return chunks or [text]

def stream_tts(text):
    model = get_model()
    for chunk in split_text(text):
        if not chunk: continue
        lang = "zh" if any('\\u4e00'<=c<='\\u9fff' for c in chunk) else "en"
        for r in model.generate(text=chunk, voice="Vivian", speed=1.0, lang_code=lang, temperature=0.3, verbose=False):
            print(json.dumps({{"type":"audio_chunk","audio":audio_to_wav_base64(r.audio, r.sample_rate),"sample_rate":r.sample_rate,"duration":len(r.audio)/r.sample_rate}}), flush=True)
            break
    print(json.dumps({{"type":"done"}}), flush=True)

for line in sys.stdin:
    if line.strip():
        try:
            data = json.loads(line)
            if data.get("text"): stream_tts(data["text"])
        except: pass
'''

    with open(script_path, 'w') as f:
        f.write(script_content)

    return script_path


_tts_process = None
_tts_stdin = None


async def start_tts_process(tts_script: str):
    """Start persistent TTS process"""
    global _tts_process, _tts_stdin
    if _tts_process is not None:
        return

    log.info("[TTS] Starting...")
    _tts_process = await asyncio.create_subprocess_exec(
        TTS_PYTHON, tts_script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(QWEN_TTS_PATH) if QWEN_TTS_PATH.exists() else None
    )
    _tts_stdin = _tts_process.stdin

    async def monitor_stderr():
        while True:
            try:
                line = await _tts_process.stderr.readline()
                if not line: break
                msg = line.decode('utf-8', errors='ignore').strip()
                if msg: log.info(f"[TTS] {msg}")
            except: break

    asyncio.create_task(monitor_stderr())
    await asyncio.sleep(3)
    log.info("[TTS] Ready")


async def stream_tts(text: str):
    """Stream TTS audio"""
    if _tts_process is None or _tts_stdin is None:
        return

    log.info(f"[TTS] {text[:30]}...")
    _tts_stdin.write((json.dumps({"text": text}) + "\n").encode())
    await _tts_stdin.drain()

    buffer = bytearray()
    while True:
        try:
            chunk = await _tts_process.stdout.read(1024)
            if not chunk: break
            buffer.extend(chunk)

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.decode('utf-8').strip()
                if not line: continue

                try:
                    data = json.loads(line)
                    if data.get("type") == "audio_chunk":
                        yield data
                    elif data.get("type") == "done":
                        return
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            log.error(f"[TTS] Error: {e}")
            return


def _llm_stream_sync(question: str, context: str = ""):
    """Synchronous LLM stream - runs in thread pool"""
    tts_friendly_prompt = """You are a helpful voice assistant. Answer questions in a natural, spoken style.

Rules for TTS-friendly output:
1. Use conversational, natural language
2. Avoid special characters: * # @ & % $ ~ ^ | < > { } [ ]
3. No markdown, code blocks, or bullet points
4. Use simple punctuation: periods, commas, question marks
5. Write complete sentences, not fragments
6. Avoid abbreviations that are hard to pronounce
7. Be concise but complete"""

    if context:
        system_prompt = f"{tts_friendly_prompt}\n\nContext:\n{context}\n\nAnswer the question based on above context."
    else:
        system_prompt = tts_friendly_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY not in ["", "none", "null"]:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    resp = requests.post(
        f"{LLM_API_BASE}/chat/completions",
        headers=headers,
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "stream": True,
            "temperature": 0.5
        },
        stream=True,
        timeout=60
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data: '):
            continue
        data_str = line[6:]
        if data_str == '[DONE]':
            break
        try:
            data = json.loads(data_str)
            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                yield content
        except json.JSONDecodeError:
            continue


async def call_llm_stream(question: str, context: str = ""):
    """Async wrapper - runs blocking requests in thread pool"""
    log.info(f"[LLM] {question[:50]}...")

    loop = asyncio.get_event_loop()

    # Run the blocking generator in thread pool
    def run_in_thread():
        for chunk in _llm_stream_sync(question, context):
            yield chunk

    # Create an async generator from the sync generator
    gen = run_in_thread()

    try:
        while True:
            # Get next chunk from sync generator without blocking
            chunk = await loop.run_in_executor(None, lambda: next(gen, None))
            if chunk is None:
                break
            yield chunk
    except Exception as e:
        log.error(f"[LLM] Error: {e}")
        yield "Sorry, something went wrong."


class SentenceAccumulator:
    """Detect complete sentences from streaming text"""
    def __init__(self):
        self.buffer = ""
        self.endings = ('。', '！', '？', '.', '!', '?', '…', '\n')

    def add(self, text: str):
        self.buffer += text
        sentences = []

        while True:
            for i, c in enumerate(self.buffer):
                if c in self.endings:
                    s = self.buffer[:i+1].strip()
                    if s: sentences.append(s)
                    self.buffer = self.buffer[i+1:]
                    break
            else:
                break

        return sentences

    def remaining(self) -> str:
        return self.buffer.strip()


async def handle_connection(websocket):
    """Handle WebSocket client"""
    log.info("[WS] Connected")

    try:
        async for msg in websocket:
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "generate":
                question = data.get("question", "").strip()
                if not question:
                    continue

                log.info(f"[WS] {question}")
                start = time.time()

                yield {"type": "start", "question": question, "timestamp": time.time()}

                context = load_meeting_context()
                acc = SentenceAccumulator()
                full, first_char, first_audio = "", None, None
                chunks, dur, sent_count = 0, 0, 0

                async for text_chunk in call_llm_stream(question, context):
                    if first_char is None and text_chunk:
                        first_char = time.time()
                        yield {"type": "first_char", "delay": first_char-start, "timestamp": time.time()}

                    full += text_chunk
                    yield {"type": "llm_delta", "text": text_chunk, "full_text": full, "timestamp": time.time()}

                    for s in acc.add(text_chunk):
                        sent_count += 1
                        log.info(f"[S] {s[:30]}...")
                        async for audio in stream_tts(s):
                            chunks += 1
                            dur += audio.get("duration", 0)
                            if first_audio is None:
                                first_audio = time.time()
                                yield {"type": "first_audio", "delay": first_audio-start, "timestamp": time.time()}
                            yield audio

                # Remaining text
                if acc.remaining():
                    async for audio in stream_tts(acc.remaining()):
                        chunks += 1
                        dur += audio.get("duration", 0)
                        if first_audio is None:
                            yield {"type": "first_audio", "delay": time.time()-start, "timestamp": time.time()}
                        yield audio

                total = time.time() - start
                log.info(f"[TIME] Total: {total:.2f}s, Audio: {dur:.2f}s")

                yield {
                    "type": "done",
                    "total_time": total,
                    "llm_time": total,
                    "total_chunks": chunks,
                    "audio_duration": dur,
                    "sentence_count": sent_count,
                    "char_count": len(full),
                    "text": full,
                    "timestamp": time.time()
                }

            elif data.get("type") == "ping":
                yield {"type": "pong"}

    except websockets.exceptions.ConnectionClosed:
        log.info("[WS] Disconnected")


async def main():
    print("=" * 50)
    print("Qwen3-TTS Fast Server")
    print("=" * 50)
    print(f"  LLM: {LLM_MODEL}")
    print(f"  TTS: {QWEN3_TTS_MODEL}")
    print(f"  WS: ws://localhost:8770")
    print("=" * 50)

    log.info("Loading context...")
    ctx = load_meeting_context()
    log.info(f"✓ Context: {len(ctx)} chars")

    tts_script = create_tts_script()
    await start_tts_process(tts_script)

    try:
        from websockets import serve
        async def ws_handler(ws):
            async for r in handle_connection(ws):
                await ws.send(json.dumps(r))

        async with serve(ws_handler, "localhost", 8770):
            log.info("✓ Server ready")
            await asyncio.Future()

    finally:
        if os.path.exists(tts_script):
            os.remove(tts_script)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Stopped")
