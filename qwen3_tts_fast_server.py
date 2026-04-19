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

# --- Filler Audio (预录制填充音频) ---
# 这些是简短的延迟填充音频，可以显著降低首音频延迟
# 使用 Qwen-TTS 预生成并转为 base64
FILLER_AUDIO = {
    "let_me_see": "",  # "Let me see..." (~0.5s)
    "let_me_check": "",  # "Let me check..." (~0.5s)
    "thinking": "",  # "Hmm..." (~0.3s)
    "one_moment": "",  # "One moment..." (~0.5s)
    "give_me_sec": "",  # "Just a second..." (~0.6s)
}

MLX_ENV_PATH = Path(__file__).parent / "mlx_audio_env"
QWEN_TTS_PATH = Path(__file__).parent / "qwen3-tts-apple-silicon"
CAPTIONS_DB_PATH = Path(__file__).parent / "captions.db"

if MLX_ENV_PATH.exists() and (MLX_ENV_PATH / "bin" / "python").exists():
    TTS_PYTHON = str(MLX_ENV_PATH / "bin" / "python")
else:
    TTS_PYTHON = sys.executable

_meeting_context_cache = None
_filler_audio_cache = {}  # Cache for loaded filler audio files


def load_filler_audio():
    """Load pre-recorded filler audio files from disk"""
    global _filler_audio_cache
    filler_dir = Path(__file__).parent / "filler_audio"
    if not filler_dir.exists():
        return

    for name, _ in FILLER_AUDIO.items():
        audio_path = filler_dir / f"{name}.wav"
        if audio_path.exists():
            with open(audio_path, "rb") as f:
                audio_data = f.read()
                _filler_audio_cache[name] = base64.b64encode(audio_data).decode()
                log.info(f"[Filler] Loaded {name}")


def select_filler_audio(question: str) -> str:
    """Select appropriate filler audio based on question type"""
    q_lower = question.lower()

    # Question type matching
    if any(w in q_lower for w in ["what", "tell me", "explain", "describe", "how"]):
        return "let_me_see"
    elif any(w in q_lower for w in ["find", "search", "look", "check", "where"]):
        return "let_me_check"
    elif any(w in q_lower for w in ["calculate", "compute", "math", "count"]):
        return "thinking"
    else:
        # Randomly pick one to avoid monotony
        import random
        return random.choice(["let_me_see", "one_moment", "give_me_sec"])


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


class PhraseAccumulator:
    """更激进的短语分割器 - 按短语/从句分割，不等完整句子

    分割策略:
    - 逗号、分号、冒号后分割
    - 连词前分割 (and, but, or, so, because...)
    - 介词短语后分割
    - 最小长度保护 (避免过短的片段)
    """
    def __init__(self, min_length: int = 8, max_length: int = 40):
        self.buffer = ""
        self.min_length = min_length
        self.max_length = max_length

        # 分割标记
        self.split_before = {
            # 连词
            'and', 'but', 'or', 'so', 'because', 'however', 'therefore', 'meanwhile',
            'although', 'though', 'while', 'unless', 'since', 'whereas',
            # 中文连词
            '但是', '而且', '因为', '所以', '不过', '虽然', '但是', '然后',
        }
        self.split_after = {
            # 标点
            ',', ';', ':', '，', '；', '：',
            # 介词/副词结尾
        }

    def add(self, text: str):
        self.buffer += text
        phrases = []

        while len(self.buffer) >= self.min_length:
            # 优先按标点分割
            for i, c in enumerate(self.buffer):
                if c in self.split_after:
                    phrase = self.buffer[:i+1].strip()
                    if len(phrase) >= self.min_length:
                        phrases.append(phrase)
                        self.buffer = self.buffer[i+1:].strip()
                        break
            else:
                # 没找到标点，检查连词前分割
                words = self.buffer.split()
                for i, word in enumerate(words[1:], 1):  # 跳过第一个词
                    if word.lower() in self.split_before:
                        phrase = ' '.join(words[:i]).strip()
                        if len(phrase) >= self.min_length:
                            phrases.append(phrase)
                            self.buffer = ' '.join(words[i:]).strip()
                            break
                else:
                    # 没找到分割点，检查是否超过最大长度
                    if len(self.buffer) >= self.max_length:
                        # 强制按词分割
                        words = self.buffer.split()
                        mid = len(words) // 2
                        phrase = ' '.join(words[:mid]).strip()
                        if phrase:
                            phrases.append(phrase)
                            self.buffer = ' '.join(words[mid:]).strip()
                    else:
                        # 等待更多内容
                        break

        return phrases

    def remaining(self) -> str:
        return self.buffer.strip()


# --- LLM 预测回答开头 ---
# 根据问题类型预测回答的开头，提前进行 TTS
PREDICTIVE_RESPONSES = {
    # 问候类
    "greeting": [
        "Hello! ",
        "Hi there! ",
        "你好！",
    ],
    # 时间类
    "time": [
        "Let me check the time for you. ",
        "The current time is ",
        "让我查一下时间",
    ],
    # 天气类
    "weather": [
        "Let me check the weather for you. ",
        "Regarding the weather, ",
        "让我查看天气情况",
    ],
    # 计算类
    "calculation": [
        "Let me calculate that for you. ",
        "Here's the calculation: ",
        "让我来计算一下",
    ],
    # 定义类
    "definition": [
        "Let me explain that for you. ",
        "The definition is ",
        "让我来解释一下",
    ],
    # 默认
    "default": [
        "Let me think about that. ",
        "Good question! ",
        "让我想想",
    ]
}


def predict_response_start(question: str) -> str:
    """根据问题类型预测回答开头"""
    q_lower = question.lower()

    # 检测问题类型
    if any(w in q_lower for w in ["hello", "hi", "hey", "你好", "您好"]):
        return _random_choice(PREDICTIVE_RESPONSES["greeting"])
    elif any(w in q_lower for w in ["time", "几点", "时间", "when"]):
        return _random_choice(PREDICTIVE_RESPONSES["time"])
    elif any(w in q_lower for w in ["weather", "天气", "temperature"]):
        return _random_choice(PREDICTIVE_RESPONSES["weather"])
    elif any(w in q_lower for w in ["calculate", "加", "减", "乘", "除", "+", "-", "*", "/"]):
        return _random_choice(PREDICTIVE_RESPONSES["calculation"])
    elif any(w in q_lower for w in ["what is", "define", "definition", "什么是", "定义"]):
        return _random_choice(PREDICTIVE_RESPONSES["definition"])
    else:
        return _random_choice(PREDICTIVE_RESPONSES["default"])


def _random_choice(choices: list) -> str:
    import random
    return random.choice(choices)


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

                # === Step 1: 立即发送填充音频 (如果可用) ===
                filler_sent = False
                filler_key = select_filler_audio(question)
                if filler_key in _filler_audio_cache:
                    filler_b64 = _filler_audio_cache[filler_key]
                    # 发送填充音频
                    yield {
                        "type": "audio_chunk",
                        "audio": filler_b64,
                        "sample_rate": 24000,
                        "duration": 0.5,  # 预估时长
                        "is_filler": True,
                        "timestamp": time.time()
                    }
                    filler_sent = True
                    first_audio = time.time()
                    log.info(f"[Filler] Sent '{filler_key}' in {first_audio-start:.3f}s")
                    yield {"type": "first_audio", "delay": first_audio-start, "is_filler": True, "timestamp": time.time()}

                # === Step 2: 预测性响应开头 (提前 TTS) ===
                # 在 LLM 生成的同时，先对预测的开头进行 TTS
                predictive_start = predict_response_start(question)
                log.info(f"[Predict] '{predictive_start.strip()}'")

                # === Step 3: 并行启动 LLM 生成 + 预测开头 TTS ===
                context = load_meeting_context()

                # 使用 PhraseAccumulator 进行更激进的分割
                acc = PhraseAccumulator(min_length=8, max_length=40)
                full, first_char = "", None
                chunks, dur, sent_count = 0, 0, 0
                predictive_audio_sent = False

                async for text_chunk in call_llm_stream(question, context):
                    if first_char is None and text_chunk:
                        first_char = time.time()
                        yield {"type": "first_char", "delay": first_char-start, "timestamp": time.time()}

                    full += text_chunk
                    yield {"type": "llm_delta", "text": text_chunk, "full_text": full, "timestamp": time.time()}

                    # 发送预测开头的 TTS (仅一次)
                    if not predictive_audio_sent and len(full) > 3:
                        predictive_audio_sent = True
                        # 检查预测是否与实际开头匹配
                        if predictive_start.strip() in full[:len(predictive_start)+10]:
                            # 预测准确，发送预测音频
                            async for audio in stream_tts(predictive_start.strip()):
                                chunks += 1
                                dur += audio.get("duration", 0)
                                yield {**audio, "is_predictive": True}
                            log.info(f"[Predict] Audio sent, matches actual response")
                        else:
                            log.info(f"[Predict] Skipped, doesn't match actual")

                    # 使用短语分割器进行更激进的 TTS
                    for phrase in acc.add(text_chunk):
                        sent_count += 1
                        log.info(f"[P] {phrase[:30]}...")
                        async for audio in stream_tts(phrase):
                            chunks += 1
                            dur += audio.get("duration", 0)
                            if not filler_sent and first_audio is None:
                                first_audio = time.time()
                                yield {"type": "first_audio", "delay": first_audio-start, "timestamp": time.time()}
                            yield audio

                # Remaining text
                if acc.remaining():
                    async for audio in stream_tts(acc.remaining()):
                        chunks += 1
                        dur += audio.get("duration", 0)
                        if not filler_sent and first_audio is None:
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
                    "filler_used": filler_sent,
                    "predictive_used": predictive_audio_sent,
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

    log.info("Loading filler audio...")
    load_filler_audio()
    log.info(f"✓ Filler audio: {len(_filler_audio_cache)} loaded")

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
