#!/usr/bin/env python3
"""
Simple Conversation Server with VibeVoice TTS
- Uses Qwen3.5 2B MLX 8bit for LLM (non-streaming)
- Loads conversation history from captions.db as context
- Streams TTS audio with VibeVoice-Realtime after complete LLM response
- Uses female voice
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
import base64
import numpy as np
import threading
from pathlib import Path
from typing import AsyncGenerator, List, Dict
from datetime import datetime, timedelta

import websockets
import torch

try:
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )
    from vibevoice.modular.streamer import AudioStreamer
    VIBEVOICE_AVAILABLE = True
except ImportError as e:
    print(f"VibeVoice not available: {e}")
    VIBEVOICE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("conversation-server")

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
LOCAL_MODEL_DIR = SCRIPT_DIR / "models" / "VibeVoice-Realtime-0.5B"
HF_MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
MODEL_PATH = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else HF_MODEL_ID
VOICES_DIR = SCRIPT_DIR / "voices" / "streaming_model"

# LLM Configuration
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:12345/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "1234")  # Default API key for local server
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.5-2B-MLX-8bit")

# VibeVoice Configuration
VIBEVOICE_DEVICE = os.getenv("VIBEVOICE_DEVICE", "mps")
VIBEVOICE_INFERENCE_STEPS = int(os.getenv("VIBEVOICE_INFERENCE_STEPS", "8"))
VIBEVOICE_CFG_SCALE = float(os.getenv("VIBEVOICE_CFG_SCALE", "1.5"))
VIBEVOICE_SPEED = float(os.getenv("VIBEVOICE_SPEED", "6"))

# Conversation Configuration
DB_PATH = SCRIPT_DIR / "captions.db"
CONTEXT_LIMIT = int(os.getenv("CONTEXT_LIMIT", "20"))
CONTEXT_TIME_WINDOW = int(os.getenv("CONTEXT_TIME_WINDOW", "3600"))

# Streaming Configuration
SAMPLE_RATE = 24000


class ConversationContext:
    """Manages conversation context from captions database"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = None

    def connect(self):
        """Create database connection"""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

    def close(self):
        """Close database connection"""
        if self._conn:
            self._conn.close()

    def get_recent_context(self, limit: int = None, time_window: int = None, all_data: bool = False) -> List[Dict]:
        """
        Get conversation history from captions table
        """
        if not self._conn:
            self.connect()

        cursor = self._conn.cursor()

        if all_data:
            query = """
                SELECT speaker, text, received_at
                FROM captions
                ORDER BY received_at ASC
            """
            cursor.execute(query)
        else:
            limit = limit or CONTEXT_LIMIT
            time_window = time_window or CONTEXT_TIME_WINDOW
            cutoff_time = (datetime.now() - timedelta(seconds=time_window)).strftime("%Y-%m-%d %H:%M:%S")

            query = """
                SELECT speaker, text, received_at
                FROM captions
                WHERE received_at >= ?
                ORDER BY received_at DESC
                LIMIT ?
            """
            cursor.execute(query, (cutoff_time, limit))

        rows = cursor.fetchall()

        context = []
        for row in rows if all_data else reversed(rows):
            context.append({
                "speaker": row["speaker"] or "Unknown",
                "text": row["text"],
                "time": row["received_at"]
            })

        return context

    def format_context_for_llm(self, context: List[Dict]) -> str:
        """Format conversation history for LLM prompt"""
        if not context:
            return "No previous conversation context."

        lines = ["Previous conversation:"]
        for item in context:
            speaker = item["speaker"]
            text = item["text"]
            lines.append(f"  [{speaker}]: {text}")

        return "\n".join(lines)


class VibeVoiceTTS:
    """VibeVoice TTS Wrapper - optimized for female voice"""

    def __init__(self, model_path: str, voices_dir: Path, device: str = "mps", inference_steps: int = 6, cfg_scale: float = 1.5, speed: float = 1.5):
        self.model_path = model_path
        self.voices_dir = voices_dir
        self.device = device
        self.inference_steps = inference_steps
        self.cfg_scale = cfg_scale
        self.speed = speed
        self.sample_rate = SAMPLE_RATE

        self.processor = None
        self.model = None
        self.voice_presets = {}
        self.default_voice_key = None
        self._voice_cache = {}
        self._initialized = False

        if device == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS not available, falling back to CPU")
            self.device = "cpu"

        self._torch_device = torch.device(self.device)

    def load(self):
        """Load VibeVoice model and voice presets"""
        if not VIBEVOICE_AVAILABLE:
            raise RuntimeError("VibeVoice not installed")

        log.info(f"[VibeVoice] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(str(self.model_path))

        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = "cuda"
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = "cpu"
            attn_impl = "sdpa"

        log.info(f"[VibeVoice] Using device: {device_map}, dtype: {load_dtype}, attn: {attn_impl}")

        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            str(self.model_path),
            torch_dtype=load_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )
        if self.device == "mps":
            self.model = self.model.to("mps")

        self.model.eval()

        from diffusers.schedulers import DPMSolverMultistepScheduler
        self.model.model.noise_scheduler = DPMSolverMultistepScheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self._load_voice_presets()
        self.default_voice_key = self._get_default_voice()
        self._initialized = True

        log.info(f"[VibeVoice] Model loaded, female voice: {self.default_voice_key}")

    def _load_voice_presets(self):
        """Load voice presets"""
        if not self.voices_dir.exists():
            log.warning(f"[VibeVoice] Voices directory not found: {self.voices_dir}")
            return

        for pt_file in self.voices_dir.rglob("*.pt"):
            # Build key from relative path components
            # Path structure: voices/streaming_model/experimental_voices/en/en-Breeze_woman.pt
            # Relative to VOICES_DIR (streaming_model): experimental_voices/en/en-Breeze_woman.pt
            # We want key format: en-en-Breeze_woman (language_code-voice_name)
            rel_path = pt_file.relative_to(self.voices_dir)
            parts = list(rel_path.parts[:-1])  # All directories except filename

            # Remove intermediate directories to get the language code
            # e.g., ['experimental_voices', 'en'] -> ['en']
            if 'experimental_voices' in parts:
                parts.remove('experimental_voices')

            # Build key: last_dir-voice_name (e.g., en-Breeze_woman)
            if parts:
                key = f"{parts[-1]}-{pt_file.stem}"
            else:
                key = pt_file.stem

            self.voice_presets[key] = str(pt_file)

        if self.voice_presets:
            log.info(f"[VibeVoice] Found {len(self.voice_presets)} voice presets")
            for k in sorted(self.voice_presets.keys()):
                log.info(f"  - {k}")
        else:
            log.warning("[VibeVoice] No voice presets found")

    def _get_default_voice(self):
        """Get default voice - prefer female voices"""
        # Female voices in order of preference (note: key format is en-en-Name due to path structure)
        female_voices = ["en-en-Breeze_woman", "en-en-Clarissa_woman", "en-en-Soother_woman", "en-en-Snarkling_woman",
                         "de-de-Spk2_woman", "de-de-Spk4_woman"]

        # Try to find a female voice first
        for female_voice in female_voices:
            if female_voice in self.voice_presets:
                log.info(f"[VibeVoice] Selected female voice: {female_voice}")
                return female_voice

        # Fall back to any English female voice
        for key in self.voice_presets.keys():
            if key.endswith("_woman"):
                log.info(f"[VibeVoice] Selected fallback female voice: {key}")
                return key

        # Fall back to any English voice
        for key in self.voice_presets.keys():
            if key.startswith("en-"):
                log.info(f"[VibeVoice] Selected fallback English voice: {key}")
                return key

        if self.voice_presets:
            voice = next(iter(self.voice_presets))
            log.info(f"[VibeVoice] Selected first available voice: {voice}")
            return voice

        log.warning("[VibeVoice] No voice presets found, using embedded")
        return "embedded"

    def _get_voice_prompt(self, voice_key: str):
        """Load voice preset"""
        if voice_key == "embedded":
            return None

        if voice_key not in self.voice_presets:
            voice_key = self.default_voice_key
            if voice_key not in self.voice_presets:
                return None

        if voice_key not in self._voice_cache:
            pt_path = self.voice_presets[voice_key]
            log.info(f"[VibeVoice] Loading voice preset: {voice_key}")

            try:
                prefilled_outputs = torch.load(
                    pt_path,
                    map_location=self._torch_device,
                    weights_only=False,
                )
                self._voice_cache[voice_key] = prefilled_outputs
            except Exception as e:
                log.error(f"[VibeVoice] Error loading voice preset: {e}")
                return None

        return self._voice_cache.get(voice_key)

    def generate_stream(self, text: str, voice_key: str = None):
        """Generate streaming audio with female voice"""
        if not self._initialized:
            raise RuntimeError("VibeVoice not initialized")

        text = text.strip()
        if not text:
            return

        voice_key = voice_key or self.default_voice_key
        cached_prompt = self._get_voice_prompt(voice_key)

        if cached_prompt is not None:
            inputs = self.processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=cached_prompt,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )

        inputs = {
            k: v.to(self._torch_device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        stop_event = threading.Event()
        errors = []

        def run_generation():
            try:
                import copy
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=self.cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_event.is_set,
                    verbose=False,
                    refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(cached_prompt),
                    speed=self.speed,
                )
            except Exception as exc:
                log.error(f"[VibeVoice] Generation error: {exc}")
                errors.append(exc)
                audio_streamer.end()

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                yield audio_chunk

        finally:
            stop_event.set()
            audio_streamer.end()
            thread.join()

            if errors:
                raise errors[0]

    def chunk_to_base64(self, chunk: np.ndarray) -> str:
        """Convert audio chunk to base64 encoded WAV"""
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)

        num_samples = len(pcm)
        wav_header = bytearray([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0x00, 0x00, 0x00, 0x00,  # file size
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # chunk size
            0x01, 0x00,              # format
            0x01, 0x00,              # channels
            0x80, 0x3E, 0x00, 0x00,  # sample rate (24000)
            0x00, 0x7D, 0x00, 0x00,  # byte rate
            0x02, 0x00,              # block align
            0x10, 0x00,              # bits per sample
            0x64, 0x61, 0x74, 0x61,  # "data"
            0x00, 0x00, 0x00, 0x00,  # data size
        ])

        file_size = 36 + num_samples * 2
        wav_header[4:8] = file_size.to_bytes(4, 'little')
        wav_header[40:44] = (num_samples * 2).to_bytes(4, 'little')

        wav_data = bytes(wav_header) + pcm.tobytes()
        return base64.b64encode(wav_data).decode('utf-8')


# Global instances
tts_engine = None
context_manager = None


def init_tts():
    """Initialize TTS engine"""
    global tts_engine
    try:
        tts_engine = VibeVoiceTTS(
            model_path=str(MODEL_PATH),
            voices_dir=VOICES_DIR,
            device=VIBEVOICE_DEVICE,
            inference_steps=VIBEVOICE_INFERENCE_STEPS,
            cfg_scale=VIBEVOICE_CFG_SCALE,
            speed=VIBEVOICE_SPEED
        )
        tts_engine.load()
        log.info("✓ VibeVoice TTS initialized (female voice)")
        log.info(f"  Config: inference_steps={VIBEVOICE_INFERENCE_STEPS}, cfg_scale={VIBEVOICE_CFG_SCALE}, speed={VIBEVOICE_SPEED}")
        return True
    except Exception as e:
        log.error(f"✗ VibeVoice initialization failed: {e}")
        return False


def init_context():
    """Initialize conversation context manager"""
    global context_manager
    try:
        context_manager = ConversationContext(DB_PATH)
        context_manager.connect()
        log.info(f"✓ Context manager initialized ({DB_PATH})")
        return True
    except Exception as e:
        log.error(f"✗ Context manager initialization failed: {e}")
        return False


async def call_llm_api(question: str, context_str: str) -> str:
    """Call LLM API with conversation context (non-streaming)"""
    try:
        import requests

        system_prompt = f"""You are a helpful assistant engaged in a conversation. Use the following conversation history to provide relevant, contextual responses.

{context_str}

Provide concise, natural responses. If the question is unrelated to the conversation, still be helpful but brief."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        log.info(f"[LLM] Calling API for: {question[:30]}...")

        def _do_request():
            headers = {"Content-Type": "application/json"}
            if LLM_API_KEY and LLM_API_KEY not in ["", "none", "null"]:
                headers["Authorization"] = f"Bearer {LLM_API_KEY}"

            response = requests.post(
                f"{LLM_API_BASE}/chat/completions",
                headers=headers,
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 2048,
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=120
            )

            response.raise_for_status()
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')

        content = await asyncio.to_thread(_do_request)
        log.info(f"[LLM] Received response: {len(content)} chars")
        return content

    except Exception as e:
        log.error(f"LLM API error: {e}")
        raise


async def stream_answer(question: str) -> AsyncGenerator[dict, None]:
    """Stream answer with TTS (non-streaming LLM)"""

    start_time = time.time()

    log.info(f"[START] Question: {question}")

    # Get conversation context
    context = context_manager.get_recent_context(all_data=True)
    context_str = context_manager.format_context_for_llm(context)

    log.info(f"[CONTEXT] Loaded {len(context)} total messages from database")

    yield {
        "type": "start",
        "question": question,
        "context_count": len(context),
        "timestamp": time.time()
    }

    if tts_engine is None:
        yield {
            "type": "error",
            "message": "TTS engine not initialized"
        }
        return

    # Step 1: Get complete LLM response
    llm_start_time = time.time()
    llm_text = await call_llm_api(question, context_str)
    llm_end_time = time.time()
    llm_time = llm_end_time - llm_start_time

    log.info(f"[LLM] Complete in {llm_time:.3f}s")

    yield {
        "type": "llm_complete",
        "text": llm_text,
        "llm_time": llm_time,
        "timestamp": time.time()
    }

    # Step 2: Generate TTS audio (streaming generation, but for complete text)
    tts_start_time = time.time()

    log.info(f"[TTS] Starting generation for {len(llm_text)} chars")

    try:
        def collect_audio():
            chunks = []
            for chunk in tts_engine.generate_stream(llm_text):
                chunks.append(chunk)
            return chunks

        audio_chunks = await asyncio.to_thread(collect_audio)

        tts_end_time = time.time()
        tts_time = tts_end_time - tts_start_time

        if audio_chunks:
            combined = np.concatenate(audio_chunks)
            duration = len(combined) / SAMPLE_RATE
            rt_f = tts_time / duration if duration > 0 else 0

            log.info(f"[TTS] Complete: {duration:.2f}s audio, RTF={rt_f:.2f}x")

            # Send complete audio as one chunk
            yield {
                "type": "audio",
                "audio": tts_engine.chunk_to_base64(combined),
                "duration": duration,
                "sample_rate": SAMPLE_RATE,
                "text": llm_text,
                "timestamp": time.time(),
                "timings": {
                    "llm_time": round(llm_time, 3),
                    "tts_time": round(tts_time, 3),
                    "audio_duration": round(duration, 3),
                    "rtf": round(rt_f, 3)
                }
            }

    except Exception as e:
        log.error(f"[TTS] Error: {e}")
        yield {
            "type": "error",
            "message": f"TTS error: {str(e)}"
        }
        return

    total_time = time.time() - start_time
    log.info(f"[TIMING] Session Complete: {total_time:.3f}s")

    yield {
        "type": "done",
        "total_time": round(total_time, 3),
        "full_text": llm_text,
        "timestamp": time.time(),
        "timings": {
            "llm_time_ms": round(llm_time * 1000, 1),
            "tts_time_ms": round(tts_time * 1000, 1),
            "total_time_ms": round(total_time * 1000, 1)
        }
    }


async def handle_connection(websocket):
    """WebSocket connection handler"""
    log.info("[WS] Client connected")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")

            if msg_type == "generate":
                question = data.get("question", "").strip()
                if not question:
                    continue

                log.info(f"[WS] Generate: {question}")

                try:
                    async for response in stream_answer(question):
                        await websocket.send(json.dumps(response))
                except Exception as e:
                    log.error(f"[WS] Error: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))

            elif msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

            elif msg_type == "context":
                context = context_manager.get_recent_context()
                await websocket.send(json.dumps({
                    "type": "context",
                    "count": len(context),
                    "items": context
                }))

    except websockets.exceptions.ConnectionClosed:
        log.info("[WS] Client disconnected")


def main():
    print("=" * 60)
    print("Simple Conversation Server with VibeVoice TTS")
    print("=" * 60)
    print("LLM: Non-streaming (complete response first)")
    print("TTS: VibeVoice-Realtime with female voice")
    print("=" * 60)

    if not VIBEVOICE_AVAILABLE:
        print("✗ VibeVoice not installed")
        print("Run: pip install vibevoice[streamingtts]")
        return

    if not init_tts():
        print("✗ TTS initialization failed")
        return

    if not init_context():
        print("✗ Context manager initialization failed")
        return

    print(f"  Model: {MODEL_PATH}")
    print(f"  Device: {VIBEVOICE_DEVICE}")
    print(f"  LLM: {LLM_MODEL}")
    print(f"  Database: {DB_PATH}")
    print(f"  WebSocket: ws://localhost:8772")
    print("=" * 60)
    print("Open conversation_simple_demo.html in your browser")
    print("=" * 60)

    from websockets import serve

    async def run_server():
        async with serve(handle_connection, "localhost", 8772):
            await asyncio.Future()

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        log.info("Server stopped")
        if context_manager:
            context_manager.close()


if __name__ == "__main__":
    main()
