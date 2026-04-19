#!/usr/bin/env python3
"""
Conversation Server with VibeVoice TTS
- Uses Qwen3.5 2B MLX 8bit for LLM
- Loads conversation history from captions.db as context
- Streams responses with VibeVoice-Realtime TTS
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
LLM_API_KEY = os.getenv("LLM_API_KEY", "1234")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.5-2B-MLX-8bit")

# VibeVoice Configuration
VIBEVOICE_DEVICE = os.getenv("VIBEVOICE_DEVICE", "mps")
VIBEVOICE_INFERENCE_STEPS = int(os.getenv("VIBEVOICE_INFERENCE_STEPS", "8"))
VIBEVOICE_CFG_SCALE = float(os.getenv("VIBEVOICE_CFG_SCALE", "1.5"))
VIBEVOICE_SPEED = float(os.getenv("VIBEVOICE_SPEED", "6"))  # Lower = better quality, Higher = faster (1.0-3.0 recommended)

# Conversation Configuration
DB_PATH = SCRIPT_DIR / "captions.db"
CONTEXT_LIMIT = int(os.getenv("CONTEXT_LIMIT", "20"))  # Number of recent captions to include
CONTEXT_TIME_WINDOW = int(os.getenv("CONTEXT_TIME_WINDOW", "3600"))  # Seconds

# Streaming Configuration
MIN_CHUNK_CHARS = 20
MIN_CHUNK_WORDS = 5
CHUNK_DELIMITERS = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
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

        Args:
            limit: Maximum number of captions to retrieve (ignored if all_data=True)
            time_window: Only retrieve captions from last N seconds (ignored if all_data=True)
            all_data: If True, fetch all conversation data without restrictions
        """
        if not self._conn:
            self.connect()

        cursor = self._conn.cursor()

        if all_data:
            # Fetch ALL conversation data
            query = """
                SELECT speaker, text, received_at
                FROM captions
                ORDER BY received_at ASC
            """
            cursor.execute(query)
        else:
            # Get recent captions within time window
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

        # Build context in chronological order
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
    """VibeVoice TTS Wrapper"""

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
        self._pre_generated_audio = {}  # Cache for pre-generated audio

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

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                str(self.model_path),
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
            if self.device == "mps":
                self.model = self.model.to("mps")
        except Exception as e:
            log.error(f"[VibeVoice] Error loading model: {e}")
            raise

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

        log.info(f"[VibeVoice] Model loaded, default voice: {self.default_voice_key}")

        # Pre-generate common phrases
        self._pre_generate_phrases()

    def _load_voice_presets(self):
        """Load voice presets"""
        if not self.voices_dir.exists():
            log.warning(f"[VibeVoice] Voices directory not found: {self.voices_dir}")
            return

        for pt_file in self.voices_dir.rglob("*.pt"):
            if pt_file.parent.name != "streaming_model":
                key = f"{pt_file.parent.name}-{pt_file.stem}"
            else:
                key = pt_file.stem
            self.voice_presets[key] = str(pt_file)

        if self.voice_presets:
            log.info(f"[VibeVoice] Found {len(self.voice_presets)} voice presets")
        else:
            log.warning("[VibeVoice] No voice presets found")

    def _get_default_voice(self):
        """Get default voice - prefer female voices"""
        # Female voices in order of preference (actual key format: en-en-Breeze_woman)
        female_voices = ["en-en-Breeze_woman", "en-en-Clarissa_woman", "en-en-Soother_woman", "en-en-Snarkling_woman"]

        # Try to find a female voice first
        for female_voice in female_voices:
            if female_voice in self.voice_presets:
                return female_voice

        # Fall back to any English voice
        for key in self.voice_presets.keys():
            if key.startswith("en-en-"):
                return key
        if self.voice_presets:
            return next(iter(self.voice_presets))
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

    def _pre_generate_phrases(self):
        """Pre-generate common phrases for instant playback"""
        if not self._initialized:
            return

        phrases = {
            "thinking": "Let me see...",
        }

        log.info("[VibeVoice] Pre-generating common phrases...")

        for key, text in phrases.items():
            try:
                chunks = []
                for chunk in self.generate_stream(text):
                    chunks.append(chunk)

                if chunks:
                    combined = np.concatenate(chunks)
                    self._pre_generated_audio[key] = combined
                    duration = len(combined) / self.sample_rate
                    log.info(f"[VibeVoice] Pre-generated '{key}': {duration:.2f}s")
            except Exception as e:
                log.error(f"[VibeVoice] Failed to pre-generate '{key}': {e}")

    def get_pre_generated_audio(self, key: str) -> np.ndarray:
        """Get pre-generated audio by key"""
        return self._pre_generated_audio.get(key)

    def generate_stream(self, text: str, voice_key: str = None):
        """Generate streaming audio"""
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
        log.info("✓ VibeVoice TTS initialized")
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


async def stream_llm_api(question: str, context_str: str) -> AsyncGenerator[str, None]:
    """Call LLM API with conversation context"""
    try:
        import requests
        import os

        system_prompt = f"""You are a helpful assistant engaged in a conversation. Use the following conversation history to provide relevant, contextual responses.

{context_str}

Provide concise, natural responses. If the question is unrelated to the conversation, still be helpful but brief."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        log.info(f"[LLM] Starting stream for: {question[:30]}...")

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
                    "stream": True,
                    "temperature": 0.7
                },
                stream=True,
                timeout=120
            )

            response.raise_for_status()

            chunks = []
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                chunks.append(content)
                        except json.JSONDecodeError:
                            continue
            return chunks

        chunks = await asyncio.to_thread(_do_request)

        for content in chunks:
            yield content

    except Exception as e:
        log.error(f"LLM API error: {e}")
        raise


class TextAccumulator:
    """Text accumulator for chunking"""

    def __init__(self):
        self.buffer = ""
        self.word_count = 0

    def add(self, text: str) -> bool:
        """Add text, return whether TTS should be triggered"""
        self.buffer += text
        words = self.buffer.split()
        self.word_count = len(words)

        should_trigger = False

        for delimiter in CHUNK_DELIMITERS:
            if delimiter in self.buffer:
                if len(self.buffer) >= MIN_CHUNK_CHARS:
                    should_trigger = True
                break

        if self.word_count >= MIN_CHUNK_WORDS and len(self.buffer) >= MIN_CHUNK_CHARS:
            should_trigger = True

        return should_trigger


async def stream_answer(question: str) -> AsyncGenerator[dict, None]:
    """Stream answer with TTS"""

    start_time = time.time()
    llm_start_time = None
    llm_first_token_time = None
    llm_end_time = None
    first_tts_chunk_time = None
    pre_audio_sent = False  # Track if pre-generated audio was sent

    log.info(f"[START] Question: {question}")

    # Get ALL conversation context
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

    # Initialize counters before sending pre-generated audio
    accumulator = TextAccumulator()
    full_text = ""
    first_audio_sent = False
    sent_chunks = 0
    total_tts_samples = 0

    # Send pre-generated "thinking" audio immediately
    pre_audio = tts_engine.get_pre_generated_audio("thinking")
    if pre_audio is not None:
        pre_audio_sent = True
        pre_audio_time = time.time()
        pre_audio_delay = (pre_audio_time - start_time) * 1000
        pre_audio_duration = len(pre_audio) / SAMPLE_RATE

        log.info(f"[TIMING] Pre-generated audio sent: {pre_audio_delay:.0f}ms, duration: {pre_audio_duration:.2f}s")

        yield {
            "type": "audio",
            "audio": tts_engine.chunk_to_base64(pre_audio),
            "duration": pre_audio_duration,
            "sample_rate": SAMPLE_RATE,
            "chunk_index": 0,
            "text": "Let me see...",
            "timestamp": pre_audio_time,
            "is_pre_generated": True,
            "delay": 1,  # Delay playback by 0.5 seconds
            "timings": {
                "pre_audio_delay_ms": round(pre_audio_delay, 1),
                "pre_audio_duration_s": round(pre_audio_duration, 3)
            }
        }

        sent_chunks += 1
        total_tts_samples += len(pre_audio)

    log.info("[LLM] Starting stream...")
    llm_start_time = time.time()

    async for content in stream_llm_api(question, context_str):
        if not content:
            continue

        if llm_first_token_time is None:
            llm_first_token_time = time.time()
            llm_first_token_delay = (llm_first_token_time - llm_start_time) * 1000
            log.info(f"[TIMING] LLM First Token: {llm_first_token_delay:.0f}ms")

        full_text += content

        yield {
            "type": "text_update",
            "text": full_text,
            "timestamp": time.time()
        }

        should_trigger = accumulator.add(content)

        if should_trigger:
            text_to_speak = accumulator.buffer.strip()
            if text_to_speak:
                chunk_index = sent_chunks + 1
                tts_start = time.time()

                log.info(f"[TTS-{chunk_index}] Generating: '{text_to_speak[:50]}...'")

                try:
                    def collect_audio():
                        chunks = []
                        for chunk in tts_engine.generate_stream(text_to_speak):
                            chunks.append(chunk)
                        return chunks

                    audio_chunks = await asyncio.to_thread(collect_audio)

                    tts_end = time.time()
                    tts_generation_time = tts_end - tts_start

                    # Track first TTS chunk time
                    if first_tts_chunk_time is None:
                        first_tts_chunk_time = tts_end
                        first_tts_chunk_delay = (first_tts_chunk_time - start_time) * 1000
                        log.info(f"[TIMING] First TTS Chunk: {first_tts_chunk_delay:.0f}ms")

                    if audio_chunks:
                        combined = np.concatenate(audio_chunks)
                        duration = len(combined) / SAMPLE_RATE
                        total_tts_samples += len(combined)
                        rt_f = tts_generation_time / duration if duration > 0 else 0

                        yield {
                            "type": "audio",
                            "audio": tts_engine.chunk_to_base64(combined),
                            "duration": duration,
                            "sample_rate": SAMPLE_RATE,
                            "chunk_index": chunk_index,
                            "text": text_to_speak,
                            "timestamp": time.time(),
                            "timings": {
                                "tts_generation_time": round(tts_generation_time, 3),
                                "rtf": round(rt_f, 3)
                            }
                        }

                        sent_chunks += 1
                        log.info(f"[TTS-{chunk_index}] Sent: {duration:.2f}s audio, RTF={rt_f:.2f}x")

                        if not first_audio_sent:
                            first_audio_sent = True
                            first_audio_delay = tts_end - start_time
                            log.info(f"[TIMING] First Audio Delay: {first_audio_delay:.3f}s")

                            yield {
                                "type": "first_chunk",
                                "delay": first_audio_delay,
                                "timings": {
                                    "llm_first_token_ms": round(llm_first_token_delay, 1),
                                    "tts_first_chunk_ms": round(first_tts_chunk_delay, 1),
                                    "tts_generation_s": round(tts_generation_time, 3),
                                    "total_first_audio_s": round(first_audio_delay, 3),
                                    "rtf": round(rt_f, 3)
                                }
                            }

                except Exception as e:
                    log.error(f"[TTS] Error: {e}")

                accumulator = TextAccumulator()

    # Track LLM completion time
    llm_end_time = time.time()
    llm_total_time = llm_end_time - llm_start_time
    log.info(f"[TIMING] LLM Complete: {llm_total_time:.3f}s")

    # Process remaining text
    if accumulator.buffer.strip():
        text_to_speak = accumulator.buffer.strip()
        chunk_index = sent_chunks + 1
        tts_start = time.time()

        log.info(f"[TTS-{chunk_index}] Final: '{text_to_speak[:30]}...'")

        try:
            def collect_final_audio():
                chunks = []
                for chunk in tts_engine.generate_stream(text_to_speak):
                    chunks.append(chunk)
                return chunks

            audio_chunks = await asyncio.to_thread(collect_final_audio)

            tts_end = time.time()
            final_tts_time = tts_end - tts_start

            if audio_chunks:
                combined = np.concatenate(audio_chunks)
                duration = len(combined) / SAMPLE_RATE
                total_tts_samples += len(combined)
                rt_f = final_tts_time / duration if duration > 0 else 0

                yield {
                    "type": "audio",
                    "audio": tts_engine.chunk_to_base64(combined),
                    "duration": duration,
                    "sample_rate": SAMPLE_RATE,
                    "chunk_index": chunk_index,
                    "text": text_to_speak,
                    "timestamp": time.time(),
                    "timings": {
                        "tts_generation_time": round(final_tts_time, 3),
                        "rtf": round(rt_f, 3)
                    }
                }

                sent_chunks += 1
                log.info(f"[TTS-{chunk_index}] Completed: {duration:.2f}s audio, RTF={rt_f:.2f}x")

        except Exception as e:
            log.error(f"[TTS] Error: {e}")

    total_time = time.time() - start_time
    total_audio_duration = total_tts_samples / SAMPLE_RATE if total_tts_samples > 0 else 0

    log.info(f"[TIMING] Session Complete: {total_time:.3f}s, {sent_chunks} chunks, {total_audio_duration:.2f}s audio")

    # Detailed timing statistics
    llm_first_token_ms = (llm_first_token_time - llm_start_time) * 1000 if llm_first_token_time else 0
    llm_total_ms = llm_total_time * 1000 if llm_end_time else 0
    first_tts_chunk_ms = (first_tts_chunk_time - start_time) * 1000 if first_tts_chunk_time else 0
    pre_audio_delay_ms = (pre_audio_time - start_time) * 1000 if pre_audio_sent else 0

    yield {
        "type": "done",
        "total_time": round(total_time, 3),
        "tts_chunks": sent_chunks,
        "audio_duration": round(total_audio_duration, 2),
        "full_text": full_text,
        "timestamp": time.time(),
        "pre_audio_sent": pre_audio_sent,
        "timings": {
            "pre_audio_delay_ms": round(pre_audio_delay_ms, 1) if pre_audio_sent else None,
            "llm_first_token_ms": round(llm_first_token_ms, 1),
            "llm_complete_ms": round(llm_total_ms, 1),
            "first_tts_chunk_ms": round(first_tts_chunk_ms, 1),
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
                # Request to see current context
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
    print("Conversation Server with VibeVoice TTS")
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
    print(f"  Context Limit: {CONTEXT_LIMIT} messages")
    print(f"  WebSocket: ws://localhost:8771")
    print("=" * 60)
    print("Open conversation_demo.html in your browser")
    print("=" * 60)

    from websockets import serve

    async def run_server():
        async with serve(handle_connection, "localhost", 8771):
            await asyncio.Future()

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        log.info("Server stopped")
        if context_manager:
            context_manager.close()


if __name__ == "__main__":
    import os
    main()
