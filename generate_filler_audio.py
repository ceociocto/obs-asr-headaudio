#!/usr/bin/env python3
"""
Generate filler audio files for fast response
预生成填充音频文件，用于降低首音频延迟

Usage:
    # 使用 mlx_audio_env 虚拟环境
    mlx_audio_env/bin/python generate_filler_audio.py

    # 或使用 uv (需要指定环境)
    uv run --with mlx-audio --with numpy python generate_filler_audio.py
"""

import sys
import wave
import io
import base64
import json
import subprocess
from pathlib import Path

# Auto-detect MLX environment
MLX_ENV_PATH = Path(__file__).parent / "mlx_audio_env"
QWEN_TTS_PATH = Path(__file__).parent / "qwen3-tts-apple-silicon"

# Try to use the correct Python environment
if MLX_ENV_PATH.exists() and (MLX_ENV_PATH / "bin" / "python").exists():
    # If we're not already running in the MLX env, re-exec with it
    current_python = Path(sys.executable)
    target_python = MLX_ENV_PATH / "bin" / "python"
    if current_python != target_python:
        print(f"[Info] Switching to MLX environment: {target_python}")
        result = subprocess.run([str(target_python), __file__] + sys.argv[1:])
        sys.exit(result.returncode)

# Add qwen-tts path
if QWEN_TTS_PATH.exists():
    sys.path.insert(0, str(QWEN_TTS_PATH))

from mlx_audio.tts.utils import load_model
import numpy as np

MODEL_NAME = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"

# 填充语料 - 短暂自然的延迟填充词
FILLER_PHRASES = {
    "let_me_see": "Let me see.",
    "let_me_check": "Let me check.",
    "thinking": "Hmm.",
    "one_moment": "One moment.",
    "give_me_sec": "Just a second.",
}


def audio_to_wav(audio, rate: int) -> bytes:
    """Convert audio to WAV bytes"""
    audio = np.array(audio, dtype=np.float32)
    audio_int16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def main():
    print("=" * 50)
    print("Filler Audio Generator")
    print("=" * 50)

    # Create output directory
    filler_dir = Path(__file__).parent / "filler_audio"
    filler_dir.mkdir(exist_ok=True)

    # Load TTS model
    print(f"\nLoading TTS model: {MODEL_NAME}")
    model = load_model(MODEL_NAME)
    print("✓ Model loaded\n")

    results = {}

    for key, phrase in FILLER_PHRASES.items():
        print(f"Generating: '{phrase}'")

        # Detect language
        has_zh = any('\u4e00' <= c <= '\u9fff' for c in phrase)
        lang = "zh" if has_zh else "en"

        # Generate audio (take first chunk only)
        for r in model.generate(
            text=phrase,
            voice="Vivian",
            speed=1.0,
            lang_code=lang,
            temperature=0.3,
            verbose=False
        ):
            wav_data = audio_to_wav(r.audio, r.sample_rate)
            duration = len(r.audio) / r.sample_rate

            # Save WAV file
            wav_path = filler_dir / f"{key}.wav"
            with open(wav_path, "wb") as f:
                f.write(wav_data)

            # Save base64 for reference
            b64_data = base64.b64encode(wav_data).decode()
            results[key] = {
                "text": phrase,
                "duration": round(duration, 3),
                "sample_rate": r.sample_rate,
                "base64": b64_data[:100] + "..."  # Truncated for display
            }

            print(f"  ✓ Saved to {key}.wav ({duration:.2f}s)")
            break  # Only first chunk

    # Save metadata
    meta_path = filler_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"✓ Generated {len(results)} filler audio files")
    print(f"  Location: {filler_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
