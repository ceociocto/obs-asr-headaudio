#!/usr/bin/env python3
"""
Qwen3-TTS Wrapper for WebSocket Server
"""

import asyncio
import subprocess
import json
import wave
import io
import os
import sys
from pathlib import Path

# 添加 qwen3-tts 到路径
QWEN_TTS_PATH = Path(__file__).parent / "qwen3-tts-apple-silicon"
sys.path.insert(0, str(QWEN_TTS_PATH))

try:
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio
    MLX_AVAILABLE = True
except ImportError as e:
    MLX_AVAILABLE = False
    print(f"MLX not available: {e}")


class Qwen3TTS:
    def __init__(self, model_path=None):
        """初始化 Qwen3-TTS 模型"""
        self.model = None
        self.model_path = model_path
        self.temp_dir = None

        if not MLX_AVAILABLE:
            print("Warning: MLX not available")
            return

        if model_path is None:
            model_path = self._find_model()

        if model_path:
            self.load_model(model_path)

    def _find_model(self):
        """查找已下载的模型"""
        # 检查 qwen3-tts 项目的 models 目录
        models_dir = QWEN_TTS_PATH / "models"
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    # 检查 snapshots
                    snapshots = model_dir / "snapshots"
                    if snapshots.exists():
                        for snapshot in snapshots.iterdir():
                            if snapshot.is_dir():
                                config = snapshot / "config.json"
                                if config.exists():
                                    print(f"Found model: {model_dir.name}")
                                    return str(snapshot)
                    # 直接检查
                    config = model_dir / "config.json"
                    if config.exists():
                        print(f"Found model: {model_dir.name}")
                        return str(model_dir)

        print("No model found!")
        print("Please download CustomVoice model from:")
        print("https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
        print("\nExtract to:")
        print(f"{models_dir}/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/")
        return None

    def load_model(self, model_path):
        """加载 TTS 模型"""
        try:
            print(f"Loading TTS model from: {model_path}")
            self.model = load_model(model_path)
            print("✓ TTS model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load TTS model: {e}")
            self.model = None

    def synthesize(self, text, voice="Vivian", speed=1.0, emotion="Normal tone"):
        """
        合成语音

        Args:
            text: 要合成的文本
            voice: 声音选择 (Vivian, Serena, Ryan, etc.)
            speed: 语速 (0.8-1.3)
            emotion: 情感描述

        Returns:
            成功返回音频数据（bytes），失败返回 None
        """
        if self.model is None:
            print("TTS model not loaded")
            return None

        if not text or not text.strip():
            return None

        try:
            import tempfile
            import shutil

            # 创建临时目录
            self.temp_dir = tempfile.mkdtemp(prefix="qwen_tts_")

            print(f"Synthesizing: '{text[:50]}...'")

            # 使用 generate_audio 函数
            generate_audio(
                model=self.model,
                text=text,
                voice=voice,
                instruct=emotion,
                speed=speed,
                output_path=self.temp_dir
            )

            # 读取生成的音频文件
            audio_file = os.path.join(self.temp_dir, "audio_000.wav")

            if os.path.exists(audio_file):
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()

                # 清理临时文件
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.temp_dir = None

                print(f"✓ Generated {len(audio_data)} bytes")
                return audio_data
            else:
                print("✗ Audio file not generated")
                # 清理
                if self.temp_dir:
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    self.temp_dir = None
                return None

        except Exception as e:
            print(f"✗ TTS synthesis error: {e}")
            # 清理
            if self.temp_dir:
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.temp_dir = None
            return None

    def get_available_voices(self):
        """获取可用声音列表"""
        return ["Vivian", "Serena", "Ryan", "Aiden", "Ethan", "Chelsie"]


# 全局 TTS 实例
_tts_instance = None

def get_tts():
    """获取 TTS 实例（单例）"""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = Qwen3TTS()
    return _tts_instance


# 测试
async def test():
    tts = get_tts()

    if tts.model is None:
        print("Model not loaded!")
        return

    # 测试合成
    audio = tts.synthesize("你好，这是一个测试。", voice="Vivian")

    if audio:
        # 保存测试文件
        output = Path("test_qwen_tts.wav")
        output.write_bytes(audio)
        print(f"Saved to: {output}")

        # 播放
        subprocess.run(["afplay", str(output)])
    else:
        print("Failed to generate audio")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test())
