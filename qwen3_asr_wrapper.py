#!/usr/bin/env python3
"""
Qwen3-ASR Wrapper - 本地语音识别
使用 mlx-qwen3-asr 在 Apple Silicon 上运行
"""

import asyncio
import os
import sys
import logging
import concurrent.futures
from pathlib import Path
import subprocess
import tempfile
import wave

logger = logging.getLogger(__name__)

# 修复部分量化模型的加载问题
# 4bit模型只量化了文本解码器，音频塔没有被量化
try:
    import mlx.nn as nn

    _original_quantize = nn.quantize

    def patched_quantize(model, bits, group_size=64, **kwargs):
        """只量化文本解码器，跳过音频塔"""
        if hasattr(model, 'model'):
            logger.info(f"Applying {bits}-bit quantization to text decoder only")
            _original_quantize(model.model, bits=bits, group_size=group_size, **kwargs)
        else:
            _original_quantize(model, bits=bits, group_size=group_size, **kwargs)

    nn.quantize = patched_quantize
    logger.info("Applied quantization patch for partial quantized models")
except ImportError:
    pass

# 模型路径
ASR_MODEL_PATH = "/Volumes/sn7100/jerry/.cache/huggingface/hub/Qwen3-ASR-1.7B-MLX-4bit"

# 检查 mlx-qwen3-asr 可用性
try:
    from mlx_qwen3_asr import transcribe
    MLX_ASR_AVAILABLE = True
    logger.info("mlx-qwen3-asr available")
except ImportError:
    MLX_ASR_AVAILABLE = False
    logger.warning("mlx-qwen3-asr not available. Install with: pip install mlx-qwen3-asr")


class Qwen3ASR:
    def __init__(self, model_path=None):
        """初始化 Qwen3-ASR 模型"""
        self.model_path = model_path or ASR_MODEL_PATH
        self.model_loaded = False

        if not MLX_ASR_AVAILABLE:
            logger.warning("mlx-qwen3-asr not installed")
            return

        # 检查模型文件
        if os.path.exists(self.model_path):
            logger.info(f"ASR model found: {self.model_path}")
        else:
            logger.warning(f"ASR model not found: {self.model_path}")
            return

    def ensure_model_loaded(self):
        """确保模型已加载"""
        if not MLX_ASR_AVAILABLE:
            return False

        # 模型在首次使用时自动加载
        return True

    def transcribe_file(self, audio_path):
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            识别到的文本
        """
        if not self.ensure_model_loaded():
            logger.warning("ASR not available")
            return None

        try:
            logger.info(f"Transcribing: {audio_path}")

            # 使用 mlx-qwen3-asr
            result = transcribe(
                audio_path,
                model=self.model_path
            )

            # 提取文本内容
            text = result.text if hasattr(result, 'text') else str(result)
            logger.info(f"✓ Recognized: '{text}'")
            return text

        except Exception as e:
            logger.error(f"ASR error: {e}")
            return None

    def transcribe_audio_data(self, audio_bytes, sample_rate=24000):
        """
        转录音频数据（bytes）

        Args:
            audio_bytes: 音频数据
            sample_rate: 采样率

        Returns:
            识别到的文本
        """
        if not audio_bytes:
            return None

        # 创建临时 WAV 文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            try:
                # 写入 WAV 文件
                with wave.open(tmp_file.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_bytes)

                # 转录
                result = self.transcribe_file(tmp_file.name)
                return result

            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass


# 全局 ASR 实例
_asr_instance = None

def get_asr():
    """获取 ASR 实例（单例）"""
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = Qwen3ASR()
    return _asr_instance


# 测试
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    asr = Qwen3ASR()
    logger.info(f"ASR available: {MLX_ASR_AVAILABLE}")
    logger.info(f"ASR model exists: {os.path.exists(ASR_MODEL_PATH)}")

    # 如果有测试音频文件，可以测试
    # test_audio = "/path/to/test.wav"
    # result = asr.transcribe_file(test_audio)
    # print(f"Result: {result}")
