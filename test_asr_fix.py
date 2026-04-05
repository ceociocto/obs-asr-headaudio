#!/usr/bin/env python3
"""Test loading ASR model with quantization fix"""
import sys
import mlx.nn as nn
import mlx.core as mx

# Save original quantize function
_original_quantize = nn.quantize

def patched_quantize(model, bits, group_size=64, **kwargs):
    """Quantize only the text decoder, skip audio tower."""
    # Only quantize the 'model' part (text decoder), not 'audio_tower'
    if hasattr(model, 'model'):
        print(f"Applying {bits}-bit quantization to text decoder only")
        _original_quantize(model.model, bits=bits, group_size=group_size, **kwargs)
    else:
        # Fallback to original behavior
        _original_quantize(model, bits=bits, group_size=group_size, **kwargs)

# Apply the monkey patch
nn.quantize = patched_quantize
print("Applied quantization patch")

# Now test loading
from mlx_qwen3_asr import load_model

model_path = "/Volumes/sn7100/jerry/.cache/huggingface/hub/Qwen3-ASR-1.7B-MLX-4bit"
print(f"Loading model from: {model_path}")

try:
    model, config = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
