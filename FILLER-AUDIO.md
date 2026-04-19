# Filler Audio - 快速首音频响应

## 概述

填充音频（Filler Audio）是一种 UX 优化技术，通过在 LLM 生成回答之前播放简短的预录制音频（如 "Let me see..."），显著降低用户感知的响应延迟。

## 延迟对比

| 方案 | 首音频延迟 | 说明 |
|------|-----------|------|
| 原方案 | 2-4秒 | 需等待 LLM 首句 + TTS 生成 |
| **填充音频方案** | **<100ms** | 立即播放预录制音频 |

## 工作原理

```
用户提问
    │
    ├─> [立即] 播放 "Let me see..." (预录制, <100ms)
    │
    └─> [并行] LLM 生成回答
            │
            └─> TTS 生成句子音频
                    │
                    └─> 填充音频播放完后，无缝切换到真实回答
```

## 使用方法

### 1. 生成填充音频文件

```bash
python generate_filler_audio.py
```

这会在 `filler_audio/` 目录下生成以下文件：
- `let_me_see.wav` - "Let me see."
- `let_me_check.wav` - "Let me check."
- `thinking.wav` - "Hmm."
- `one_moment.wav` - "One moment."
- `give_me_sec.wav` - "Just a second."
- `metadata.json` - 元数据信息

### 2. 启动服务器

```bash
python qwen3_tts_fast_server.py
```

服务器启动时会自动加载填充音频：
```
[Filler] Loaded let_me_see
[Filler] Loaded let_me_check
...
✓ Filler audio: 5 loaded
```

### 3. 测试

打开 `qwen3-tts-filler-demo.html`，勾选 "Enable filler audio"，然后提问测试。

## 填充音频选择策略

服务器根据问题类型自动选择合适的填充音频：

| 问题类型 | 关键词 | 填充音频 |
|---------|-------|---------|
| 解释类 | what, tell me, explain, describe, how | "Let me see..." |
| 查找类 | find, search, look, check, where | "Let me check..." |
| 计算类 | calculate, compute, math, count | "Hmm..." |
| 其他 | - | 随机选择 |

## 自定义填充音频

### 修改语料

编辑 `generate_filler_audio.py` 中的 `FILLER_PHRASES`：

```python
FILLER_PHRASES = {
    "let_me_see": "让我看看",        # 中文
    "let_me_check": "Let me check.",
    "thinking": "嗯...",
    # 添加更多...
}
```

### 使用自己的音频文件

直接将 WAV 文件放入 `filler_audio/` 目录，命名为 `{key}.wav`：

```
filler_audio/
├── let_me_see.wav
├── let_me_check.wav
├── custom_response.wav  # 自定义
└── ...
```

然后在 `select_filler_audio()` 函数中添加匹配逻辑。

## 高级配置

### 禁用填充音频

在客户端发送请求时设置：

```javascript
ws.send(JSON.stringify({
    type: 'generate',
    question: "Your question",
    use_filler: false  // 禁用填充音频
}));
```

### 服务器端配置

编辑 `qwen3_tts_fast_server.py`：

```python
# 设置默认填充音频
FILLER_AUDIO = {
    "default": "",  # 设置默认填充音频 base64
}
```

## 性能建议

1. **填充音频时长**: 建议控制在 0.3-0.8 秒
   - 太短：覆盖不了 LLM 延迟
   - 太长：用户感觉啰嗦

2. **音频质量**: 使用 24kHz 采样率，与主 TTS 保持一致

3. **多样性**: 准备 3-5 个不同的填充音频，避免单调

## 延迟测量

服务器日志会显示：
```
[Filler] Sent 'let_me_see' in 0.045s  # 填充音频发送延迟
[TIME] Total: 3.21s, Audio: 2.45s    # 总时间
```

客户端可以看到：
- `First Audio`: 首个音频块的延迟（填充音频会标注 FILLER）
- `LLM Time`: LLM 生成时间
- `Total`: 总响应时间
