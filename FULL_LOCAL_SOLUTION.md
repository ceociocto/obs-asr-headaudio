# Full-Local 离线语音助手方案

## 概述

基于 Qwen3-ASR + Qwen3-TTS 的完全离线语音助手，在 Apple Silicon 上本地运行，无需任何云端 API。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         浏览器前端                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ TalkingHead  │  │  HeadAudio   │  │   WebSocket Client   │  │
│  │  虚拟头像     │  │  口型同步     │  │    (ws://localhost)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ↕ WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                      Python 后端服务                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              WebSocket Server (localhost:8765)          │   │
│  │  - 接收客户端音频                                         │   │
│  │  - 调用 ASR 识别                                          │   │
│  │  - 调用 TTS 合成                                          │   │
│  │  - 返回识别结果和音频                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│           ┌──────────────────┴──────────────────┐              │
│           ▼                                      ▼              │
│  ┌──────────────────┐              ┌──────────────────┐       │
│  │  Qwen3-ASR       │              │  Qwen3-TTS       │       │
│  │  语音识别         │              │  语音合成         │       │
│  │  (MLX 加速)      │              │  (MLX 加速)      │       │
│  └──────────────────┘              └──────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 前端 (full-local.html)

- **TalkingHead**: 虚拟 3D 头像渲染
- **HeadAudio**: 音频驱动的口型同步
- **WebSocket**: 与后端通信

### 2. 后端服务 (voice_server_qwen3.py)

WebSocket 服务器，处理：
- 客户端连接管理
- 语音识别请求（通过 Qwen3-ASR）
- 语音合成请求（通过 Qwen3-TTS）
- 音频数据编码传输

### 3. ASR 模块 (qwen3_asr_wrapper.py)

Qwen3-ASR 封装：
- 模型加载管理
- 音频文件转录
- 支持部分量化模型

### 4. TTS 模块 (qwen3_tts_wrapper.py)

Qwen3-TTS 封装：
- 模型加载管理
- 文本转语音合成
- 多种声音支持

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端框架 | HTML5 + Vanilla JS |
| 3D 渲染 | Three.js |
| 虚拟头像 | @met4citizen/TalkingHead |
| 口型同步 | HeadAudio |
| 后端服务 | Python + asyncio |
| WebSocket | websockets |
| ASR | mlx-qwen3-asr (MLX) |
| TTS | mlx-audio/tts (MLX) |

## 模型要求

### Qwen3-ASR 模型
- 路径: `~/.cache/huggingface/hub/Qwen3-ASR-1.7B-MLX-4bit`
- 大小: ~1.7GB (4-bit 量化)
- 下载: https://huggingface.co/Qwen/Qwen3-ASR-1.7B-MLX-4bit

### Qwen3-TTS 模型
- 路径: `./qwen3-tts-apple-silicon/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- 大小: ~1.7GB (8-bit 量化)
- 下载: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit

## 安装步骤

1. **安装 MLX 相关依赖**
```bash
pip install mlx mlx-audio mlx-qwen3-asr websockets
```

2. **克隆 qwen3-tts-apple-silicon**
```bash
git clone https://github.com/Xenova/qwen3-tts-apple-silicon.git
```

3. **下载模型**
   - 从 Hugging Face 下载上述模型到指定路径

## 使用方法

1. **启动后端服务**
```bash
python voice_server_qwen3.py
```

2. **打开前端页面**
```bash
open full-local.html
# 或在浏览器中打开文件
```

3. **点击页面激活**，之后可以：
   - 按住麦克风按钮说话（语音识别）
   - 在输入框输入文字
   - 点击播放按钮（语音合成）

## 工作流程

### 语音输入流程
1. 用户点击麦克风按钮
2. 客户端发送开始录音消息
3. 服务器启动麦克风录制
4. 用户松开按钮
5. 服务器将录音发送给 Qwen3-ASR
6. 识别结果返回并显示

### 语音输出流程
1. 用户输入文字或语音识别完成
2. 客户端发送 TTS 请求
3. 服务器调用 Qwen3-TTS 生成音频
4. 音频编码为 base64 返回
5. 客户端解码并播放
6. HeadAudio 同步驱动口型

## 量化模型修复

由于 4-bit ASR 模型只量化了文本解码器，需要修复 `quantize` 函数：

```python
def patched_quantize(model, bits, group_size=64, **kwargs):
    """只量化文本解码器，跳过音频塔"""
    if hasattr(model, 'model'):
        _original_quantize(model.model, bits=bits, group_size=group_size, **kwargs)
    else:
        _original_quantize(model, bits=bits, group_size=group_size, **kwargs)
```

## 特性

- ✅ 100% 离线运行
- ✅ 无需云端 API
- ✅ Apple Silicon 加速
- ✅ 实时语音识别
- ✅ 高质量语音合成
- ✅ 3D 虚拟头像
- ✅ 口型同步
- ✅ WebSocket 通信

## 文件列表

- `full-local.html` - 前端页面
- `voice_server_qwen3.py` - 后端服务
- `qwen3_asr_wrapper.py` - ASR 封装
- `qwen3_tts_wrapper.py` - TTS 封装
- `test_asr_fix.py` - 量化修复测试
