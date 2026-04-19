# Qwen3-TTS 快速响应示例

使用 Qwen3-TTS-12Hz-0.6B-Base-4bit 模型实现最小延迟的 LLM+TTS 流水线。

## 特点

- **LLM 流式**: 逐 token 返回，边生成边处理
- **TTS 流式输出**: 使用 MLX Audio 的流式功能
- **句子级并行**: LLM 生成一句 → TTS 立即转换一句
- **持久化 TTS 进程**: 避免重复加载模型
- **尽早播放**: 第一个音频块生成后立即开始播放
- **uv 运行**: 使用 uv 管理依赖，无需创建虚拟环境

## 架构

```
用户输入 → LLM API (流式) → SentenceAccumulator → TTS (流式) → 音频块队列 → 浏览器播放
              ↓                    ↓                        ↓
         token 级流式         句子边界检测              按句子转换
                                                   (持久化进程)
```

### 时间流程

1. **LLM 流式阶段**: 逐 token 返回内容
2. **句子检测**: `SentenceAccumulator` 检测完整句子
3. **并行 TTS**: 每检测到完整句子 → 立即送入 TTS 进程转换
4. **流式音频**: TTS 输出音频块 → 通过 WebSocket 发送
5. **前端播放**: 浏览器使用 Web Audio API 调度，无缝连续播放

## 文件说明

- `qwen3_tts_fast_server.py` - WebSocket 服务器
- `qwen3-tts-fast.html` - 前端界面
- `start-qwen3-tts-fast.sh` - 启动脚本

## 依赖要求

### 系统要求
- macOS (Apple Silicon) - MLX Audio 需要
- Python 3.10+
- uv (Python 包管理器)

### LLM API
需要运行兼容 OpenAI API 的 LLM 服务，例如：
- Qwen MLX: `http://127.0.0.1:12345/v1`
- Ollama: `http://localhost:11434/v1`

## 安装

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装 MLX Audio

运行自动安装脚本：

```bash
./install_mlx_audio.sh
```

或手动安装：

```bash
# 创建独立环境
uv venv mlx_audio_env
source mlx_audio_env/bin/activate

# 安装 MLX Audio
uv pip install mlx-audio

# 测试安装
python test_mlx_audio.py
```

### 3. 模型下载

模型会在首次运行时自动下载到 `~/.cache/huggingface/hub/`。

预下载（可选）：

```bash
pip install huggingface-cli
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit
```

## 使用

### 启动服务器

```bash
./start-qwen3-tts-fast.sh
```

或手动运行：

```bash
# 设置 LLM API
export LLM_API_BASE="http://127.0.0.1:12345/v1"
export LLM_API_KEY="your-key"
export LLM_MODEL="Qwen3.5-2B-MLX-8bit"

# 运行服务器
uv run --with websockets --with requests --with numpy python qwen3_tts_fast_server.py
```

### 打开前端

在浏览器中打开 `qwen3-tts-fast.html`

### 使用界面

1. 等待 WebSocket 连接成功（状态显示"已连接"）
2. 在输入框中输入问题
3. 点击"发送"按钮
4. 观察时间统计和音频播放

## 配置选项

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_BASE` | `http://127.0.0.1:12345/v1` | LLM API 地址 |
| `LLM_API_KEY` | `1234` | LLM API 密钥 |
| `LLM_MODEL` | `Qwen3.5-2B-MLX-8bit` | LLM 模型名称 |

### 前端消息协议

| 消息类型 | 方向 | 说明 |
|---------|------|------|
| `generate` | 客户端→服务器 | 发送问题 |
| `start` | 服务器→客户端 | 开始处理 |
| `llm_delta` | 服务器→客户端 | LLM 流式文本更新 |
| `first_char` | 服务器→客户端 | 首字符时间 |
| `llm_done` | 服务器→客户端 | LLM 完成 |
| `first_audio` | 服务器→客户端 | 首个音频块时间 |
| `audio_chunk` | 服务器→客户端 | 音频数据 (base64) |
| `done` | 服务器→客户端 | 全部完成 |
| `ping/pong` | 双向 | 心跳保活 |

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_BASE` | `http://127.0.0.1:12345/v1` | LLM API 地址 |
| `LLM_API_KEY` | `1234` | LLM API 密钥 |
| `LLM_MODEL` | `Qwen3.5-2B-MLX-8bit` | LLM 模型名称 |

### 服务器配置 (qwen3_tts_fast_server.py)

```python
# TTS 模型
QWEN3_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"

# 句子结束符
SENTENCE_ENDINGS = ('。', '！', '？', '.', '!', '?', '…', '\n')

# LLM 流式参数
MAX_TOKENS = 1024
TEMPERATURE = 0.5
```

## 性能指标

在 M1/M2 Mac 上的预期性能：

| 指标 | 预期值 |
|------|--------|
| LLM 首字符延迟 | 0.3-1s |
| 首句子完成 | 0.5-2s |
| TTS 首音频延迟 | 0.3-0.8s |
| **总首音频延迟** | **0.8-2.8s** |
| 音频生成速度 | 0.8-1.5x 实时 |

### 关键优化点

1. **LLM 流式**: 首字符快速返回，无需等待完整回答
2. **句子级流水线**: 完成一个句子立即开始 TTS，无需等待 LLM 全部完成
3. **持久化 TTS 进程**: 模型只加载一次，后续请求无冷启动
4. **WebSocket 双向通信**: 服务端主动推送音频块，减少轮询开销

## 工作原理

### 服务器端

#### 核心组件

1. **LLM 流式调用 (`call_llm_stream`)**
   - 使用 `requests` 库的 `stream=True` 参数
   - 在线程池中运行，避免阻塞事件循环
   - 逐 token yield 内容

2. **句子检测器 (`SentenceAccumulator`)**
   - 缓冲流式文本
   - 检测句子结束符（。！？.!?）
   - 完整句子立即 yield

3. **持久化 TTS 进程**
   - 服务器启动时创建独立 Python 子进程
   - MLX Audio 模型常驻内存
   - 通过 stdin/stdout 通信（JSON 行协议）

4. **WebSocket 处理器**
   - 接收客户端问题
   - 协调 LLM 流式输出和 TTS 并行处理
   - 实时推送音频块

#### 数据流

```
WebSocket 收到问题
  ↓
启动 LLM 流式调用
  ↓
SentenceAccumulator 检测句子
  ↓ (每完整句子)
发送到 TTS 进程 (stdin)
  ↓
从 TTS 进程读取音频块 (stdout)
  ↓
通过 WebSocket 推送到客户端
```

### 客户端

1. **WebSocket 通信**
   - 接收多种消息类型：`llm_delta`, `first_audio`, `audio_chunk`, `done`
   - 实时显示 LLM 生成文本

2. **音频队列调度**
   - 使用 Web Audio API 的 `AudioContext`
   - 预计算每个音频块的开始时间
   - 无缝连续播放

3. **可视化**
   - 实时显示各项延迟指标
   - 音频波形可视化
   - 详细的日志面板

## 与传统方案对比

| 特性 | 本方案 (流式 LLM + TTS) | 传统非流式 |
|------|------------------------|-----------|
| LLM 响应 | 逐 token 返回 | 等待完整回答 |
| TTS 启动时机 | 检测到完整句子后 | LLM 完全结束后 |
| 首音频延迟 | **0.8-2.8s** | 3-8s |
| 并行度 | LLM 和 TTS 重叠执行 | 串行执行 |
| 复杂度 | 中等 (句子检测器) | 简单 |
| 适用场景 | 问答、对话、长内容生成 | 短问答 |

## 代码结构

```
qwen3_tts_fast_server.py
├── SentenceAccumulator     # 句子边界检测
├── call_llm_stream()        # 流式 LLM 调用
├── create_tts_script()      # 创建 TTS 工作进程脚本
├── start_tts_process()      # 启动持久化 TTS 进程
├── stream_tts()             # 向 TTS 进程发送文本
└── handle_connection()      # WebSocket 请求处理
```

## 会议上下文集成

服务器支持从 `captions.db` 加载会议上下文，使 LLM 能够基于之前的对话内容回答问题。

## 故障排查

### TTS 模型未加载
```
错误信息: [TTS] Error: ...
解决方案: 确保模型路径正确，首次运行会自动下载到 ~/.cache/huggingface/hub/
检查: python -c "from mlx_audio.tts.utils import load_model; load_model('mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit')"
```

### WebSocket 连接失败
```
错误信息: WebSocket connection failed
解决方案:
  1. 检查服务器是否启动: ps aux | grep qwen3_tts_fast_server
  2. 检查端口是否被占用: lsof -i :8770
  3. 查看服务器日志
```

### 音频无法播放
```
错误信息: AudioContext was not allowed to start
解决方案:
  1. 点击页面任意位置激活 AudioContext
  2. 确保浏览器支持 Web Audio API
  3. 检查浏览器控制台是否有错误
```

### LLM API 超时
```
错误信息: [LLM] Error: ...
解决方案:
  1. 检查 LLM 服务是否运行: curl http://127.0.0.1:12345/v1/models
  2. 检查环境变量: echo $LLM_API_BASE
  3. 检查模型名称是否正确
```

### TTS 进程卡死
```
现象: 首音频延迟很高，没有音频块输出
解决方案:
  1. 重启服务器
  2. 检查 TTS Python 路径是否正确
  3. 查看是否有子进程残留: ps aux | grep python
```

## 进一步优化方向

### 1. 更激进的流式策略
当前在句子级别进行 TTS，可以考虑：
- 子句级别（逗号分隔）
- 固定 token 数量（如每 10 个 token）

### 2. 音频预加载
在 LLM 生成开始时预先生成开场白音频

### 3. 多模型支持
- 支持不同的 TTS 模型切换
- 支持不同的音色配置

### 4. 缓存机制
对常见问题的回答进行缓存

### 5. 连接复用
支持多个客户端共享同一个 TTS 进程

## 开发者指南

### 调试日志
服务器使用 Python `logging` 模块，可以设置环境变量调整日志级别：
```bash
LOG_LEVEL=DEBUG uv run python qwen3_tts_fast_server.py
```

### 性能分析
前端会显示详细的时序指标：
- 首字符延迟 (First Char Time)
- LLM 总时间 (LLM Time)
- 首音频延迟 (First Audio Time)
- 首播放延迟 (First Play Time)
- 总处理时间 (Total Time)
- 音频块数量
- 句子数量
- 字符数量
- 音频总长度

## 许可

本示例代码遵循项目主许可证。
