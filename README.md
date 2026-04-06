# HeadAudio

---

### Quick Start / 快速开始

Qwen3-ASR 离线语音识别 + Chrome TTS + 3D 虚拟头像口型同步，全部在本地运行。

#### 前置条件

- macOS (Apple Silicon)
- [uv](https://docs.astral.sh/uv/) Python 包管理器
- OBS Studio (macOS 版)

#### 1. 安装依赖

```bash
uv sync
```

#### 2. 下载 Qwen3-ASR 模型

ASR 后端使用 [Qwen3-ASR-1.7B-MLX-4bit](https://huggingface.co/) 模型（Apple Silicon MLX 4-bit 量化），需提前下载到本地：

```bash
# 安装 huggingface-cli（如未安装）
uv pip install huggingface_hub

# 下载模型（约 1GB）
huggingface-cli download Qwen/Qwen3-ASR-1.7B-MLX-4bit --local-dir ~/.cache/huggingface/hub/Qwen3-ASR-1.7B-MLX-4bit
```

> 模型默认路径配置在 `qwen3_asr_wrapper.py` 的 `ASR_MODEL_PATH` 变量中，可按需修改。

#### 3. 启动 OBS (释放浏览器媒体权限)

OBS 需要以特殊参数启动，解除 Chrome 内核对自动播放和麦克风的限制：

```bash
chmod +x start_obs.sh
./start_obs.sh
```

> 如果 OBS 已经在运行，先完全退出 (`Command+Q`) 再执行脚本。

脚本通过 `--use-fake-ui-for-media-stream` 等参数，让 OBS 内嵌浏览器面板跳过媒体权限弹窗和自动播放限制，TTS 音频可无需用户点击即播放。

#### 3.1 OBS 音频配置（避免回声）

在 OBS 中需正确配置音频监听模式，否则 TTS 播放的音频会通过麦克风回路产生回声。请按以下截图配置：

<img src="assets/ScreenShot_obs_setting_avoid_echo.png" width="600" alt="OBS 音频配置 - 避免回声"/>

**关键设置：**
- 在 OBS 的 `高级音频属性` 中，将浏览器源的 **音频监听** 模式设置为仅监听（Monitor Only），不要选择「监听并输出（Monitor and Output）」
- 确保麦克风源的音频监听设为关闭（None），避免麦克风采集的声音再次从扬声器输出形成回路
- 如使用耳机，可适当放宽上述限制，但建议保持一致以获得最佳效果

#### 3.2 macOS 音频路由配置（BlackHole）

若需将系统音频（如 TTS 输出）同时路由到 OBS 和扬声器，推荐安装 [BlackHole](https://existential.audio/blackhole/) 虚拟音频驱动，配合 macOS 自带的「音频 MIDI 设置」创建多输出设备。

**1) 安装 BlackHole**

```bash
# 推荐使用 Homebrew 安装
brew install blackhole-16ch
```

**2) 创建多输出设备（Multi-Output Device）**

打开「音频 MIDI 设置」（Audio MIDI Setup），点击左下角 `+` → 「创建多输出设备」，勾选：
- 扬声器 / 耳机（物理输出）
- BlackHole 16ch

将此多输出设备重命名为 `Multi-Output`，然后将系统默认输出设为该设备。

**3) 配置麦克风聚合（Aggregate Device）**

<img src="assets/ScreenShot_midi_mic.png" width="600" alt="macOS 音频 MIDI 设置 - 麦克风聚合设备"/>

创建聚合设备（Aggregate Device），勾选：
- Mac 内置麦克风（或外接麦克风）
- BlackHole 16ch

这样 OBS 可以从该聚合设备采集到麦克风输入，同时 BlackHole 通道用于内部音频路由。

**4) 系统音频输出配置**

<img src="assets/ScreenShot_midi_audio.png" width="600" alt="macOS 音频 MIDI 设置 - 多输出设备"/>

确保多输出设备中主设备（Master Device）设为物理扬声器/耳机，BlackHole 作为从设备。这样系统声音既能从扬声器听到，也能被 OBS 通过 BlackHole 采集到。

> **提示：** 配置完成后，在 OBS 的音频设置中将麦克风输入选为刚创建的聚合设备，即可实现麦克风 + 系统音频的同时采集。

#### 4. 启动后端服务

```bash
uv run python voice-asr-server.py
```

服务器监听 `ws://localhost:8765`，提供 Qwen3-ASR 离线语音识别。

#### 5. 启动前端 (HTTP)

```bash
uv run python -m http.server 8000
```

浏览器打开 `http://localhost:8000/voice-asr.html`。

> 必须通过 HTTP 服务访问，直接以 `file://` 打开会导致 WebSocket 和模块加载失败。

#### 演示流程

```
终端 1: uv run python voice-asr-server.py   # ASR 识别服务
终端 2: ./start_obs.sh                       # 启动 OBS (可选)
终端 3: uv run python -m http.server 8000    # 前端 HTTP 服务
浏览器:  http://localhost:8000/voice-asr.html
```

页面加载后，语音检测默认自动开启，对麦克风说话即可触发 ASR 识别 → Chrome TTS 朗读 → HeadAudio 驱动口型同步。点击麦克风按钮可切换检测开关状态。

#### 技术架构

```
浏览器 (前端)                         Python (后端)
┌─────────────┐    WebSocket    ┌──────────────────┐
│ TalkingHead │◄──────────────►│ voice-asr-server │
│  3D 头像     │   JSON/Base64  │   Qwen3-ASR      │
│ HeadAudio   │                 │   (MLX GPU 加速)  │
│  口型同步    │                 └──────────────────┘
│ Chrome TTS  │
│  语音合成    │
└─────────────┘
```

- **HeadAudio**: 浏览器 AudioWorklet，实时分析音频 MFCC 特征 → 马氏距离分类 → 输出 Oculus viseme 口型参数
- **TalkingHead**: Three.js 3D 头像渲染，接收 viseme 值驱动 blend shape
- **Qwen3-ASR**: Apple Silicon MLX 加速的离线语音识别，浏览器录音经 WebSocket 发送至后端
- **Chrome TTS**: 浏览器内置语音合成，将识别结果朗读出来

---

### Introduction

HeadAudio is an audio worklet node/processor for audio-driven,
real-time viseme detection and lip-sync in browsers. It uses
MFCC feature vectors and Gaussian prototypes with
a Mahalanobis-distance classifier. As output, it generates
Oculus viseme blend-shape values in real time and can be
integrated into an existing 3D animation loop.

- **Pros**: Audio-driven lip-sync works with any audio stream
or TTS output without requiring text transcripts or timestamps.
It is fast, fully in-browser, and requires no server.

- **Cons**: Voice activity detection (VAD) and prediction accuracy
are far from optimal, especially when the signal-to-noise
ratio (SNR) is low. In general, the audio-driven approach is
less accurate and computationally more demanding than
[TalkingHead](https://github.com/met4citizen/TalkingHead)'s
text-driven approach.

The solution is fully compatible with the
[TalkingHead](https://github.com/met4citizen/TalkingHead).
It doesn't have any external dependencies, and it is
MIT licensed.

[HeadTTS](https://github.com/met4citizen/HeadTTS),
[webpack](https://github.com/webpack/webpack),
and [jest](https://jestjs.io) were used during
development, training, and testing.

The implementation has been tested with the latest
versions of Chrome, Edge, Firefox, and Safari desktop
browsers, as well as on iPad/iPhone.

> [!IMPORTANT]
> The model's accuracy will hopefully improve over time.
However, since all audio processing occurs fully in-browser
and in real time, it will never be perfect and may not be
suitable for all use cases. Some precision will always need
to be sacrificed to stay within the real-time processing budget.

---

### Demo / Test App

App | Description
--- | ---
<span style="display: block; min-width:400px">[<img src="images/openai.jpg" width="400"/>](https://met4citizen.github.io/HeadAudio/openai.html)</span> | A demo web app using HeadAudio, [TalkingHead](https://github.com/met4citizen/TalkingHead), and [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) (WebRTC). It supports speech-to-speech, moods, hand gestures, and facial expressions through function calling. \[[Run](https://met4citizen.github.io/HeadAudio/openai.html)] \[[Code](https://github.com/met4citizen/HeadAudio/blob/main/openai.html)]<br/><br/>Note: The app uses OpenAI's [gpt-realtime-mini](https://platform.openai.com/docs/models/gpt-realtime-mini) model and requires an OpenAI API key. The “mini” model is a cost-effective version of GPT Realtime, but still relatively expensive for extended use.
[<img src="images/tester.jpg" width="400"/>](https://met4citizen.github.io/HeadAudio/tester.html) | A test app for HeadAudio that lets you experiment with audio-stream processing and various parameters using [HeadTTS](https://met4citizen.github.io/HeadTTS) (in-browser neural text-to-speech engine), your own audio file(s), or microphone input. \[[Run](https://met4citizen.github.io/HeadAudio/tester.html)] \[[Code](https://github.com/met4citizen/HeadAudio/blob/main/tester.html)]

---

### Using the HeadAudio Worklet Node/Processor

The steps needed to setup and use HeadAudio:

1. Import the Audio Worklet Node `HeadAudio` from
`"./modules/headaudio.mjs"`. Alternatively, use
the minified version `"./dist/headaudio.min.mjs"`
or a CDN build.

2. Register the Audio Worklet Processor from
`"./modules/headworklet.mjs"`. Alternatively, use
the minified version `"./dist/headworklet.min.mjs"`
or a CDN build.

3. Create a new `HeadAudio` instance.

4. Load a pre-trained viseme model containing Gaussian
prototypes, e.g., `"./dist/model-en-mixed.bin"`.

5. Connect your speech audio node to the HeadAudio node.
The node node has a single mono input and does not output
any audio.

6. Optional: To compensate for processing latency (50–100 ms),
add delay to your speech-audio path using the browser's standard
[DelayNode](https://developer.mozilla.org/en-US/docs/Web/API/DelayNode).

7. Assing `onvalue` callback function `(key, value)`
that updates your avatar's blend shape `key` (Oculus viseme name, e.g, "viseme_aa")
to the given `value` in the range \[0,1].

8. Call the node's `update` method inside your 3D animation
loop, passing the delta time (in milliseconds).

9. Optional: Set up any additional user event handlers as needed.

Here is a simplified code example using the above steps with a
[TalkingHead](https://github.com/met4citizen/TalkingHead)
class instance `head`:

```javascript
// 1. Import
import { TalkingHead } from "talkinghead";
import { HeadAudioNode } from "./modules/headaudio.mjs";

// 2. Register processor
const head = new TalkingHead( /* Your normal parameters */ );
await head.audioCtx.audioWorklet.addModule("./modules/headworklet.mjs");

// 3. Create new HeadAudio node
const headaudio = new HeadAudio(head.audioCtx, {
  processorOptions: { },
  parameterData: {
    vadGateActiveDb: -40,
    vadGateInactiveDb: -60
  }
});

// 4. Load a pre-trained model
await headaudio.loadModel("./dist/model-en-mixed.mjs");

// 5. Connect TalkingHead's speech gain node to HeadAudio node
head.audioSpeechGainNode.connect(headaudio);

// 6. OPTIONAL: Add some delay between gain and reverb nodes
const delayNode = new DelayNode( head.audioCtx, { delayTime: 0.1 });
head.audioSpeechGainNode.disconnect(head.audioReverbNode);
head.audioSpeechGainNode.connect(delayNode);
delayNode.connect(head.audioReverbNode);

// 7. Register callback function to set blend shape values
headaudio.onvalue = (key,value) => {
  Object.assign( head.mtAvatar[ key ],{ newvalue: value, needsUpdate: true });
};

// 8. Link node's `update` method to TalkingHead's animation loop
head.opt.update = headaudio.update.bind(headaudio);

// 9. OPTIONAL: Take eye contact and make a hand gesture when new sentence starts
let lastEnded = 0;
headaudio.onended = () => {
  lastEnded = Date.now();
};

headaudio.onstarted = () => {
  const duration = Date.now() - lastEnded;
  if ( duration > 150 ) { // New sentence, if 150 ms pause (adjust, if needed)
    head.lookAtCamera(500);
    head.speakWithHands();
  }
};
```

See the test app
[source code](https://github.com/met4citizen/HeadAudio/blob/main/tester.html)
for more details.

The supported `processerOptions` are:

Option | Description | Default
--- | --- | ---
`frameEventsEnabled` | If `true`, sends `frame` user-event objects containing a downsampled samples array and timestamp: `{ event: 'frame', frame, t }`. NOTE: Mainly for testing. | `false`
`vadEventsEnabled` | If `true`, sends `vad` user-event objects with status counters and current log-energy in decibels:  `{ event: 'vad', active, inactive, db, t }`. NOTE: Mainly for testing. | `false`
`featureEventsEnabled` | If `true`, send `feature` user-event objects with the normalized feature vector, log-energy, timestamp, and duration: `{ event: 'feature', vector, le, t, d }`. NOTE: Mainly for testing. | `false`
`visemeEventsEnabled` | If `true`, sends `viseme` user-event objects containing extended viseme information, including the predicted viseme, feature vector, distance array, timestamp, and duration: `{ event: 'viseme', viseme, vector, distances, t, d }`. NOTE: Mainly for testing. | `false`

The supported `parameterData` are:

Parameter | Description | Default
--- | --- | ---
`vadMode` | `0` = Disabled, `1` = Gate. If disabled, processing relies only on silence prototypes (see `silMode`). Gate mode is a simple energy-based VAD suitable for low and stable noise floors with high SNR. | `1`
`vadGateActiveDb` | Decibel threshold above which audio is classified as active. | `-40`
`vadGateActiveMs` | Duration (ms) required before switching from inactive to active. | `10`
`vadGateInactiveDb` | Decibel threshold below which audio is classified as inactive. | `-50`
`vadGateInactiveMs` | Duration (ms) required before switching from active to inactive. | `10`
`silMode` | `0` = Disabled, `1` = Manual calibration, `2` = Auto (NOT IMPLEMENTED). If disabled, only trained SIL prototypes are used. In manual mode, the app must perform silence calibration. Auto mode is currently not implemented. | `1`
`silCalibrationWindowSec` | Silence-calibration window in seconds. | `3.0`
`silSensitivity` | Sensitivity to silence. | `1.2`
`speakerMeanHz` | Estimated speaker mean frequency in Hz [50–500]. Adjusting this gently stretches/compresses the Mel spacing and frequency range to better match the speaker’s vocal-tract resonances and harmonic structure. Typical values: adult male 100–130, adult female 200–250, child 300–400. EXPERIMENTAL | `150`

> [!TIP]
> All audio parameters can be changed in real-time,
e.g.: `headaudio.parameters.get("vadMode").value = 0;`

Supported `HeadAudio` class events:

Event | Description
--- | ---
`onvalue(key, value)` | Called when a viseme blend-shape value is updated. `key` is one of: 'viseme_aa', 'viseme_E', 'viseme_I', 'viseme_O', 'viseme_U', 'viseme_PP', 'viseme_SS', 'viseme_TH' 'viseme_DD', 'viseme_FF', 'viseme_kk', 'viseme_nn', 'viseme_RR', 'viseme_CH', 'viseme_sil'. `value` is in the range \[0,1].
`onstarted(data)` | Speech start event `{ event: "start", t }`.
`onended(data)` | Speech end event `{ event: "end", t }`.
`onframe(data)` |  Frame event `{ event: "frame", frame, t }`. Contains 32-bit float 16 kHz mono samples. Requires `frameEventEnabled` to be `true`.
`onvad(data)` |  VAD event `{ event: "vad", t, db, active, inactive }`. Requires `vadEventEnabled` to be `true`.
`onfeature(data)` | Feature event `{ event: "feature", vector, t, d }`. Requires `featureEventEnabled` to be `true`.
`onviseme(data)` | Viseme event `{ event: "viseme", viseme, t, d, vector, distances }`. Requires `visemeEventEnabled` to be `true`.
`oncalibrated(data)` | Calibration event `{ event: "calibrated", t, \[error] }`.
`onprocessorerror(event)` | Fired when an internal processor error occurs.

---

### Training

> [!IMPORTANT]
> You do NOT need to train your own model as a pre-trained
model is provided. However, if you want to train a custom model,
the process below describes how the existing model was created.

The lip-sync model `./models/model-en-mixed.bin` was trained with four 
[HeadTTS](https://github.com/met4citizen/HeadTTS) voices using
[Harvard Sentences](https://www.cs.columbia.edu/~hgs/audio/harvard.html)
as input text. HeadTTS is ideal for generating training data because
it can produce audio, phonemes, visemes, and highly accurate
phoneme-level timestamps.

1. Install and start HeadTTS text-to-speech REST service
locally (requires Node.js v20+):

```bash
git clone https://github.com/met4citizen/HeadTTS
cd HeadTTS
npm install
npm start
```

Note: Before using the HeadTTS server, download all the voices
that you will be using from
[onnx-community/Kokoro-82M-v1.0-ONNX-timestamped](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped)
to your HeadTTS `./voices` directory.

2. Generate training data (.wav and .json)

In a separate console window, install HeadAudio (if you haven't already),
then generate the training files from the text prompts (.txt):

```bash
git clone https://github.com/met4citizen/HeadAudio
cd HeadAudio
npm install
cd training
node precompile-headtts.mjs -i "./headtts/headtts-1.txt" -v "af_bella"
node precompile-headtts.mjs -i "./headtts/headtts-2.txt" -v "af_heart"
node precompile-headtts.mjs -i "./headtts/headtts-3.txt" -v "am_adam"
node precompile-headtts.mjs -i "./headtts/headtts-4.txt" -v "am_fenrir"
```

Each voice takes about 2 minutes to process and generates roughly
10 minutes of training audio (.wav) along with corresponding
phoneme/viseme timestamp data (.json).

For Mahalanobis classification with 12 features, you should aim
for at least 60–120 samples per phoneme. The process above will
generate more than enough training data.

3. Compile the Gaussian prototypes into a binary model

Once the .wav and .json files are ready, build the final model (.bin):

```bash
node compile.mjs -i "./headtts" -o "model-en-mixed.bin"
```

Compilation takes about 30 seconds, and the resulting
binary file will be approximately 14 kB.

For all console apps, use the `--help` option to view
available arguments.

---

### Technical Overview

The viseme-detection process uses Mel-Frequency Cepstral
Coefficient (MFCC) feature vectors, Gaussian prototypes, and a
[Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
classifier.

SSince only the relative ordering of distances matters,
the squared Mahalanobis distance $d_M^2$ is used:

$\displaystyle\qquad d_{M}^2 = (\vec{x} - \vec{\mu})^\top \Sigma^{-1} (\vec{x} - \vec{\mu})$

The real-time processing steps are as follows:

- **Audio input**: The audio worklet processor receives
real-time speech input on a dedicated audio thread.
The incoming audio typically consists of 128 mono float-32
samples at 44.1 kHz or 48 kHz.

- **Pre-emphasis and downsampling**:
A pre-emphasis filter is applied to emphasize high-frequency
components, after which the signal is downsampled to
16 kHz using a polyphase low-pass filter.

- **MFCC feature vector**: The downsampled audio is
processed in 512-sample frames with a 256-sample hop.
Each frame is converted into MFCC features through the
following steps:
(1) Apply a Hamming window,
(2) Compute the FFT,
(3) Calculate the power spectrum,
(4) Apply Mel filters,
(5) Perform a Discrete Cosine Transform (DCT).
The output is a normalized log-energy value and
a 12-coefficient MFCC feature vector.

- **Compression**:
MFCC features are compressed using `tanh` to improve robustness
across speakers and recording conditions.
If `feature` events are enabled, each feature vector is posted
to the main thread via `onfeature`.

- **Voice Activity Detection (VAD)**: A simple gate-based
VAD model determines active vs. inactive speech based on
energy thresholds.
If `vad` events are enabled, active/inactive status and dB value
are posted to the main thread via `onvad`.

- **Classification**: Each pre-trained Gaussian prototype contains
a mean vector $\vec{\mu}$ and an inverse covariance matrix
$\Sigma^{-1}$ for a given phoneme. The classifier computes the
Mahalanobis distance for each prototype and selects the viseme
with the lowest distance. The result is posted to the main thread.
If the `viseme` events are enabled, the viseme, MFCC vector,
and the distance array are sent via `onviseme`.

- **Lip animation**: Based on the detected viseme, lip movements
are calculated with easing and blending/cross-fading.
Real-time blend-shape values are delivered via the `onvalue`
callback, using the Oculus morph-target name and the computed value.

The common parameters are set in `./modules/parameters.mjs`:

Parameter | Description | Default
--- | --- | ---
AUDIO_SAMPLE_RATE | Sample rate used in downsampling and processing. | `16000`
AUDIO_DOWNSAMPLE_FILTER_N | Number of taps per phase. | `32`
AUDIO_DOWNSAMPLE_PHASE_N | Number of fractional phases. | `64`
AUDIO_PREEMPHASIS_ENABLED | Apply pre-emphasis to boost higher frequencies. | `true`
AUDIO_PREEMPHASIS_ALPHA | Pre-emphasis alpha. | `0.97`
MFCC_SAMPLES_N | Number of samples per feature vector. | `512`
MFCC_SAMPLES_HOP | Number of sample hops, If the same as MFCC_SAMPLES_N, no overlap. | `256`
MFCC_COEFF_N | The number of MFCC coefficients ignoring log energy c0 and possible deltas. | `12`
MFCC_MEL_BANDS_N | The number of Mel bands. | `40`
MFCC_LIFTER | Lifter parameter. | `22`
MFCC_DELTAS_ENABLED | Calculate and include MFCC deltas. | `false` 
MFCC_DELTA_DELTAS_ENABLED | Calculate and include MFCC delta-deltas. Requires that `MFCC_DELTAS_ENABLED` is `true`. | `false`
MFCC_COEFF_N_WITH_DELTAS | Derived constant. |
MFCC_COMPRESSION_ENABLED | If true, apply tanh compression. | `true`
MFCC_COMPRESSION_TANH_R | Tanh compression range. | `1.0`
MODEL_VISEMES_N | The number of visemes. | `15`
MODEL_VISEME_SIL | Silence viseme ID. | `14`

---

### Appendix A: Oculus visemes

The used Oculus viseme numbering, naming and phoneme mapping:

ID | Oculus viseme| Phonemes
--- | --- | ---
0 | "viseme_aa" | "aa" (open / low): `a`, `aː`, `ɑ`, `ɑː`, `ɐ`, `aɪ`, `aʊ`, `ä`.
1 | "viseme_E" | "E" (mid) + Central vowels: `ɛ`, `ɛː`, `e`, `eː`, `eɪ`, `œ`, `ɜ`, `ʌ`, `ə`, `ɚ`, `ɘ`.
2 | "viseme_I" | "I" (close front): `i`, `iː`, `ɪ`, `ɨ`, `y`, `yː`, `ʏ`.
3 | "viseme_O" | "O" (mid back): `o`, `oː`, `ɔ`, `ɔː`, `ɒ`, `ø`, `øː`.
4 | "viseme_U" | "U" (close back): `u`, `uː`, `ʊ`, `ɯ`, `ɯː`, `ɤ`.
5 | "viseme_PP" | Plosives / bilabials: `p`, `b`, `m`.
6 | "viseme_SS" | Fricatives / sibilants: `s`, `z`, `ʃ`, `ʒ`, `ɕ`, `ʑ`, `ç`, `ʝ`, `x`, `ɣ`, `h`,
7 | "viseme_TH" | Dentals: `θ`, `ð`.
8 | "viseme_DD" | Alveolar stops: `t`, `d`.
9 | "viseme_FF" | Labiodentals: `f`, `v`.
10 | "viseme_kk" | Velar stops: `k`, `g`, `q`, `ɢ`.
11 | "viseme_nn" | Nasals: `n`, `ŋ`, `ɲ`, `ɳ`, `m̩`.
12 | "viseme_RR" | Liquids / approximants: `ɹ`, `r`, `ɾ`, `ɽ`, `l`, `ɫ`, `j`, `w`.
13 | "viseme_CH" | Affricates: `tʃ`, `dʒ`, `ts`, `dz`.
14 | "viseme_sil" | Silence / pause markers: ``, `ˈ`, `ˌ`, `‖`, `\|`.

---

### Appendix B: Data Structures

Models such as `./dist/model-en-mixed.bin` are binary files.
Each model contains multiple Gaussian prototypes, typically
one or more per viseme ID.

The byte layout of each prototype within the `.bin` file
is as follows:

Field | Length in bytes | Description
--- | --- | ---
phoneme | 4 | IPA phoneme stored as 1–2 UTF-16 characters.
N/A | 1 | Reserved for future use.
group | 1 | Group ID
N/A | 1 | Reserved for future use.
viseme | 1 | Viseme ID as an unsigned integer.
$\vec{\mu}$ | 12 * 4 | Mean feature vector (12 × float-32).
$\Sigma^{-1}$ | 78 * 4 | Lower-triangular part of the inverse covariance matrix (78 × float-32).

The file structure of each JSON (.json) viseme data is the following:

```json
[
  {
    "section": "A word or sentence", // the word/sentence (optional)
    "ps": [ // Phonemes
      {
        "p": "ɪ", // Phoneme, 1-2 letters
        "v": 0, // Viseme ID
        "t": 100, // Start time (ms)
        "d": 50  // Duration (ms)
      },
      ...
    ]
  },
  {
    // Next word/sentence
  }
]
```

---

### Appendix C: Performance

In theory, the processing window for each 128-sample
audio block is 2.9 ms at 44.1 kHz or 2.7 ms at 48 kHz.
However, we must allow generous headroom for overhead,
jitter, CPU spikes, and lower-end hardware. If we don't,
the browser will begin dropping audio frames.

In practice, the processing should finish within 15-20%
of the block duration. In our test environment<sup>[1]</sup>,
we targeted a typical execution time of < 0.4 ms.

Below are measurement results for real-time processing inside
the HeadAudio processor:

Step | Duration<sup>\[1]</sup> | Notes
--- | --- | ---
MFCC/FFT | 0.025 ms | Computes a single MFCC feature vector (including FFT) from a 512-sample frame.
Classifier | 0.005 ms | Computes a viseme prediction by evaluating Mahalanobis distances against 50 Gaussian prototypes.
Processor | ~0.035 ms | Total time to process one 128-sample block (MFCC/FFT + classification). Represents typical peak values during a 21-second speech test (44.1 kHz mono).
**TOTAL** | **<0.1 ms** | **Estimated max processing time per 128-sample block for 99.9% of frames.**

Analysis of test-run statistics:

<img src="images/processor.png" width="500"/>

Assuming processing time per prediction ~0.5 ms,
MFCC/FFT window size: 512 samples @ 16 kHz,
hop size: 256 samples @ 16 kHz,
and requiring three consecutive identical predictions before changing viseme,
the total end-to-end latency is approximately 50 ms.

Training performance:

Training step | Duration<sup>\[1]</sup> | Notes
--- | --- | ---
Prototype | 0.286 ms | Computes one Gaussian prototype from 1000 MFCC vectors.
**TOTAL** | **<30 s** | **Training 50 prototypes including audio processing.**

Distance matrix for `./dist/model-en-mixed.bin`:

<img src="images/distances.png"/>


<sup>\[1]</sup> *Test/training setup: Macbook Air M2 laptop, 8 cores, 16GB memory, latest desktop Chrome browser.*
