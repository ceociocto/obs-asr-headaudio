# Zoom 虚拟头像接入指南

## ⚠️ 重要说明

**OBS 浏览器源不支持麦克风权限**，因此需要在**外部浏览器**中运行页面，然后用 OBS 捕获浏览器窗口。

---

## 推荐方案：外部浏览器 + 窗口捕获

### 步骤

1. **启动本地服务器**
   ```bash
   cd /Volumes/sn7100/jerry/code/HeadAudio
   ./start-zoom-avatar.sh
   ```
   或手动启动：
   ```bash
   python3 -m http.server 8080
   ```

2. **在外部浏览器中打开页面**
   ```bash
   open http://localhost:8080/glm-tts.html?mode=auto
   ```
   或在浏览器中访问：
   ```
   http://localhost:8080/glm-tts.html?mode=auto
   ```

3. **配置 OBS 窗口捕获**
   - 打开 OBS Studio
   - 在"来源"面板点击 `+` 按钮
   - 选择 **"窗口捕获"** (Window Capture)
   - 属性设置：
     - 窗口：选择你的浏览器 → `glm-tts.html` 标签页
   - 点击"确定"

4. **调整浏览器窗口（可选）**
   - 将浏览器窗口调整到合适大小
   - 按 `F11` 全屏浏览器（去除地址栏和标签栏）
   - 或使用 CSS 来隐藏页面元素

5. **启动虚拟摄像头**
   - 在 OBS 中点击菜单栏 `控制` → `启动虚拟摄像机`
   - 或使用工具栏的相机图标

6. **在 Zoom 中使用**
   - 打开 Zoom 并加入会议
   - 点击"启动视频"旁的箭头 → "选择视频"
   - 选择 `OBS Virtual Camera`
   - 完成！

---

## 备选方案：Screen Capture

如果窗口捕获有问题，可以使用 Screen Capture：

1. 在 OBS 中添加 **"屏幕捕获"** (Screen Capture)
2. 方法：选择 `窗口` (Window)
3. 窗口：选择你的浏览器 → `glm-tts.html` 标签页
4. 鼠标指针：取消勾选

---

## 音频路由设置

### 方案 A：直接使用（简单）
- 浏览器音频直接输出到扬声器
- Zoom 使用相同扬声器
- **缺点**：会产生回声

### 方案 B：使用 BlackHole（推荐）
1. 安装 BlackHole：
   ```bash
   brew install blackhole-2ch
   ```

2. 配置多输出设备：
   - 打开"音频 MIDI 设置"
   - 创建多输出设备，包含：
     - BlackHole 2ch
     - 你的扬声器/耳机

3. 配置系统音频：
   - 系统输出：多输出设备
   - Zoom 输入：BlackHole 2ch

---

## URL 参数说明

| 参数 | 说明 |
|------|------|
| `?mode=auto` | 自动模式：隐藏控制面板，显示状态栏，自动开始监听 |
| `?status=show` | 显示状态栏 |
| `?controls=hide` | 隐藏控制面板 |

**推荐用于 Zoom**：
```
http://localhost:8080/glm-tts.html?mode=auto
```

---

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| 监听状态不启动 | 必须在外部浏览器中运行，OBS 浏览器源不支持麦克风 |
| 看不到控制面板 | URL 中去掉 `?mode=auto` 或 `?controls=hide` |
| 找不到窗口捕获 | 更新 OBS 到最新版本 |
| 画面是黑的 | 检查屏幕录制权限：系统设置 → 隐私 → 屏幕录制 |
| Zoom 听不到声音 | 配置 BlackHole 音频路由（见上方） |

---

## macOS 屏幕录制权限

首次使用必须授权：
- `系统设置` → `隐私与安全性` → `屏幕录制`
- 确保 OBS Studio 已勾选
- 重启 OBS 后生效
