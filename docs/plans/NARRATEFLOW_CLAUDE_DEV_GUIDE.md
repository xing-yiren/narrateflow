# NarrateFlow Claude Code 从头开发指南

版本：1.0
日期：2026-06-11
来源：整合《NarrateFlow视频配音方案V2》实现细节 + 《独立技术方案》架构设计 + 《工程可行版》安全约束
用途：作为 Claude Code 从零开始构建 NarrateFlow 配音系统的完整任务说明书

## 1. 项目目标

构建一个面向"无声视频 + 已准备好的旁白文稿"的自动配音与视频重定时系统。

**输入：**
- 无声视频文件（`.mp4`）
- 旁白文稿（`.txt`）
- 可选：参考音色、发音词典、人工锁定时间点

**输出：**
- 带旁白音轨的成片视频
- 可复查的中间产物：视频理解结果、TTS 音频、时间轴对齐文件、人工修正记录

**明确不做：**
- 口型对齐
- 画面重绘
- VLM 输出当作物理真值
- 第一版全自动闭眼出片

## 2. 核心设计原则

1. **真实时间轴优先**：视频边界、帧率、时长来自 `ffprobe`、OpenCV、PySceneDetect 等确定性工具。
2. **大模型只给候选**：VLM 输出视觉摘要、动作候选、意境标签，但不直接决定最终时间轴。
3. **人工审核内置**：关键节点（视觉理解、TTS 效果、时间轴）允许人类快速修正。
4. **渐进式演进**：先 MVP，再本地 VLM，再高级对齐，最后数据飞轮和微调。

## 3. 推荐代码结构

```
narrateflow/
├── cli.py                    # 主命令行入口
├── config.py                 # 全局配置管理
├── schemas.py                # 所有 JSON schema 定义和校验
├── paths.py                  # 项目目录和路径管理
│
├── stages/
│   ├── __init__.py
│   ├── stage0_prepare.py     # 资产探测和标准化
│   ├── stage1_video_meta.py  # 视频结构化理解
│   ├── stage2_audio.py       # 旁白音频生成
│   ├── stage3_timeline.py    # 时间轴对齐
│   └── stage4_render.py      # FFmpeg 渲染合成
│
├── video/
│   ├── __init__.py
│   ├── probe.py              # ffprobe 封装
│   ├── scenes.py             # PySceneDetect 镜头检测
│   └── frames.py             # OpenCV 抽帧
│
├── text/
│   ├── __init__.py
│   ├── segmenter.py          # 文稿解析和分段
│   └── pronunciation.py      # 发音词典管理
│
├── providers/
│   ├── __init__.py
│   ├── vlm_base.py           # VLM provider 抽象基类
│   ├── vlm_ollama.py         # Ollama 本地 VLM（Qwen3-VL）
│   ├── vlm_gemini.py         # Gemini API fallback
│   ├── vlm_mock.py           # Mock provider（测试用）
│   ├── tts_base.py           # TTS provider 抽象基类
│   ├── tts_qwen.py           # Qwen-TTS 本地 TTS
│   └── tts_mock.py           # Mock TTS（测试用）
│
├── audio/
│   ├── __init__.py
│   └── duration.py           # 音频时长检测
│
├── timeline/
│   ├── __init__.py
│   ├── scoring.py            # 综合匹配评分
│   ├── planner.py            # 时间轴规划
│   └── retime.py             # 重定时策略
│
├── render/
│   ├── __init__.py
│   └── ffmpeg.py             # FFmpeg 合成
│
├── review/
│   ├── __init__.py
│   └── status.py             # 审核状态管理
│
└── tests/
    ├── fixtures/
    ├── test_schemas.py
    ├── test_text_segmenter.py
    ├── test_timeline_scoring.py
    └── test_paths.py
```

## 4. 项目目录约定

每个视频任务使用一个独立 project dir：

```
<project-dir>/
├── project_manifest.json                    # 项目元信息
├── input/
│   ├── video.mp4
│   └── narration.txt
├── video/
│   ├── scenes.json                          # 镜头边界
│   └── frames/                              # 抽取的关键帧
├── meta/
│   ├── video_meta.ai.json                   # VLM 原始输出
│   └── video_meta.human.json                # 人工修正版
├── audio/
│   ├── segments/
│   │   ├── p001.wav
│   │   └── p002.wav
│   ├── narration_audio_manifest.ai.json     # TTS 原始清单
│   └── narration_audio_manifest.human.json  # 人工修正版
├── timeline/
│   ├── timeline_plan.ai.json                # 对齐计划
│   └── timeline_plan.human.json             # 人工修正版
├── render/
│   ├── final_video.mp4
│   └── render_log.json
└── logs/
    └── run_metrics.jsonl                    # 性能指标
```

**规则：**
- `.ai.json` 是系统或模型产物，可以覆盖。
- `.human.json` 是人工修正版，下游优先读取。
- 不自动覆盖人工修正版。
- 所有 JSON 必须带 `schema_version`。

## 5. 核心数据契约

### 5.1 `project_manifest.json`

```json
{
  "schema_version": "3.1",
  "video_path": "input/video.mp4",
  "script_path": "input/narration.txt",
  "video": {
    "duration_sec": 128.4,
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "codec": "h264"
  },
  "created_at": "2026-06-11T10:00:00+08:00"
}
```

**字段来源：**
- `duration_sec`、`fps`、`width`、`height`、`codec`：来自 `ffprobe`
- 禁止手工填写或估算

### 5.2 `video_meta.json`

```json
{
  "schema_version": "3.1",
  "windows": [
    {
      "window_id": "w0001",
      "start_time": 0.0,
      "end_time": 5.2,
      "boundary_source": "scenedetect",
      "visual_summary": "一名程序员在办公室中看着屏幕思考",
      "physical_tags": ["person", "office", "computer"],
      "art_tags": ["focused"],
      "visible_text": [],
      "actions": [
        {
          "label": "person touches head",
          "time_hint": 3.4,
          "confidence": 0.58,
          "source": "vlm_candidate"
        }
      ],
      "source_frames": [
        { "time": 1.3, "path": "frames/w0001_head.jpg" },
        { "time": 2.6, "path": "frames/w0001_mid.jpg" },
        { "time": 3.9, "path": "frames/w0001_tail.jpg" }
      ],
      "review_status": "pending"
    }
  ]
}
```

**字段约束：**
- `start_time` / `end_time`：来自物理切片（PySceneDetect），不可被 VLM 覆盖
- `boundary_source`：记录切片来源，例如 `scenedetect`、`manual`、`uniform`
- `actions[].time_hint`：VLM 估计的动作时间，只是候选
- `actions[].confidence`：VLM 输出置信度，0-1
- `actions[].source`：标注来源，如 `vlm_candidate`、`human_fixed`
- `review_status`：`pending` / `approved` / `fixed` / `rejected`

### 5.3 `narration_audio_manifest.json`

```json
{
  "schema_version": "3.1",
  "segments": [
    {
      "paragraph_index": 1,
      "text": "今天，我们来看一个本地视频配音系统。",
      "tts_text": "今天，我们来看一个本地视频配音系统。",
      "audio_path": "audio/segments/p001.wav",
      "duration_sec": 4.12,
      "word_timestamps": [],
      "tts_provider": "qwen_tts",
      "review_status": "approved"
    }
  ],
  "total_duration_sec": 45.8
}
```

**字段约束：**
- `duration_sec`：必须读取实际 WAV 文件头得出，不能用字数估算
- `tts_text`：TTS 实际使用的文本（可能经发音词典替换）
- `word_timestamps`：预留字段，P2/P3 时接入 forced alignment
- `review_status`：`pending` / `approved` / `rejected` / `regenerate`

### 5.4 `timeline_plan.json`

```json
{
  "schema_version": "3.1",
  "segments": [
    {
      "paragraph_index": 1,
      "audio_start": 0.0,
      "audio_end": 4.12,
      "audio_path": "audio/segments/p001.wav",
      "matched_window_ids": ["w0001"],
      "video_start": 0.0,
      "video_end": 5.2,
      "match_score": 0.84,
      "match_details": {
        "order_score": 1.0,
        "duration_score": 0.78,
        "visual_semantic_score": 0.82,
        "physical_tag_score": 0.9,
        "ocr_score": 0.0,
        "action_score": 0.5,
        "art_tag_score": 0.7
      },
      "retime_strategy": "slight_speed",
      "retime_params": {
        "speed_factor": 0.95,
        "freeze_frame_time": null,
        "freeze_duration_sec": null
      },
      "review_status": "approved"
    }
  ]
}
```

**字段约束：**
- `matched_window_ids`：可以是一个或多个连续窗口
- `match_score`：综合分数，0-1
- `retime_strategy`：`original` / `slight_speed` / `merge_windows` / `freeze_frame` / `broll` / `needs_review`
- 每个段落的 `video_end` 应大于等于 `video_start` + 最小可接受时长

## 6. Stage 0：资产探测与标准化

### 6.1 ffprobe 视频探测

```python
# video/probe.py

def probe_video(video_path: Path) -> dict:
    """使用 ffprobe 获取视频真实属性"""
    import subprocess, json

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "video"),
        None
    )
    if not video_stream:
        raise ValueError("No video stream found")

    return {
        "duration_sec": float(data["format"]["duration"]),
        "fps": eval(video_stream["r_frame_rate"]),  # 例如 "30/1" -> 30.0
        "width": video_stream["width"],
        "height": video_stream["height"],
        "codec": video_stream.get("codec_name", "unknown"),
    }
```

### 6.2 文稿解析

```python
# text/segmenter.py

def parse_narration(text: str) -> list[dict]:
    """将旁白文稿解析为带编号的段落列表"""
    paragraphs = []
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    # 也支持单行按句号分段
    for i, para in enumerate(raw, 1):
        paragraphs.append({
            "paragraph_index": i,
            "original_text": para,
            "is_title": i == 1 and len(para) < 30,  # 简短首段可能是标题
        })
    return paragraphs
```

### 6.3 生成 project_manifest.json

```python
# stages/stage0_prepare.py

def run_stage0_prepare(project_dir: Path, video_path: Path, script_path: Path) -> Path:
    from video.probe import probe_video
    from text.segmenter import parse_narration

    video_info = probe_video(video_path)
    script_text = script_path.read_text(encoding="utf-8")
    paragraphs = parse_narration(script_text)

    manifest = {
        "schema_version": "3.1",
        "video_path": str(video_path),
        "script_path": str(script_path),
        "video": video_info,
        "paragraphs": paragraphs,
        "created_at": datetime.now().isoformat(),
    }
    manifest_path = project_dir / "project_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
```

## 7. Stage 1：视频结构化理解

### 7.1 PySceneDetect 镜头检测

```python
# video/scenes.py

def detect_scenes(video_path: Path) -> list[dict]:
    """使用 PySceneDetect 检测镜头边界"""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    scene_manager.detect_scenes(video)

    scenes = scene_manager.get_scene_list()
    windows = []
    for i, (start, end) in enumerate(scenes):
        windows.append({
            "window_id": f"w{i+1:04d}",
            "start_time": round(start.get_seconds(), 2),
            "end_time": round(end.get_seconds(), 2),
            "boundary_source": "scenedetect",
        })
    return windows
```

**注意：**
- 对于长镜头（超过 15 秒），需要做二级切分，按固定间隔（例如 8-12 秒）切
- 对于录屏、PPT 类视频，可加入帧差检测或 UI 变化检测辅助切分

### 7.2 OpenCV 抽帧

```python
# video/frames.py

import cv2
from pathlib import Path

def extract_frames(
    video_path: Path,
    windows: list[dict],
    output_dir: Path,
    max_frames_per_window: int = 8,
    detection_max_width: int = 600,
) -> list[dict]:
    """为每个窗口抽取关键帧"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_paths = []

    for window in windows:
        duration = window["end_time"] - window["start_time"]
        # 短镜头：抽 1-3 帧
        if duration <= 5:
            times = [window["start_time"] + duration * p for p in [0.25, 0.5, 0.75]]
            # 去重
            times = sorted(set(round(t, 2) for t in times))
        else:
            # 长镜头：均匀抽帧，但不超过 max_frames_per_window
            n_frames = min(max_frames_per_window, int(duration / 1.5))
            times = [round(window["start_time"] + i * duration / (n_frames + 1), 2)
                     for i in range(1, n_frames + 1)]

        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            # 缩放到 detection_max_width
            h, w = frame.shape[:2]
            if w > detection_max_width:
                scale = detection_max_width / w
                frame = cv2.resize(frame, (detection_max_width, int(h * scale)))

            frame_name = f"{window['window_id']}_t{int(t*100):05d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append({
                "window_id": window["window_id"],
                "time": t,
                "path": str(frame_path.relative_to(output_dir.parent)),
            })

    cap.release()
    return frame_paths
```

**V2 实现要点：**
- 单帧长边限制到 600px，控制视觉 token 消耗
- RTX 3060 12GB：每窗口最多 8-12 帧，避免一次性塞给 VLM 太多帧
- 帧文件保存为 JPEG，质量 85-90%

### 7.3 VLM Provider 抽象

```python
# providers/vlm_base.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class VLMProvider(ABC):
    """视频理解 provider 抽象基类"""

    @abstractmethod
    def understand_window(self, window: dict, frame_paths: list[Path]) -> dict:
        """对单个视频窗口进行视觉理解，返回标准化结果"""
        pass

    @abstractmethod
    def supported(self) -> bool:
        """当前环境是否支持该 provider"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
```

### 7.4 Ollama 本地 VLM Provider（V2 核心实现）

```python
# providers/vlm_ollama.py

import json
import re
import requests
from pathlib import Path
from providers.vlm_base import VLMProvider

class OllamaVLMProvider(VLMProvider):
    """通过 Ollama 调用本地 Qwen3-VL-8B-Thinking"""

    def __init__(self, model: str = "qwen3-vl:8b-thinking-q6_k",
                 base_url: str = "http://localhost:11434",
                 max_tokens: int = 1024,
                 timeout: int = 120):
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return f"ollama/{self.model}"

    def supported(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _build_prompt(self, window: dict) -> str:
        """构建 VLM prompt（来源于 V2 的 TimeChat-Captioner 风格）"""
        return f"""你是一个视频内容分析助手。请仔细观察这些按时间顺序排列的视频帧，然后输出一个严格的 JSON 对象。

要求：
1. 用一句话描述这个镜头中正在发生什么（visual_summary）。
2. 列出画面中可见的物理对象和场景标签（physical_tags）。
3. 列出画面中的氛围或意境（art_tags），例如 "focused"、"tense"、"calm"。
4. 如果画面中有可见文字，请逐条列出（visible_text）。
5. 如果画面中有明显的动作，请描述动作并估计它在视频时间轴上的大致秒数（time_hint）。
6. 对每个动作给出置信度（confidence, 0-1）并标注来源为 "vlm_candidate"。
7. 不要编造任何不在画面中的信息。
8. 动作时间是估计值，不需要精确。
9. 只输出 JSON，不要输出任何其他文字。

当前视频窗口时间范围：{window['start_time']}s - {window['end_time']}s

输出格式：
{{
  "visual_summary": "...",
  "physical_tags": ["...", "..."],
  "art_tags": ["...", "..."],
  "visible_text": ["..."],
  "actions": [
    {{ "label": "...", "time_hint": 0.0, "confidence": 0.0, "source": "vlm_candidate" }}
  ]
}}"""

    def understand_window(self, window: dict, frame_paths: list[Path]) -> dict:
        """调用 Ollama API 进行视觉理解"""
        # 将帧编码为 base64
        import base64
        images = []
        for fp in frame_paths:
            with open(fp, "rb") as f:
                images.append(base64.b64encode(f.read()).decode())

        # Ollama API 调用
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": self._build_prompt(window),
                "images": images,
            }],
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.0,
            }
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw_content = resp.json()["message"]["content"]
        except Exception as e:
            return self._fallback_result(window, f"Ollama API error: {e}")

        return self._parse_response(raw_content, window)

    def _parse_response(self, raw_content: str, window: dict) -> dict:
        """解析 VLM 响应，提取 JSON（V2 的 thinking + regex 策略）"""
        # 先尝试直接解析 JSON
        try:
            return json.loads(raw_content.strip())
        except json.JSONDecodeError:
            pass

        # 提取 <thinking>...</thinking> 之后的 JSON（Qwen3-VL-Thinking 特性）
        # 移除 thinking 块
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', raw_content, flags=re.DOTALL)
        # 找到最后一个 JSON 块
        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        # 尝试更鲁棒的大括号匹配
        if not json_match:
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # 兜底：返回带错误标记的结果
        return self._fallback_result(window, "JSON parse failed")

    def _fallback_result(self, window: dict, reason: str) -> dict:
        return {
            "visual_summary": f"[VLM 解析失败: {reason}]",
            "physical_tags": [],
            "art_tags": [],
            "visible_text": [],
            "actions": [],
            "_parse_error": reason,
        }
```

**V2 关键实现细节：**
- 模型推荐：Mac M4 Pro 用 `qwen3-vl:8b-thinking-q8_0`，RTX 3060 用 `qwen3-vl:8b-thinking-q6_k`
- 设置 `OLLAMA_KEEP_ALIVE=0s` 让推理完成后立即释放显存
- Ollama 本地接口：`POST http://localhost:11434/api/chat`
- Prompt 风格源于 TimeChat-Captioner：要求输出结构化 JSON，明确窗口时间范围
- Thinking 模式：VLM 会输出 `<thinking>...</thinking>` + JSON，用正则提取最后一个 `{}` 块
- 输出 token 上限 1024，防止无限生成
- temperature=0.0 保证输出稳定性
- 推理超时 120 秒
- JSON 解析失败时只标记当前窗口为 error，不污染整条流水线

### 7.5 Gemini API Fallback Provider

```python
# providers/vlm_gemini.py

import os
from providers.vlm_base import VLMProvider
from google import genai

class GeminiVLMProvider(VLMProvider):
    """Gemini API 作为 cloud fallback"""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

    @property
    def provider_name(self) -> str:
        return f"gemini/{self.model}"

    def supported(self) -> bool:
        return bool(self.api_key)

    def understand_window(self, window: dict, frame_paths: list[Path]) -> dict:
        # ... Gemini API 调用实现 ...
        pass
```

### 7.6 Stage 1 主流程

```python
# stages/stage1_video_meta.py

def run_stage1_video_meta(
    project_dir: Path,
    vlm_provider: VLMProvider,
    detection_max_width: int = 600,
    max_frames_per_window: int = 8,
) -> Path:
    """运行视频结构化理解阶段"""
    from video.scenes import detect_scenes
    from video.frames import extract_frames
    from schemas import validate_video_meta

    manifest = json.loads((project_dir / "project_manifest.json").read_text(encoding="utf-8"))
    video_path = Path(manifest["video_path"])

    # 1. 检测镜头边界
    windows = detect_scenes(video_path)

    # 2. 长镜头二级切分
    windows = split_long_windows(windows, max_duration=15)

    # 3. 抽取关键帧
    frames_dir = project_dir / "video" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_info = extract_frames(video_path, windows, frames_dir,
                                max_frames_per_window, detection_max_width)

    # 4. VLM 理解每个窗口
    for window in windows:
        window_frames = [f for f in frame_info if f["window_id"] == window["window_id"]]
        frame_paths = [project_dir / f["path"] for f in window_frames]
        result = vlm_provider.understand_window(window, frame_paths)
        window.update(result)
        window["source_frames"] = window_frames
        # 动作时间自动 clamp 到窗口范围内
        for action in window.get("actions", []):
            hint = action.get("time_hint", 0)
            action["time_hint"] = max(window["start_time"],
                                      min(window["end_time"], hint))

    # 5. 输出 video_meta.ai.json
    video_meta = {
        "schema_version": "3.1",
        "windows": windows,
    }
    validate_video_meta(video_meta)
    output_path = project_dir / "meta" / "video_meta.ai.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(video_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
```

## 8. Stage 2：旁白音频生成

### 8.1 文本分段

```python
# text/segmenter.py

def segment_for_tts(paragraphs: list[dict], max_chars: int = 60, min_chars: int = 25) -> list[dict]:
    """将段落进一步切分为适合 TTS 的片段（25-60 字/段）"""
    segments = []
    for para in paragraphs:
        text = para["original_text"]
        # 优先按句号、分号切分
        parts = re.split(r'(?<=[。！？；])', text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= max_chars:
                segments.append({"text": part, "paragraph_index": para["paragraph_index"]})
            else:
                # 对超长句按逗号或固定长度切分
                sub = re.split(r'(?<=[，,、])', part)
                current = ""
                for s in sub:
                    if len(current) + len(s) <= max_chars:
                        current += s
                    else:
                        if current:
                            segments.append({"text": current.strip(),
                                             "paragraph_index": para["paragraph_index"]})
                        current = s
                if current:
                    segments.append({"text": current.strip(),
                                     "paragraph_index": para["paragraph_index"]})
    return segments
```

### 8.2 发音词典

```python
# text/pronunciation.py

class PronunciationDict:
    """发音替换词典"""

    def __init__(self, rules_path: Path | None = None):
        self.rules = {}
        if rules_path and rules_path.exists():
            self.rules = json.loads(rules_path.read_text(encoding="utf-8"))

    def apply(self, text: str) -> str:
        """应用所有发音规则替换文本"""
        result = text
        applied = []
        for pattern, replacement in self.rules.items():
            if pattern in result:
                result = result.replace(pattern, replacement)
                applied.append({"pattern": pattern, "replacement": replacement})
        return result, applied

    def add_rule(self, pattern: str, replacement: str):
        self.rules[pattern] = replacement

    def save(self, path: Path):
        path.write_text(json.dumps(self.rules, ensure_ascii=False, indent=2), encoding="utf-8")
```

**V2 发音词典示例：**
```json
{
  "CUDA": "库达",
  "Qwen": "千问",
  "3060": "三零六零",
  "VoxCPM2": "Vox CPM Two",
  "FFmpeg": "F F mpeg"
}
```

### 8.3 TTS Provider 抽象

```python
# providers/tts_base.py

from abc import ABC, abstractmethod
from pathlib import Path

class TTSProvider(ABC):
    """TTS provider 抽象基类"""

    @abstractmethod
    def synthesize(self, text: str, output_path: Path, **kwargs) -> float:
        """合成单段音频，返回实际时长（秒）"""
        pass

    @abstractmethod
    def supported(self) -> bool:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
```

### 8.4 音频真实时长检测

```python
# audio/duration.py

import struct

def get_wav_duration(wav_path: Path) -> float:
    """读取 WAV 文件头获取真实音频时长（无需依赖 soundfile/librosa）"""
    with open(wav_path, "rb") as f:
        # RIFF header
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError(f"Not a valid WAV file: {wav_path}")
        f.read(4)  # file size
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError(f"Not a valid WAV file: {wav_path}")

        # Find fmt chunk
        while True:
            chunk_id = f.read(4)
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                # 解包关键字段
                audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                num_channels = struct.unpack('<H', fmt_data[2:4])[0]
                sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                byte_rate = struct.unpack('<I', fmt_data[8:12])[0]
                bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
            elif chunk_id == b'data':
                data_size = chunk_size
                break
            else:
                f.seek(chunk_size, 1)

        if audio_format != 1:  # PCM
            raise ValueError(f"Unsupported audio format: {audio_format}")

        bytes_per_sample = bits_per_sample // 8
        num_samples = data_size // (num_channels * bytes_per_sample)
        return num_samples / sample_rate
```

### 8.5 Stage 2 主流程

```python
# stages/stage2_audio.py

def run_stage2_audio(
    project_dir: Path,
    tts_provider: TTSProvider,
    pronunciation: PronunciationDict | None = None,
) -> Path:
    """运行旁白音频生成阶段"""
    from text.segmenter import segment_for_tts
    from audio.duration import get_wav_duration

    manifest = json.loads((project_dir / "project_manifest.json").read_text(encoding="utf-8"))
    paragraphs = manifest["paragraphs"]

    # 1. 文本分段
    segments = segment_for_tts(paragraphs)

    # 2. 生成音频
    audio_dir = project_dir / "audio" / "segments"
    audio_dir.mkdir(parents=True, exist_ok=True)
    narration_segments = []

    for i, seg in enumerate(segments):
        # 发音词典替换
        tts_text = seg["text"]
        if pronunciation:
            tts_text, applied = pronunciation.apply(tts_text)

        output_path = audio_dir / f"p{seg['paragraph_index']:03d}_{i+1:02d}.wav"

        # 调用 TTS
        tts_provider.synthesize(tts_text, output_path)

        # 读取真实时长
        duration = get_wav_duration(output_path)

        narration_segments.append({
            "paragraph_index": seg["paragraph_index"],
            "text": seg["text"],
            "tts_text": tts_text,
            "audio_path": str(output_path.relative_to(project_dir)),
            "duration_sec": round(duration, 3),
            "tts_provider": tts_provider.provider_name,
            "review_status": "pending",
        })

    # 3. 输出
    audio_manifest = {
        "schema_version": "3.1",
        "segments": narration_segments,
        "total_duration_sec": round(sum(s["duration_sec"] for s in narration_segments), 3),
    }
    output_path = project_dir / "audio" / "narration_audio_manifest.ai.json"
    output_path.write_text(json.dumps(audio_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
```

## 9. Stage 3：时间轴对齐

### 9.1 综合匹配评分（V3 核心算法）

```python
# timeline/scoring.py

def score_paragraph_window(
    paragraph: dict,
    window: dict,
    paragraph_order: int,
    window_order: int,
    total_paragraphs: int,
    total_windows: int,
) -> dict:
    """计算一个段落与一个窗口的综合匹配分数"""

    # 1. 顺序分数（order_score）
    # 期望：第 i 个段落匹配约第 i * (total_windows/total_paragraphs) 个窗口
    expected_pos = paragraph_order * total_windows / max(total_paragraphs, 1)
    order_diff = abs(window_order - expected_pos) / max(total_windows, 1)
    order_score = max(0, 1.0 - order_diff)

    # 2. 时长分数（duration_score）
    para_duration = paragraph.get("duration_sec", 0)
    window_duration = window["end_time"] - window["start_time"]
    if para_duration > 0 and window_duration > 0:
        ratio = min(para_duration, window_duration) / max(para_duration, window_duration)
        duration_score = ratio
    else:
        duration_score = 0.5

    # 3. 视觉语义分数（visual_semantic_score）
    para_text = paragraph.get("text", "")
    visual_text = window.get("visual_summary", "")
    visual_semantic_score = _simple_text_similarity(para_text, visual_text)

    # 4. 物理标签分数（physical_tag_score）
    physical_tags = window.get("physical_tags", [])
    physical_tag_score = _tag_overlap_score(para_text, physical_tags)

    # 5. OCR 文本分数（ocr_score）
    visible_text = window.get("visible_text", [])
    ocr_score = _tag_overlap_score(para_text, visible_text)

    # 6. 动作分数（action_score）
    actions = window.get("actions", [])
    action_labels = [a["label"] for a in actions]
    action_score = _tag_overlap_score(para_text, action_labels) * 0.5  # 动作匹配打折

    # 7. 意境标签分数（art_tag_score）
    art_tags = window.get("art_tags", [])
    art_tag_score = _tag_overlap_score(para_text, art_tags) * 0.3  # 意境匹配要很大折扣

    # 综合加权
    weights = {
        "order_score": 0.25,
        "duration_score": 0.25,
        "visual_semantic_score": 0.20,
        "physical_tag_score": 0.10,
        "ocr_score": 0.10,
        "action_score": 0.05,
        "art_tag_score": 0.05,
    }

    total = (
        weights["order_score"] * order_score +
        weights["duration_score"] * duration_score +
        weights["visual_semantic_score"] * visual_semantic_score +
        weights["physical_tag_score"] * physical_tag_score +
        weights["ocr_score"] * ocr_score +
        weights["action_score"] * action_score +
        weights["art_tag_score"] * art_tag_score
    )

    return {
        "total_score": round(total, 3),
        "order_score": round(order_score, 3),
        "duration_score": round(duration_score, 3),
        "visual_semantic_score": round(visual_semantic_score, 3),
        "physical_tag_score": round(physical_tag_score, 3),
        "ocr_score": round(ocr_score, 3),
        "action_score": round(action_score, 3),
        "art_tag_score": round(art_tag_score, 3),
    }


def _simple_text_similarity(text1: str, text2: str) -> float:
    """简单的 Jaccard 风格文本相似度"""
    import re
    def tokens(s):
        return set(re.findall(r'[一-鿿]+|\w+', s.lower()))
    t1, t2 = tokens(text1), tokens(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _tag_overlap_score(text: str, tags: list[str]) -> float:
    """标签与文本的匹配度"""
    if not tags:
        return 0.0
    import re
    text_lower = text.lower()
    matches = sum(1 for tag in tags if tag.lower() in text_lower)
    return matches / len(tags)
```

### 9.2 全局对齐规划

```python
# timeline/planner.py

def plan_timeline(
    paragraphs: list[dict],
    windows: list[dict],
    human_locks: dict | None = None,
) -> list[dict]:
    """使用贪心 + 滑窗 + 单调约束做全局对齐"""
    from timeline.scoring import score_paragraph_window

    segments = []
    used_window_idx = -1  # 已使用的最后一个窗口索引

    for i, para in enumerate(paragraphs):
        best_window_idx = None
        best_score = -1
        best_details = None

        # 候选窗口：从上一个位置开始，往后搜索
        search_start = max(0, used_window_idx + 1)
        search_end = min(len(windows), search_start + 5)  # 最多往后看 5 个窗口

        # 人工锁定最高优先级
        if human_locks and i in human_locks:
            locked_win = human_locks[i]
            segments.append({
                "paragraph_index": i,
                "matched_window_ids": [locked_win],
                "match_score": 1.0,
                "review_status": "human_locked",
            })
            used_window_idx = windows.index(
                next(w for w in windows if w["window_id"] == locked_win))
            continue

        for j in range(search_start, min(search_end, len(windows))):
            # 一个段落可以匹配多个连续窗口
            # 先算单个窗口匹配分数
            details = score_paragraph_window(para, windows[j], i, j,
                                             len(paragraphs), len(windows))
            if details["total_score"] > best_score:
                best_score = details["total_score"]
                best_window_idx = j
                best_details = details

        if best_window_idx is not None:
            # 尝试滑窗合并：如果一段音频时长超过单窗口，尝试合并相邻窗口
            para_duration = para.get("duration_sec", 0)
            window_duration = windows[best_window_idx]["end_time"] - windows[best_window_idx]["start_time"]
            matched_ids = [windows[best_window_idx]["window_id"]]
            video_start = windows[best_window_idx]["start_time"]
            video_end = windows[best_window_idx]["end_time"]

            # 如果音频比窗口长，合并下一个窗口
            k = best_window_idx + 1
            while para_duration > (video_end - video_start) * 0.8 and k < len(windows):
                # 合并相邻窗口
                next_details = score_paragraph_window(para, windows[k], i, k,
                                                      len(paragraphs), len(windows))
                if next_details["total_score"] > 0.3:  # 阈值
                    matched_ids.append(windows[k]["window_id"])
                    video_end = windows[k]["end_time"]
                    k += 1
                else:
                    break

            segments.append({
                "paragraph_index": i,
                "matched_window_ids": matched_ids,
                "video_start": video_start,
                "video_end": video_end,
                "match_score": best_score,
                "match_details": best_details,
                "review_status": "approved" if best_score >= 0.5 else "needs_review",
            })
            used_window_idx = best_window_idx + len(matched_ids) - 1
        else:
            segments.append({
                "paragraph_index": i,
                "matched_window_ids": [],
                "match_score": 0.0,
                "review_status": "needs_review",
            })

    return segments
```

### 9.3 重定时策略

```python
# timeline/retime.py

def determine_retime_strategy(
    segment: dict,
    speed_range: tuple[float, float] = (0.90, 1.10),
    max_freeze_sec: float = 0.8,
) -> dict:
    """决定每个段落的视频重定时策略

    优先级（V3 修正）：
    1. 原速使用
    2. 滑窗合并镜头（已在 planner 中处理）
    3. 轻微变速
    4. B-roll 补画面
    5. 短定格（<= 0.8s）
    6. 人工 review
    """
    video_duration = segment["video_end"] - segment["video_start"]
    audio_duration = segment.get("audio_duration", video_duration)

    if video_duration <= 0 or audio_duration <= 0:
        return {"strategy": "needs_review", "params": {}}

    ratio = audio_duration / video_duration

    # 1. 原速：时长匹配在 ±5% 以内
    if 0.95 <= ratio <= 1.05:
        return {"strategy": "original", "params": {"speed_factor": 1.0}}

    # 2. 轻微变速：时长差在 speed_range 范围内
    min_speed, max_speed = speed_range
    if min_speed <= ratio <= max_speed:
        return {"strategy": "slight_speed",
                "params": {"speed_factor": round(ratio, 3)}}

    # 3. 如果音频比视频长很多，用定格补充
    #    但定格不能超过 max_freeze_sec
    if ratio > max_speed:
        remainder = audio_duration - video_duration * max_speed
        if remainder <= max_freeze_sec:
            return {
                "strategy": "freeze_frame",
                "params": {
                    "speed_factor": max_speed,
                    "freeze_frame_time": segment["video_end"],  # 窗口结尾定格
                    "freeze_duration_sec": round(remainder, 2),
                }
            }

    # 4. 音频太短，视频太长 → 可以加速视频
    #    或者标记为需要 B-roll / 人工 review
    if ratio < min_speed:
        return {
            "strategy": "slight_speed",
            "params": {"speed_factor": round(min_speed, 3)},
        }

    # 5. 兜底：人工 review
    return {"strategy": "needs_review", "params": {}}
```

**V2/V3 定格策略要点：**
- 定格优选动作完成后的低运动帧，不是最后一帧
- 定格局限于低运动帧或句子停顿点附近
- 单次定格上限 0.8 秒
- 超过阈值标记为 `needs_review`

## 10. Stage 4：FFmpeg 渲染合成

### 10.1 FFmpeg 渲染

```python
# render/ffmpeg.py

import subprocess
from pathlib import Path

def render_final_video(
    project_dir: Path,
    timeline_path: Path,
    audio_manifest_path: Path,
    output_path: Path,
    use_hw_accel: bool = True,
) -> Path:
    """使用 FFmpeg 合成最终视频"""
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    audio_manifest = json.loads(audio_manifest_path.read_text(encoding="utf-8"))
    manifest = json.loads((project_dir / "project_manifest.json").read_text(encoding="utf-8"))

    video_path = project_dir / manifest["video_path"]

    # 构建 FFmpeg filter_complex
    # 对于每个段落：
    # 1. 切割视频片段
    # 2. 应用变速（setpts）
    # 3. 可选定格（tpad）
    # 4. 拼接视频片段（concat）
    # 5. 拼接音频
    # 6. 混音输出

    # 简化版：使用 concat demuxer
    segments = timeline["segments"]
    video_filters = []
    audio_inputs = []
    concat_file = project_dir / "render" / "concat_list.txt"
    concat_lines = []

    for seg in segments:
        if seg["review_status"] == "needs_review":
            continue

        video_start = seg["video_start"]
        video_end = seg["video_end"]
        audio_path = project_dir / audio_manifest["segments"][seg["paragraph_index"] - 1]["audio_path"]

        # 变速处理
        speed = seg.get("retime_params", {}).get("speed_factor", 1.0)
        if speed != 1.0:
            # 使用 setpts 滤镜变速
            pts_factor = 1.0 / speed
            segment_output = project_dir / "render" / f"seg_{seg['paragraph_index']:03d}.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(video_start),
                "-to", str(video_end - video_start),
                "-i", str(video_path),
                "-filter:v", f"setpts={pts_factor}*PTS",
                "-an",
                str(segment_output),
            ], check=True)
        else:
            segment_output = project_dir / "render" / f"seg_{seg['paragraph_index']:03d}.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(video_start),
                "-to", str(video_end - video_start),
                "-i", str(video_path),
                "-c", "copy",
                "-an",
                str(segment_output),
            ], check=True)

        concat_lines.append(f"file '{segment_output}'")
        audio_inputs.append(str(audio_path))

    # 写入 concat 列表
    concat_file.write_text("\n".join(concat_lines), encoding="utf-8")

    # 拼接视频
    video_concat = project_dir / "render" / "video_concat.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(video_concat),
    ], check=True)

    # 拼接音频并混入视频
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_concat),
        "-i", f"concat:{'|'.join(audio_inputs)}",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ], check=True)

    return output_path
```

**硬件加速（V2 建议）：**
```python
def get_encoder():
    """探测可用的硬件编码器"""
    import platform
    system = platform.system()
    if system == "Windows":
        return "h264_nvenc"  # NVIDIA GPU
    elif system == "Darwin":
        return "h264_videotoolbox"  # macOS
    else:
        return "libx264"  # 软件编码兜底
```

### 10.2 输出检查

```python
def verify_output(output_path: Path, expected_duration: float):
    """检查输出视频"""
    from video.probe import probe_video

    if not output_path.exists():
        raise FileNotFoundError(f"Output video not found: {output_path}")

    info = probe_video(output_path)
    actual_duration = info["duration_sec"]
    diff = abs(actual_duration - expected_duration)

    if diff > 5.0:  # 允许 5 秒误差
        print(f"WARNING: Duration mismatch. Expected {expected_duration}s, got {actual_duration}s")
    else:
        print(f"Output OK: {actual_duration}s, size: {output_path.stat().st_size / 1024 / 1024:.1f}MB")
```

## 11. 显存与进程安全（V2 关键实现）

### 11.1 GPU 显存监控

```python
# utils/gpu_monitor.py

def get_gpu_memory() -> dict:
    """获取 GPU 显存使用情况（NVIDIA）"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "total_mb": info.total // 1024 // 1024,
            "used_mb": info.used // 1024 // 1024,
            "free_mb": info.free // 1024 // 1024,
        }
    except Exception:
        return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "error": "pynvml unavailable"}
```

### 11.2 进程隔离与 OOM 防护

```python
# utils/process_guard.py

import psutil
import subprocess
import time

def run_stage_in_subprocess(stage_func, *args, **kwargs):
    """在独立子进程中运行阶段，完成后强制释放内存"""
    # V2 策略：VLM 完成后 os.kill 销毁进程
    # 使用 subprocess 或 multiprocessing

    import multiprocessing
    proc = multiprocessing.Process(target=stage_func, args=args, kwargs=kwargs)
    proc.start()
    proc.join()

    if proc.exitcode is None:
        proc.kill()  # 强制销毁
        proc.join()

    return proc.exitcode


def cleanup_ollama():
    """清理 Ollama 模型释放显存（V2 关键步骤）"""
    import requests
    try:
        # 发送 keep_alive=0 让模型立即卸载
        requests.post("http://localhost:11434/api/generate", json={
            "model": "qwen3-vl:8b-thinking-q6_k",
            "keep_alive": 0,
        }, timeout=5)
    except Exception:
        pass


def ensure_vram_budget(required_mb: int = 4000):
    """等待直到有足够显存"""
    for _ in range(30):  # 最多等待 30 秒
        mem = get_gpu_memory()
        if mem.get("free_mb", 0) >= required_mb:
            return True
        time.sleep(1)
    return False
```

**V2 显存安全要点：**
- VLM 完成后设置 `OLLAMA_KEEP_ALIVE=0s`，立即释放模型
- RTX 3060 12GB：VLM 和 TTS 不同时常驻
- Windows：Python 使用 `taskkill` 或 `psutil` 杀掉僵尸进程
- macOS：Python 使用 `process.kill()` 并通过 `psutil`/`pynvml` 监控
- 若显存超阈值：降帧数 → 降分辨率 → 切短片段 → 换小模型 → 云端 fallback

## 12. Human-in-the-loop 审核流程

### 12.1 审核状态管理

```python
# review/status.py

def get_reviewable_path(project_dir: Path, stage: str) -> Path:
    """获取当前阶段的审核文件路径"""
    ai_path = project_dir / get_ai_path(stage)
    human_path = project_dir / get_human_path(stage)
    # 优先返回人工修正版
    if human_path.exists():
        return human_path
    return ai_path


def approve_segment(file_path: Path, segment_index: int):
    """将某段标记为已审核"""
    data = json.loads(file_path.read_text(encoding="utf-8"))
    segments = data.get("segments") or data.get("windows")
    if segments and 0 <= segment_index < len(segments):
        segments[segment_index]["review_status"] = "approved"
        # 写入 human.json
        human_path = Path(str(file_path).replace(".ai.json", ".human.json"))
        human_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
```

### 12.2 各阶段审核内容

| 阶段 | 审核内容 | 操作 |
|------|---------|------|
| Stage 1 | 镜头切片是否合理、视觉摘要是否准确、OCR 是否遗漏 | 修改 `video_meta.human.json`，不覆盖 `.ai.json` |
| Stage 2 | 发音、断句、语速、段落衔接 | 单段重跑 TTS，更新发音词典，标记段落 approved |
| Stage 3 | 旁白画面匹配、卡顿、定格长度、手动锁点 | 修改 `timeline_plan.human.json`，从 Stage 4 重渲染 |

## 13. CLI 命令行

```python
# cli.py

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="NarrateFlow 视频配音系统")
    sub = parser.add_subparsers(dest="command")

    # init
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--project-dir", required=True)
    init_parser.add_argument("--video", required=True)
    init_parser.add_argument("--script", required=True)

    # run-stage
    stage_parser = sub.add_parser("run-stage")
    stage_parser.add_argument("--project-dir", required=True)
    stage_parser.add_argument("--stage", required=True,
                              choices=["0", "1", "2", "3", "4", "all"])
    stage_parser.add_argument("--vl-provider", default="ollama",
                              choices=["ollama", "gemini", "mock"])
    stage_parser.add_argument("--tts-provider", default="qwen",
                              choices=["qwen", "mock"])
    stage_parser.add_argument("--from-stage", type=int)

    # review
    review_parser = sub.add_parser("review")
    review_parser.add_argument("--project-dir", required=True)
    review_parser.add_argument("--stage", required=True,
                               choices=["1", "2", "3"])

    args = parser.parse_args()
    # ... dispatch to stage functions

if __name__ == "__main__":
    main()
```

## 14. JSON Schema 校验

```python
# schemas.py

VIDEO_META_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "windows"],
    "properties": {
        "schema_version": {"type": "string"},
        "windows": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["window_id", "start_time", "end_time"],
                "properties": {
                    "window_id": {"type": "string"},
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "boundary_source": {"type": "string"},
                    "visual_summary": {"type": "string"},
                    "physical_tags": {"type": "array", "items": {"type": "string"}},
                    "art_tags": {"type": "array", "items": {"type": "string"}},
                    "visible_text": {"type": "array", "items": {"type": "string"}},
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["label"],
                            "properties": {
                                "label": {"type": "string"},
                                "time_hint": {"type": "number"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "source": {"type": "string"},
                            }
                        }
                    },
                    "source_frames": {"type": "array"},
                    "review_status": {"type": "string"},
                }
            }
        }
    }
}


def validate_video_meta(data: dict):
    """校验 video_meta.json schema"""
    import jsonschema
    jsonschema.validate(data, VIDEO_META_SCHEMA)
    # 额外的业务规则校验
    for w in data["windows"]:
        assert w["start_time"] < w["end_time"], f"Invalid window times in {w['window_id']}"
        for a in w.get("actions", []):
            hint = a.get("time_hint", 0)
            assert w["start_time"] <= hint <= w["end_time"], \
                f"Action time_hint {hint} out of window range in {w['window_id']}"
```

## 15. 性能监控

```python
# utils/metrics.py

import time
import json
from pathlib import Path

class RunMetrics:
    """记录每个阶段的运行指标"""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.metrics = []

    def record(self, stage: str, provider: str, **kwargs):
        entry = {
            "timestamp": time.time(),
            "stage": stage,
            "provider": provider,
            **kwargs,
        }
        self.metrics.append(entry)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**建议记录的指标：**
- VLM 推理耗时（单窗口和总耗时）
- VLM 显存峰值
- TTS 推理耗时（单段和总耗时）
- TTS 显存峰值
- FFmpeg 渲染耗时
- JSON 解析失败次数
- 人工修改次数
- 最终视频时长

## 16. 渐进式技术路线

### P0：MVP 跑通（当前阶段）

**输入**：视频 + 旁白文稿
**输出**：带旁白音轨的视频

**实现**：
1. `project_manifest.json`：视频探测 + 文稿解析
2. `video_meta.json`：PySceneDetect 镜头边界 + VLM（Ollama/Gemini）视觉理解
3. `narration_audio_manifest.json`：TTS 分段生成 + 真实时长检测
4. `timeline_plan.json`：综合评分匹配 + 滑窗 + 轻微变速
5. FFmpeg 合成

**验收标准**：
- 3-5 分钟视频完整跑通
- 支持从任意阶段重跑
- 人工可修改视觉理解和时间轴
- 3060 上不 OOM 或可自动降级

### P1：本地 VLM 深度接入

- 接入 Qwen3-VL-8B-Thinking（Ollama）
- 抽帧预算控制（8-12 帧/窗口，600px）
- 结构化输出 + schema 校验
- 局部失败重试
- 显存峰值记录

### P2：对齐质量提升

- 综合评分匹配调优
- 滑窗镜头组合优化
- 轻微变速精细化
- B-roll 补画面
- 短定格增强
- forced alignment 可选接入

### P3：TTS 质量提升

- VoxCPM2 / 多 TTS provider 压测
- 发音词典完善
- 长文本分段优化
- 音频上下文承接（上一段末尾 8 字 + 4 秒作为 prompt）

### P4：产品化与数据飞轮

- 审核 UI
- 评测集构建
- few-shot 示例池
- 规则库沉淀
- 可选微调实验

## 17. 主要风险与兜底

| 风险 | 影响 | 兜底 |
|------|------|------|
| VLM 动作时间不准 | 对位错位 | 只作为候选，低置信进入人工审核 |
| 本地 VLM 显存溢出 | 任务失败 | 降帧→降分辨率→换小模型→云端 fallback |
| TTS 读错词 | 成片不可用 | 发音词典 + 单段重跑 |
| 文稿过长 > 视频 | 对齐困难 | 文稿压缩建议 + 镜头合并 + B-roll + 人工 review |
| 长定格观感差 | 画面卡顿 | 定格上限 0.8s，优先变速和补画面 |
| JSON 解析失败 | 下游阻塞 | schema 校验 + 局部重试 + 人工修正 |
| VLM Thinking 输出格式乱 | 解析困难 | 正则兜底 + 结构化 prompt + 单窗口重跑 |

## 18. 开发顺序建议

1. **先写 schemas.py + tests/test_schemas.py**：所有 JSON 都从严格 schema 开始
2. **再写 video/probe.py + video/scenes.py + video/frames.py**：纯确定性工具，不需要模型
3. **再写 text/segmenter.py + text/pronunciation.py**：纯文本处理
4. **再写 providers/vlm_mock.py + providers/tts_mock.py**：先跑通全流程，不依赖真实模型
5. **Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4**：逐步串联
6. **最后接入真实 VLM（Ollama）和 TTS**：替换 mock provider
7. **加入 review 流程 + metrics 记录**
8. **调优对齐算法 + FFmpeg 渲染质量**
