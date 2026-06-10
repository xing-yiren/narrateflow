# NarrateFlow Claude Code 开发实施方案

版本：V3.1 Implementation Ready  
日期：2026-06-09  
来源：整合 `NARRATEFLOW_STANDALONE_TECH_PLAN.md` 与 `NARRATEFLOW_TECH_PLAN_V3_FEASIBLE.md`。  
用途：作为 Claude Code 开始编码的任务说明书。本文档优先描述“怎么落地开发”，不再重复大篇幅方案论证。

## 1. 开发总目标

开发一个面向“无声视频 + 已准备好的旁白文稿”的自动配音与视频重定时系统。

第一版必须做到：

- 读取无声视频和旁白文稿。
- 获取视频真实时长、fps、分辨率、镜头窗口。
- 将旁白切成可 TTS 的段落。
- 生成或接入分段音频，并读取每段音频真实时长。
- 将旁白段落匹配到视频窗口。
- 生成可人工修改的时间轴计划。
- 调用 FFmpeg 输出最终带旁白视频。

第一版明确不做：

- 不做口型对齐。
- 不做画面重绘。
- 不依赖 VLM 给出精确动作时间戳。
- 不在 MVP 阶段做模型微调。
- 不把长时间尾帧定格作为默认策略。

核心工程原则：

- 真实时间轴来自 `ffprobe`、OpenCV、PySceneDetect 或等价确定性工具。
- VLM 只提供语义候选，不直接决定最终时间轴。
- TTS 对齐以实际音频文件时长为准，不用字数估算替代。
- 所有阶段都通过 JSON 中间产物交接，方便人工审核和单阶段重跑。
- 任何 AI 输出都必须经过 schema 校验。

## 2. MVP 范围

Claude Code 应优先完成 P0-MVP，而不是立刻接入 Qwen3-VL、VoxCPM2、GUI 或微调。

### P0-MVP 必做能力

1. 项目目录初始化。
2. 视频资产探测。
3. 旁白文稿解析和分段。
4. 视频窗口生成。
5. mock VLM provider，先生成结构合法的 `video_meta.json`。
6. mock 或已有 TTS provider，先生成结构合法的 `narration_audio_manifest.json`。
7. 基于时长和顺序的基础 timeline 匹配。
8. FFmpeg 合成。
9. CLI 支持从任意阶段继续运行。
10. JSON schema 校验和明确错误信息。

### P0-MVP 可暂缓

- Qwen3-VL 本地推理。
- VoxCPM2 真实 TTS。
- forced alignment。
- B-roll 智能补画面。
- 复杂 embedding 语义匹配。
- 审核 GUI。
- 数据飞轮自动 few-shot 注入。

## 3. 推荐代码结构

如果当前仓库已有类似模块，优先复用现有模块；否则按下面结构新增。

```text
narrateflow/
  __init__.py
  cli.py
  config.py
  schemas.py
  paths.py

  stages/
    __init__.py
    stage0_prepare.py
    stage1_video_meta.py
    stage2_audio.py
    stage3_timeline.py
    stage4_render.py

  video/
    __init__.py
    probe.py
    scenes.py
    frames.py

  text/
    __init__.py
    segmenter.py
    pronunciation.py

  providers/
    __init__.py
    vlm_base.py
    vlm_mock.py
    vlm_qwen3_local.py      # P1 再实现
    tts_base.py
    tts_mock.py
    tts_voxcpm2.py          # P3 再实现

  audio/
    __init__.py
    duration.py
    align_words.py          # P2/P3 可选

  timeline/
    __init__.py
    scoring.py
    planner.py
    retime.py

  render/
    __init__.py
    ffmpeg.py

  review/
    __init__.py
    status.py

tests/
  fixtures/
  test_schemas.py
  test_text_segmenter.py
  test_timeline_planner.py
  test_paths.py
```

如果不想新建完整包，也可以把这些模块映射到现有目录；但对 Claude Code 的要求是：阶段边界和 JSON 契约必须保持清晰。

## 4. 项目目录约定

每个视频任务使用一个独立 project dir。

```text
<project-dir>/
  input/
    video.mp4
    narration.txt
    pronunciation_rules.json

  manifest/
    project_manifest.json

  video/
    scenes.json
    frames/

  meta/
    video_meta.ai.json
    video_meta.human.json

  audio/
    segments/
    narration_audio_manifest.ai.json
    narration_audio_manifest.human.json

  timeline/
    timeline_plan.ai.json
    timeline_plan.human.json

  render/
    final_video.mp4
    render_log.json

  logs/
    run_metrics.jsonl
```

规则：

- `.ai.json` 是系统或模型产物。
- `.human.json` 是人工修正版。
- 下游阶段优先读取 `.human.json`，不存在时读取 `.ai.json`。
- 不覆盖人工修正版。

## 5. 数据契约

所有 JSON 必须带 `schema_version`。推荐当前版本为 `3.1`。

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
    "height": 1080
  },
  "created_at": "2026-06-09T10:00:00+08:00"
}
```

### 5.2 `video_meta.ai.json`

```json
{
  "schema_version": "3.1",
  "windows": [
    {
      "window_id": "w0001",
      "start_time": 0.0,
      "end_time": 5.2,
      "boundary_source": "scenedetect",
      "visual_summary": "A person works at a computer in an office.",
      "physical_tags": ["person", "office", "computer"],
      "art_tags": ["focused"],
      "visible_text": [],
      "actions": [
        {
          "label": "person looks at screen",
          "time_hint": 2.6,
          "confidence": 0.5,
          "source": "vlm_candidate"
        }
      ],
      "source_frames": [
        {
          "time": 2.6,
          "path": "video/frames/w0001_mid.jpg"
        }
      ],
      "review_status": "pending"
    }
  ]
}
```

字段约束：

- `start_time < end_time`。
- `start_time` 和 `end_time` 必须来自物理切片。
- `actions[].time_hint` 只能作为候选。
- `review_status` 枚举：`pending`、`approved`、`fixed`、`rejected`、`needs_review`。

### 5.3 `narration_audio_manifest.ai.json`

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
      "tts_provider": "mock",
      "review_status": "pending"
    }
  ]
}
```

字段约束：

- `duration_sec` 必须来自实际音频检测。
- `paragraph_index` 从 1 开始并保持稳定。
- `tts_text` 是经过发音词典处理后的文本。

### 5.4 `timeline_plan.ai.json`

```json
{
  "schema_version": "3.1",
  "segments": [
    {
      "paragraph_index": 1,
      "audio_start": 0.0,
      "audio_end": 4.12,
      "matched_window_ids": ["w0001"],
      "video_start": 0.0,
      "video_end": 5.2,
      "match_score": 0.84,
      "retime_strategy": "slight_speed",
      "retime_params": {
        "speed": 1.0,
        "freeze_points": []
      },
      "review_status": "pending"
    }
  ]
}
```

字段约束：

- `audio_start` / `audio_end` 根据音频段累加生成。
- `video_start` / `video_end` 来自匹配窗口。
- `match_score` 范围为 `0.0-1.0`。
- `retime_strategy` 枚举：`none`、`slight_speed`、`merge_windows`、`short_freeze`、`needs_review`。

## 6. CLI 设计

推荐命令：

```bash
python -m narrateflow.cli prepare --project-dir outputs/demo --video input.mp4 --script narration.txt
python -m narrateflow.cli video-meta --project-dir outputs/demo --vlm-provider mock
python -m narrateflow.cli audio --project-dir outputs/demo --tts-provider mock
python -m narrateflow.cli timeline --project-dir outputs/demo
python -m narrateflow.cli render --project-dir outputs/demo
python -m narrateflow.cli run --project-dir outputs/demo --video input.mp4 --script narration.txt --vlm-provider mock --tts-provider mock
```

CLI 要求：

- 每个 stage 可单独运行。
- `run` 等于从 prepare 到 render 顺序运行。
- 每个 stage 运行前校验依赖文件是否存在。
- 每个 stage 输出明确路径。
- 失败时打印缺失文件、schema 错误或外部命令错误。

## 7. Stage 开发说明

### 7.1 Stage 0：prepare

输入：

- `--video`
- `--script`
- `--project-dir`

处理：

- 创建 project dir。
- 复制或记录输入资产。
- 调用 `ffprobe` 获取视频真实信息。
- 解析旁白文稿，生成内部段落列表。
- 写入 `manifest/project_manifest.json`。

输出：

- `manifest/project_manifest.json`
- `input/video.mp4`
- `input/narration.txt`

验收：

- 无 FFmpeg/ffprobe 时给出明确错误。
- `project_manifest.json` 通过 schema 校验。
- 视频时长、fps、分辨率不是空值。

### 7.2 Stage 1：video-meta

输入：

- `project_manifest.json`
- `input/video.mp4`
- `--vlm-provider`

处理：

- 使用 PySceneDetect 或 fallback 均匀窗口生成视频窗口。
- 对每个窗口抽帧。
- 调用 VLM provider。
- 当前 MVP 先实现 `mock` provider：
  - 根据窗口时间生成稳定占位摘要。
  - 输出合法 physical tags / art tags / actions。

输出：

- `video/scenes.json`
- `video/frames/*.jpg`
- `meta/video_meta.ai.json`

验收：

- 没有真实 VLM 时也能生成合法 `video_meta.ai.json`。
- 所有窗口覆盖视频时长，且窗口不重叠、不倒序。
- 长镜头抽帧数量不超过配置上限。

### 7.3 Stage 2：audio

输入：

- `input/narration.txt`
- 可选 `input/pronunciation_rules.json`
- `--tts-provider`

处理：

- 文本分段，建议每段 `25-60` 个中文字符。
- 应用发音词典。
- 调用 TTS provider。
- 当前 MVP 先实现 `mock` provider：
  - 生成静音 WAV 或简单 tone WAV。
  - duration 可按文本长度估算后写入音频文件。
  - 但最终 `duration_sec` 必须读取 WAV 文件得到。

输出：

- `audio/segments/p001.wav`
- `audio/narration_audio_manifest.ai.json`

验收：

- 每个段落都有音频文件。
- `duration_sec` 与实际 WAV 文件一致。
- 单段重跑不会删除其他段落音频。

### 7.4 Stage 3：timeline

输入：

- `meta/video_meta.human.json` 或 `meta/video_meta.ai.json`
- `audio/narration_audio_manifest.human.json` 或 `audio/narration_audio_manifest.ai.json`

处理：

- 按段落顺序匹配连续视频窗口。
- 初版评分以顺序和时长为主。
- 如果音频时长和视频窗口差距小，使用 `none`。
- 如果差距在轻微变速范围内，使用 `slight_speed`。
- 如果差距过大，标记 `needs_review`。

基础评分建议：

```text
score =
  0.45 * order_score
+ 0.40 * duration_score
+ 0.10 * physical_tag_score
+ 0.05 * art_tag_score
```

MVP 阶段如果没有 embedding，`physical_tag_score` 和 `art_tag_score` 可以用简单 token overlap 或固定默认值。

输出：

- `timeline/timeline_plan.ai.json`

验收：

- 所有音频段都出现在 timeline 中。
- audio 时间轴连续递增。
- video 时间轴默认不倒序。
- 超过阈值的段落标记为 `needs_review`。

### 7.5 Stage 4：render

输入：

- `input/video.mp4`
- `audio/segments/*.wav`
- `timeline/timeline_plan.human.json` 或 `timeline/timeline_plan.ai.json`

处理：

- 拼接旁白音频。
- 根据 timeline 切片或重定时视频。
- 使用 FFmpeg 合成音视频。
- 硬件编码失败时 fallback 到 `libx264`。

输出：

- `render/final_video.mp4`
- `render/render_log.json`

验收：

- 输出视频存在且可被 `ffprobe` 读取。
- 输出时长接近总音频时长。
- FFmpeg 失败时保留完整命令和 stderr。

## 8. Provider 开发接口

### 8.1 VLM provider

```python
class VLMProvider:
    name: str

    def describe_window(self, window: dict, frame_paths: list[str]) -> dict:
        ...
```

返回字段必须能合并进 `video_meta.ai.json` 的 window：

```python
{
    "visual_summary": "...",
    "physical_tags": [],
    "art_tags": [],
    "visible_text": [],
    "actions": []
}
```

实现顺序：

1. `MockVLMProvider`
2. `Qwen3LocalVLMProvider`，P1 再做
3. 云端 fallback provider，可选

### 8.2 TTS provider

```python
class TTSProvider:
    name: str

    def synthesize(self, text: str, output_path: str, voice_config: dict | None = None) -> dict:
        ...
```

返回：

```python
{
    "audio_path": "audio/segments/p001.wav",
    "provider": "mock",
    "metadata": {}
}
```

实现顺序：

1. `MockTTSProvider`
2. `VoxCPM2Provider`，P3 再做
3. 其他备用 provider，可选

## 9. Retiming 规则

不要默认长时间尾帧定格。

推荐策略：

```text
audio_duration ~= video_duration:
  retime_strategy = none

0.90 <= video_duration / audio_duration <= 1.10:
  retime_strategy = slight_speed

可以合并相邻窗口:
  retime_strategy = merge_windows

缺口 <= 0.8s 且存在低运动帧:
  retime_strategy = short_freeze

其他:
  retime_strategy = needs_review
```

MVP 阶段可以先只实现：

- `none`
- `slight_speed`
- `needs_review`

`short_freeze` 和 B-roll 后续再做。

## 10. Schema 校验

Claude Code 应优先实现 schema 校验工具。

建议：

- 使用 Pydantic 或 dataclasses + 手写校验。
- 每个 stage 写文件前校验。
- 每个 stage 读取上游文件后校验。

需要校验的点：

- 必填字段。
- 类型。
- 时间字段非负。
- start < end。
- score 范围。
- review_status 枚举。
- 文件路径是否存在。

测试要求：

- `test_schemas.py` 覆盖合法与非法样例。
- schema 错误信息要指出字段路径。

## 11. 测试计划

### 11.1 单元测试

必须覆盖：

- 文本分段。
- 发音词典替换。
- JSON schema 校验。
- timeline 时长匹配。
- `.human.json` 优先读取逻辑。

### 11.2 集成测试

准备一个最小 fixture：

- 5-10 秒测试视频。
- 2-3 句旁白。

测试：

```bash
python -m narrateflow.cli run --project-dir tmp/demo --video tests/fixtures/demo.mp4 --script tests/fixtures/narration.txt --vlm-provider mock --tts-provider mock
```

验收：

- 生成所有中间 JSON。
- 生成所有音频段。
- 生成 `render/final_video.mp4`。

### 11.3 外部依赖测试

启动时检查：

- `ffmpeg` 是否可用。
- `ffprobe` 是否可用。
- PySceneDetect 是否安装；未安装时使用 fallback 均匀窗口。

## 12. 开发里程碑

### Milestone 1：数据契约与 CLI 骨架

交付：

- `schemas.py`
- `paths.py`
- `cli.py`
- project dir 初始化。
- `prepare` stage。

验收：

- 能生成合法 `project_manifest.json`。
- schema 测试通过。

### Milestone 2：视频窗口与 mock VLM

交付：

- `video/probe.py`
- `video/scenes.py`
- `video/frames.py`
- `providers/vlm_mock.py`
- `stage1_video_meta.py`

验收：

- 能生成合法 `video_meta.ai.json`。
- 窗口覆盖完整视频。

### Milestone 3：文本分段与 mock TTS

交付：

- `text/segmenter.py`
- `text/pronunciation.py`
- `audio/duration.py`
- `providers/tts_mock.py`
- `stage2_audio.py`

验收：

- 能生成分段 WAV。
- 能生成合法 `narration_audio_manifest.ai.json`。

### Milestone 4：timeline planner

交付：

- `timeline/scoring.py`
- `timeline/planner.py`
- `timeline/retime.py`
- `stage3_timeline.py`

验收：

- 能生成合法 `timeline_plan.ai.json`。
- 音频段全部匹配。
- 超阈值段落进入 `needs_review`。

### Milestone 5：FFmpeg render

交付：

- `render/ffmpeg.py`
- `stage4_render.py`
- `run` 全流程命令。

验收：

- mock 全流程可输出 `final_video.mp4`。
- render 失败时写入 `render_log.json`。

### Milestone 6：真实 provider 接入准备

交付：

- `vlm_qwen3_local.py` 接口壳。
- `tts_voxcpm2.py` 接口壳。
- provider 配置与错误处理。
- 显存/耗时 metrics 记录。

验收：

- 未安装真实模型时不影响 mock pipeline。
- provider 错误信息清楚。

## 13. Claude Code 执行顺序建议

Claude Code 开始开发时，按下面顺序执行：

1. 先读本文件，不要先接模型。
2. 建立 package / 模块骨架。
3. 实现 schema 和路径工具。
4. 实现 `prepare`。
5. 实现 `video-meta --vlm-provider mock`。
6. 实现 `audio --tts-provider mock`。
7. 实现 `timeline`。
8. 实现 `render`。
9. 跑通 mock end-to-end。
10. 再决定是否接入 Qwen3-VL 或 VoxCPM2。

## 14. 后续增强路线

P1：本地 Qwen3-VL provider

- 控制抽帧预算。
- 限制输出 token。
- 结构化 JSON 输出。
- 局部失败重试。
- 记录显存峰值和耗时。

P2：对齐质量增强

- embedding 语义匹配。
- OCR 加权。
- action time hint 只作为候选。
- forced alignment。
- short freeze。
- B-roll 补画面。

P3：真实 TTS provider

- VoxCPM2 压测。
- 发音词典完善。
- 单段重跑。
- 音频上下文承接。

P4：审核与数据飞轮

- Review UI 或 CLI review。
- 评测集归档。
- few-shot 池。
- 规则库。
- 微调实验。

## 15. 成功标准

P0-MVP 成功标准：

- 使用 mock provider 可以完整跑通全流程。
- 每个 stage 都可单独重跑。
- 中间 JSON 可人工修改后继续下游。
- 输出视频存在且可播放。
- 没有真实模型时系统仍可开发、测试和回归。

V3 工程成功标准：

- 接入真实 VLM/TTS 后，模型失败不会破坏流水线。
- 大模型输出不稳定时，schema、人工审核和 fallback 能兜住。
- 系统逐步从“能出片”演进到“少修改、质量稳定出片”。
