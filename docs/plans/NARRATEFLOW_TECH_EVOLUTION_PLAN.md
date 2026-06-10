# NarrateFlow 视频配音技术演进方案

版本：V3 工程收敛版  
日期：2026-06-08  
定位：在原 V2 方案基础上，保留“三阶段流水线 + 本地优先 + Human-in-the-loop”的核心方向，同时去掉不可验证或承诺过满的假设，改为可测试、可回退、可逐步演进的工程路线。

## 1. 目标与边界

NarrateFlow 面向“无声视频 + 已准备好的旁白文稿”场景，目标是在不做口型重绘的前提下，生成节奏自然、语义基本匹配、可人工快速修正的视频配音成片。

核心目标：

- 输入：无声视频 `.mp4` 与旁白文稿 `.txt` / `spoken.json`。
- 输出：带 AI 旁白音轨的成片视频。
- 支持人工审核：视觉理解结果、旁白音频、时间轴对齐均可单独检查与重跑。
- 本地优先：优先支持 RTX 3060 12GB 与 Mac M4 Pro 24GB，但保留云端 VLM 作为 baseline / fallback。

非目标：

- 不做口型对齐、Wav2Lip、视频重绘。
- 不承诺模型自动给出帧级或毫秒级动作真值。
- 不把“艺术意境标签”作为唯一对齐依据。
- 不在 MVP 阶段承诺本地微调或全自动闭眼出片。

## 2. 对 V2 方案的关键修正

V2 方案中有价值的部分包括：分阶段显存隔离、视频理解和 TTS 解耦、`video_meta.json` 中间产物、FFmpeg 负责最终渲染、人工审核关卡、数据飞轮归档。

需要修正的部分如下：

| V2 表述 | 风险 | V3 修正 |
| --- | --- | --- |
| Qwen3-VL 可直接吐出精确动作秒数 | VLM 时间戳是模型估计，不是物理真值 | VLM 时间戳只作为候选；真实边界以 PySceneDetect、ffprobe、帧号和音频强制对齐为准 |
| 12G 显存绰绰有余 | 多帧输入、Thinking 输出和 KV cache 会产生峰值 | 加入帧数、分辨率、输出 token、显存水位硬限制，并配置降级路径 |
| 动作帧定格能极度平滑 | 定格过长会像卡顿 | 定格只作为兜底，优先镜头组合、轻微变速、B-roll 复用 |
| 意境匹配可消除字面代沟 | 主观标签不稳定，容易过拟合 | 意境标签改为软特征，综合时间顺序、物理标签、OCR、主体动作和 embedding |
| 正则提取最后一个 JSON 足够 | Thinking 输出可能混杂、截断或嵌套 | 使用结构化输出、schema 校验、局部重试；正则只做最后兜底 |
| 500 条数据即可微调质变 | VLM/TTS 微调门槛高，数据质量要求高 | MVP 只做 few-shot、规则库、评测集；微调放到后期研究路线 |

## 3. 总体架构

```text
输入资产
  ├── video.mp4
  └── narration.txt / spoken.json
        │
        ▼
Stage 1: 视频结构化理解
  ├── ffprobe 获取真实时长、帧率、分辨率
  ├── PySceneDetect / 关键帧采样构建物理窗口
  ├── VLM provider 生成视觉摘要、动作候选、OCR/实体标签
  └── 输出 understanding/video_understanding.json
        │
        ▼
人工关卡一：视觉与时间轴候选审核
        │
        ▼
Stage 2: 旁白音频生成
  ├── 文本分段与发音词典预处理
  ├── TTS 生成分段音频
  ├── 可选 forced alignment 生成词级/句级音频时间戳
  └── 输出 voice/segments_manifest.json
        │
        ▼
人工关卡二：配音听审
        │
        ▼
Stage 3: 时间轴对齐与视频合成
  ├── 综合评分匹配旁白段落与视频窗口
  ├── 滑窗组合镜头，必要时局部变速或短定格
  ├── FFmpeg 渲染与混音
  └── 输出 compose/page_composed.mp4
```

## 4. 数据契约

所有阶段必须通过稳定 JSON 中间产物交接，避免上游模型变化牵动下游逻辑。

### 4.1 `video_understanding.json`

建议字段：

```json
{
  "schema_version": "3.0",
  "video": {
    "duration_sec": 120.5,
    "fps": 30.0,
    "width": 1920,
    "height": 1080
  },
  "windows": [
    {
      "window_id": "w0001",
      "start_time": 0.0,
      "end_time": 4.2,
      "source": "scenedetect",
      "confidence": 0.92,
      "visual_summary": "办公室中一名程序员盯着屏幕思考",
      "physical_tags": ["office", "screen", "person"],
      "art_tags": ["focused", "pressure"],
      "visible_text": [],
      "actions": [
        {
          "label": "person looks at screen",
          "time_hint": 2.1,
          "confidence": 0.62,
          "source": "vlm_candidate"
        }
      ],
      "source_frames": [
        {
          "time": 2.1,
          "path": "understanding/keyframes/w0001_002.100.jpg"
        }
      ],
      "review_status": "pending"
    }
  ]
}
```

约束：

- `start_time`、`end_time` 必须来自真实视频时间轴。
- `actions[].time_hint` 是候选，不允许下游直接当真值。
- provider 可以追加字段，但下游只能依赖稳定字段。
- 所有人工修改应另存 `human_fixed_video_understanding.json`，不要覆盖原始模型输出。

### 4.2 `segments_manifest.json`

建议字段：

```json
{
  "schema_version": "3.0",
  "segments": [
    {
      "paragraph_index": 1,
      "text": "今天我们来看一个本地视频配音系统。",
      "audio_path": "voice/segments/p001.wav",
      "duration_sec": 4.35,
      "word_timestamps": [],
      "review_status": "approved"
    }
  ]
}
```

约束：

- `duration_sec` 必须来自实际音频文件检测结果，而不是 TTS 预估。
- forced alignment 输出可为空，但字段应预留。
- 发音修正记录进入 `user_dict.txt` 或 `pronunciation_rules.json`。

### 4.3 `timeline.final.json`

建议字段：

```json
{
  "schema_version": "3.0",
  "segments": [
    {
      "paragraph_index": 1,
      "audio_start": 0.0,
      "audio_end": 4.35,
      "video_start": 0.0,
      "video_end": 4.2,
      "retime_strategy": "slight_speed_or_short_freeze",
      "matched_window_ids": ["w0001"],
      "match_score": 0.81,
      "review_status": "pending"
    }
  ]
}
```

## 5. 模型与外部依赖策略

### 5.1 VLM provider 分层

第一层：`gemini` baseline  
当前仓库已经以 Gemini 作为视频理解主线。它应继续作为可用 baseline，用于验证数据契约和成片质量。

第二层：`local_qwen3_vl` experimental  
接入 Qwen3-VL-8B-Thinking 或同级本地 VLM，但必须满足：

- 单个窗口最多 8-12 帧。
- 单帧长边默认不超过 600px。
- 输出 token 上限固定。
- 显存峰值可记录。
- JSON 解析失败可局部重试。
- 本地结果可和 Gemini baseline 进行差异对比。

第三层：`mock` / `manual` provider  
用于无网络、无模型、回归测试和人工录入场景。

### 5.2 TTS provider 分层

第一层：沿用当前 Qwen-TTS  
与现有仓库兼容，优先保证 pipeline 跑通。

第二层：VoxCPM2 experimental  
VoxCPM2 可作为后续高音质本地 TTS 候选，但接入前必须完成：

- Windows CUDA 兼容性测试。
- Mac MPS / MLX 可用性验证。
- 长文本分段稳定性测试。
- 多音字、数字、英文缩写、专业术语测试。

第三层：云端或备用 TTS  
当本地模型失败时，保留可替代 provider，不让整条流水线阻塞。

## 6. 时间轴对齐算法

V3 不再采用“一句台词对一个镜头”的硬规则，而采用综合评分 + 单调约束 + 局部人工锁定。

### 6.1 匹配特征

每个旁白段落与视频窗口组合计算匹配分：

- 时间顺序分：是否符合原始视频顺序。
- 时长适配分：音频时长与窗口组合时长差距。
- 物理语义分：旁白与 `visual_summary`、`physical_tags` 的 embedding 相似度。
- OCR 分：旁白是否提到屏幕文字、标题、产品名。
- 动作候选分：旁白关键词是否命中 `actions`。
- 意境软分：`art_tags` 与旁白情绪是否接近。
- 人工锁定分：用户手动指定的窗口优先级最高。

意境标签只能作为加分项，不能覆盖时间顺序和物理窗口约束。

### 6.2 Retiming 策略优先级

当音频与视频窗口长度不一致时，按以下顺序处理：

1. 合并相邻短镜头。
2. 从中性 B-roll 池补画面。
3. 对视频做轻微变速，建议范围 `0.90x-1.10x`。
4. 在动作候选点或低运动帧短定格，建议单次不超过 `0.8s`。
5. 若仍无法对齐，标记为 `needs_review`，交给人工调整文稿或窗口。

禁止默认长时间尾帧定格。

### 6.3 强制音频对齐

如果引入 stable-ts / whisperx / force-align-wordstamps 类工具，输出只作为音频时间轴真值，用于判断每句、每词的实际结束点。

用途：

- 判断 TTS 实际语速。
- 辅助选择自然停顿点。
- 让画面切换尽量落在句末、逗号、语义停顿后。

## 7. 显存与进程管理

### 7.1 基本原则

- VLM 与 TTS 不在 RTX 3060 12GB 上同时常驻。
- 所有模型阶段独立进程运行。
- 进程退出后必须轮询显存水位，而不是只依赖 `process.kill()`。
- Mac M4 Pro 可探索常驻策略，但必须以压测结果为准。

### 7.2 RTX 3060 策略

- 默认串行：Stage 1 完成后释放 VLM，再启动 Stage 2。
- 显存安全水位：启动新模型前可用显存应高于配置阈值。
- 若 VLM 峰值超过阈值，自动降低帧数、分辨率或切换更小模型。
- 批处理时可采用“批量 Stage 1 -> 批量 Stage 2 -> 批量 Stage 3”，减少模型反复加载。

### 7.3 Mac M4 Pro 策略

- 不默认承诺全员常驻。
- 允许配置 `keep_alive` 策略：
  - `off`：每阶段释放，最稳。
  - `adaptive`：根据内存压力决定是否保留。
  - `always`：仅在压测通过后启用。

## 8. Human-in-the-loop 设计

人工关卡不是临时补丁，而是系统质量控制核心。

### 8.1 关卡一：视觉理解审查

用户检查：

- 镜头窗口是否正确。
- `visual_summary` 是否描述准确。
- `visible_text` 是否漏掉关键屏幕文字。
- `actions[].time_hint` 是否可作为候选。
- `art_tags` 是否明显跑偏。

输出：

- 原始模型输出：`ai_video_understanding.json`
- 人工修正版：`human_fixed_video_understanding.json`

### 8.2 关卡二：配音听审

用户检查：

- 多音字。
- 断句。
- 语速。
- 情绪。
- 数字、英文、产品名发音。

输出：

- 原始 TTS 音频。
- 审核通过音频。
- 发音词典增量。

### 8.3 关卡三：时间轴预览

用户检查：

- 每段旁白对应的视频窗口是否合理。
- 是否出现明显卡顿、长定格、突兀变速。
- 是否需要手动锁定某些段落起止时间。

## 9. 数据飞轮

V3 保留数据飞轮，但调整用途优先级。

第一优先级：评测集  
记录模型输出、人工修正、最终成片质量，用于比较不同 provider 和 prompt。

第二优先级：few-shot 池  
将高质量修正样例注入 prompt，提高同类视频的稳定性。

第三优先级：规则库  
沉淀发音词典、OCR 替换、术语映射、物理标签到意境标签的可解释映射。

第四优先级：微调数据  
当数据规模、质量、标注一致性都达标后，再考虑 VLM LoRA 或 TTS 风格微调。

推荐目录：

```text
<project-dir>/
  data_flywheel/
    raw_frames/
    ai_video_understanding.json
    human_fixed_video_understanding.json
    ai_tts_segments/
    human_approved_segments/
    final_timeline.json
    final_video.mp4
    review_notes.json
```

## 10. 技术演进路线

### P0：稳定现有 pipeline 和数据契约

目标：不替换当前 Gemini/Qwen-TTS 主线，先把中间产物稳定下来。

任务：

- 固化 `video_understanding.json`、`segments_manifest.json`、`timeline.final.json` schema。
- 增加 schema 校验与错误提示。
- 明确人工修正版文件命名。
- 保证 `compose` 不关心 VLM provider 来源。

验收：

- 当前 Gemini 主线不回退。
- 无模型情况下可通过 mock 数据跑通 timeline / compose。
- 人工修改 JSON 后可从下游阶段继续运行。

### P1：抽象 provider 与本地 VLM 实验接入

目标：让本地 Qwen3-VL 成为可比较的 experimental provider。

任务：

- 增加 `--vl-provider gemini|mock|local_qwen3_vl`。
- 本地 VLM 输出适配统一 schema。
- 加入帧数、分辨率、token、显存限制配置。
- 实现结构化输出解析、schema 校验、局部重试。

验收：

- 同一个视频可分别生成 Gemini 与本地 VLM 两份理解结果。
- 能输出差异报告：窗口摘要、动作候选、OCR、匹配分变化。
- 3060 上出现 OOM 前能自动降级或中止并给出明确原因。

### P2：提升时间轴匹配与 retiming 质量

目标：让成片从“能对上”提升到“看起来顺”。

任务：

- 实现综合评分匹配。
- 支持人工锁定窗口。
- 支持轻微变速、短定格、B-roll 补画面策略。
- 可选接入 forced alignment。

验收：

- 单句过长时不会默认长时间尾帧定格。
- 时间轴结果能解释每段匹配原因。
- 预览中可定位所有 `needs_review` 段落。

### P3：TTS provider 扩展与音频质量优化

目标：在现有 Qwen-TTS 稳定基础上实验 VoxCPM2 或其他高质量 TTS。

任务：

- 增加 `--tts-provider qwen_tts|voxcpm2|fallback`。
- 建立发音词典和文本预处理。
- 测试长文本切分、语速控制、情绪控制。
- 可选保留 prompt wav / 上下文承接能力。

验收：

- 每段音频实际时长可稳定记录。
- 读错词可通过词典修正并单独重跑 Stage 2。
- 新 provider 失败不影响旧 provider 使用。

### P4：产品化与数据飞轮

目标：降低人工审核成本，沉淀长期质量资产。

任务：

- 建立 review UI 或轻量 CLI 审核流程。
- 自动归档 AI 输出、人工修正、最终产物。
- few-shot 池自动注入。
- 建立小型评测集。

验收：

- 每次人工修正都可追踪。
- provider / prompt / TTS 版本变化后可复测历史样例。
- 常见风格视频的人工修改次数下降。

## 11. 风险与兜底

| 风险 | 表现 | 兜底 |
| --- | --- | --- |
| VLM 时间戳不准 | 动作候选与真实画面错位 | 只作为候选，最终依赖物理窗口和人工确认 |
| 本地 VLM OOM | 3060 推理卡死或 swap 暴涨 | 降帧、降分辨率、降输出 token、切换 Gemini |
| Thinking 输出不可解析 | JSON 截断、混入思考文本 | 结构化输出、schema 校验、局部重试、人工修正 |
| TTS 读错词 | 多音字、英文缩写错误 | 发音词典、单段重跑 |
| 长句无法塞入镜头 | 画面卡顿或定格过长 | 文稿压缩建议、镜头组合、B-roll、短定格、人工 review |
| 意境标签误导匹配 | 画面和文案情绪贴近但内容不符 | 意境只做软特征，物理标签和时间顺序优先 |
| 外部项目能力不匹配 | 参考项目无法直接集成 | 只借鉴设计，先做 adapter 和数据契约 |

## 12. 推荐立即执行的改造顺序

1. 固化 JSON schema，并把人工修正版产物纳入 pipeline。
2. 增加 mock provider，验证下游完全不依赖 Gemini。
3. 改造 timeline 匹配评分，让意境从主依据降为软特征。
4. 增加 retiming 策略限制，禁止长时间默认尾帧定格。
5. 接入本地 Qwen3-VL experimental provider，并做 3060 / Mac 压测。
6. 再评估 VoxCPM2 是否替换或补充当前 Qwen-TTS。

## 13. 最终判断

优化后的方案是可落地的，但前提是把大模型从“自动决策者”降级为“候选信息提供者”。真实数据和工程约束应由 ffprobe、PySceneDetect、音频时长检测、forced alignment、schema 校验和人工审核共同保证。

换句话说，NarrateFlow 的核心竞争力不应是“相信某个模型一次性完美理解视频”，而应是“把不稳定的模型输出纳入稳定、可验证、可回退的音视频生产流水线”。
