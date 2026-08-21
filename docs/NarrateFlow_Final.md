# NarrateFlow 视频配音系统 — 最终技术方案

版本：Final V2.0 | 日期：2026-06-11

---

## 1. 项目概要

**输入**：无声视频 + 旁白文稿  
**输出**：带旁白音轨的成片视频  
**运行环境**：本地单机，Mac M4 Pro（24G）或 RTX 3060（12G）  
**明确不做**：口型对齐、画面重绘、全自动闭眼出片

### 1.1 V2 原方案的六项修正

评审文档指出 V2 中以下内容不切实际，本方案已全部修正：

| 修正项 | V2 原问题 | 修正后 |
|--------|----------|--------|
| 时间戳 | 认为 VLM 能"吐出精确秒数"直接用于剪辑 | VLM 动作时间只作为 time_hint 候选；真实时间轴来自 PySceneDetect + ffprobe |
| 显存 | "12G 永远绰绰有余" | 每镜头 ≤12 帧，单帧 ≤600px，Ollama 硬锁 num_ctx=4096；3060 上 VLM/TTS 串行 |
| 定格 | 以"动作帧定格"为主策略 | 合并镜头 > B-roll > 轻微变速 > 短定格 ≤0.8s > 人工 review |
| 意境 | 以"艺术意境"为主要匹配依据 | 意境权重仅 5%，以时序(25%)+时长(25%)+物理标签(10%)+OCR(10%)为主 |
| 解析 | "正则提取 JSON 即可" | 优先 JSON mode / structured output，正则仅兜底，必须做 schema 校验 |
| 微调 | "500 条数据即可微调质变" | 远期目标：先建评测集 → few-shot 池 → 规则库，条件成熟后再启动 |

---

## 2. 总体流程

```
Stage 0: 资产探测           → project_manifest.json
Stage 1: 视频理解（VLM）    → video_meta.json           [关卡一: 视觉审核]
Stage 2: 旁白生成（TTS）    → 音频 + manifest            [关卡二: 配音听审]
Stage 3: 时间轴对齐         → timeline_plan.json         [关卡三: 时间轴审视]
Stage 4: FFmpeg 合成       → final_video.mp4 + 质检
```

- Stage 1/2 各独占一个模型，完成即释放
- 所有数据通过 JSON 文件传递，AI 输出 *.ai.json，人工修正 *.human.json，读取优先 .human
- 支持从任意 Stage 重新执行

### 2.2 参考代码目录结构

```
narrateflow/
├── main.py                     # 总控脚本
├── config.yaml                 # 全局配置（模型路径、对齐权重、阈值等）
├── requirements.txt
│
├── stage0_manifest.py          # Stage 0: 资产探测与文稿解析
├── stage1_vision.py            # Stage 1: 视频结构化理解
├── stage2_tts.py               # Stage 2: 旁白音频生成
├── stage3_align.py             # Stage 3: 时间轴对齐
├── stage4_render.py            # Stage 4: FFmpeg 渲染
│
├── providers/                  # 模型 Provider（与具体引擎交互的核心模块）
│   ├── __init__.py
│   ├── vlm_base.py             # VLM 抽象接口
│   ├── vlm_ollama.py           # Ollama（Qwen3-VL）实现
│   ├── tts_base.py             # TTS 抽象接口
│   └── tts_voxcpm2.py          # VoxCPM2 实现
│
├── utils/                      # 工具函数（纯逻辑，不依赖模型）
│   ├── __init__.py
│   ├── ffprobe.py              # ffprobe 命令封装
│   ├── audio.py                # 音频时长读取（wave/ffprobe/librosa）
│   ├── schema_validator.py     # JSON Schema 校验 + 字段范围检查
│   └── vram_guard.py           # 显存监控 + 自动降级
│
├── schemas/                    # JSON Schema 定义文件
│   ├── project_manifest.json
│   ├── video_meta.json
│   ├── narration_audio_manifest.json
│   └── timeline_plan.json
│
├── prompts/                    # Prompt 模板
│   └── vlm_system.txt
│
├── dicts/                      # 词典
│   └── pronunciation.json
│
├── input/                      # 原始素材
│   ├── video.mp4
│   └── narration.txt
│
├── output/                     # 中间产物和成片（按项目 ID 组织）
│   └── {project_id}/
│       ├── project_manifest.json
│       ├── frames/
│       ├── video_meta.ai.json
│       ├── video_meta.human.json
│       ├── audio/
│       ├── narration_audio_manifest.json
│       ├── timeline_plan.ai.json
│       ├── timeline_plan.human.json
│       ├── final_video.mp4
│       ├── qc_report.json
│       └── run_metrics.json
│
└── tests/                      # 测试
    ├── test_stage0.py
    ├── test_stage1.py
    ├── test_stage2.py
    ├── test_stage3.py
    └── test_stage4.py
```


## 3. Stage 0 — 资产探测与文稿解析

**目标**：获取视频真实元信息，将文稿拆成可逐段处理的段落。

**要做的事**：
1. 用 ffprobe 读取视频的 duration、fps、分辨率、编码。**所有数值必须来自 ffprobe 实测，不可用任何方式估算**
2. 读取旁白文稿，按标点分句并编号。每段 25-60 个中文字符
3. 文稿规范化：数字转中文、英文缩写查发音词典。保留 original_text 和 tts_text 两个字段
4. 输出 project_manifest.json

**产出**：`project_manifest.json`（含视频元信息 + 规范化段落列表）

---

## 4. Stage 1 — 视频结构化理解（VLM 独占）

**目标**：把视频切成带语义标签的时间窗口，为 Stage 3 对齐提供素材。

**要点**：本阶段独占 VLM，结束后必须彻底释放模型（OLLAMA_KEEP_ALIVE=0）

**要做的事**：

### 4.1 物理切片
- 用 PySceneDetect 检测镜头边界，输出窗口列表。每个窗口的 start_time / end_time 是物理真值
- 长镜头（>30s）做二级切分，录屏/PPT 类视频附加帧差检测

### 4.2 抽帧（硬性约束）
- 短镜头 ≤5s：中间帧或首中尾三帧
- 长镜头 >5s：均匀抽取，**硬上限 12 帧，长边统一缩放到 600px**
- 动态帧差：画面静止只抽 1 帧，剧烈变化才多抽。目的：消灭 80% 重复画面

### 4.3 VLM 模型与部署

选用 **Qwen3-VL-8B-Thinking**，通过 Ollama 本地部署：

| 平台 | 量化方案 | 权重占用 | 配置 |
|------|----------|----------|------|
| RTX 3060 12G | Q6_K（6 位混合精度） | ~6.9 GB | num_ctx=4096, num_predict=512 |
| Mac M4 Pro 24G | Q8_0（8 位量化） | ~5.5 GB | 可尝试放宽 num_ctx 限制 |

**设计 System Prompt 时关注以下要点**（参考 TimeChat-Captioner 的"六维剧本式打标 Schema"，让模型像导演一样描述画面）：
- 要求输出：视觉摘要、物理标签、意境标签、可见文字、动作候选（含 time_hint 和 confidence）、不确定性说明
- 强制只输出 JSON，不使用 markdown 包裹
- 限制 Thinking 链条长度，防止推理耗时过长

### 4.4 VLM 推理
- 将每个窗口的帧序列 + System Prompt 发给 Ollama
- 3060：每窗口推理前检查显存水位，超阈值则触发降级（详见第 8 章）
- Mac：内存宽裕，可保持 VLM 常驻

### 4.5 输出解析
- 优先用 JSON mode / structured output
- 解析失败：去掉 <thinking> 块 → 正则提取 JSON → 标记 needs_review
- 做 schema 校验：字段类型、必填项、confidence ∈ [0,1]
- 失败只重跑当前窗口，不阻塞后续

**产出**：`video_meta.ai.json` + `frames/`  
**人工关卡**：审阅 video_meta，修正错误，保存为 video_meta.human.json

---

## 5. Stage 2 — 旁白音频生成（TTS 独占）

**目标**：逐段生成配音，记录每段的真实音频时长。

**启动前**：确认 VLM 已释放，显存降到安全水位

**要做的事**：

### 5.1 文本准备
- 加载发音词典（多音字、品牌名、专业术语、英文缩写）
- 对每段文稿做发音词典替换，生成 tts_text

### 5.2 TTS 合成
- 模型：**VoxCPM2**（2B 参数），48kHz 输出
- 3060：使用 BF16（Ampere 架构原生加速，显存约 4G）
- Mac：使用 FP16（Mac MPS 后端禁止使用 BF16，会导致报错或降级为 CPU 运行）
- 每段合成后立即用 wave/ffprobe 读取真实时长。**禁止用字数估算**

### 5.3 模型压测（接入前必须完成）

VoxCPM2 在正式接入流水线前，需要完成以下压测验证：

| 验证项 | 标准 |
|--------|------|
| 中文旁白自然度 | 听感自然，无明显机械感 |
| 48kHz 输出稳定性 | 不降采样、不走调 |
| 长文本分段一致性 | 多段合成后音色和语气保持一致 |
| 数字/英文/专业名词/多音字 | 发音正确率 ≥ 95% |
| Windows CUDA（3060） | BF16 不崩溃、不 OOM |
| Mac MPS（M4 Pro） | FP16 可运行，确认不使用 BF16 |
| 单段重跑速度 | 单段合成耗时 < 音频时长 × 3 |
| 显存占用（独立运行） | 3060 端 < 4GB |

### 5.4 单段重跑
- 人工标记 regenerate 后只重跑该段，其他段不受影响
- 发音词典更新后自动重跑受影响的段落

**产出**：`audio/*.wav` + `narration_audio_manifest.json`  
**人工关卡**：听审音频，标记问题段落，更新发音词典，重跑

---

## 6. Stage 3 — 时间轴对齐（纯 CPU）

**目标**：把旁白段落匹配到视频窗口，决定每段的画面处理方式。

**要做的事**：

### 6.1 匹配评分（权重来自评审修正）

| 维度 | 权重 | 含义 |
|------|------|------|
| 顺序匹配 | 25% | 保持视频原始播放顺序 |
| 时长匹配 | 25% | 音频时长与窗口时长的接近程度 |
| 语义相似 | 20% | 旁白文本与视觉摘要的语义相似度 |
| 物理标签 | 10% | 旁白关键词与画面物理标签的重叠 |
| OCR 匹配 | 10% | 旁白提及的词是否出现在画面文字中 |
| 动作匹配 | 5% | 旁白动作描述与检测动作的匹配 |
| 意境匹配 | 5% | 情绪/氛围匹配（权重最低） |

权重可通过 config.yaml 调整。人工锁定点拥有最高优先级。

### 6.2 滑窗组合
- 一个段落可匹配 1-3 个连续窗口，保持原始播放顺序
- 低分匹配（<0.4）标记 needs_review

### 6.3 重定时策略（优先级降序）

| 优先级 | 策略 | 触发条件 |
|--------|------|----------|
| 1 | 原速播放 | 音频与视频时长差 ≤ 0.5s |
| 2 | 合并镜头 | 音频更长时，滑窗阶段已处理 |
| 3 | 轻微变速 | 视频更长时压缩 ≤10%，即 0.90x-1.10x |
| 4 | B-roll 补画面 | 音频更长且差值 ≤ 3s |
| 5 | 短定格 | 音频更长且差值 ≤ 0.8s，选低运动帧/动作完成点 |
| 6 | 人工审核 | 超出以上范围 |

定格选点原则：优先低运动帧 → 优先动作完成点 → 优先句子停顿点 → **禁止默认用最后一帧**

**产出**：`timeline_plan.ai.json`  
**人工关卡**：审视对齐方案，修正后保存为 timeline_plan.human.json，直接进 Stage 4

---

## 7. Stage 4 — FFmpeg 渲染合成

**目标**：根据对齐方案生成 FFmpeg 命令，渲染输出并质检。

**要做的事**：
1. 读取 timeline_plan（优先 .human.json）
2. 按每段的 retime_strategy 生成 FFmpeg 滤镜（trim/setpts/loop/concat）
3. 音频拼接混音
4. 编码：Windows h264_nvenc，macOS h264_videotoolbox，兜底 libx264
5. 自动质检：时长校验（偏差 ≤2s）、黑帧检测、静音检测

**产出**：`final_video.mp4` + `qc_report.json`

---

## 8. 显存管理

### 8.1 两套运行策略

| 硬件 | 策略 | 做法 |
|------|------|------|
| RTX 3060 12G | 批次换岗 | VLM 跑完 → 释放（OLLAMA_KEEP_ALIVE=0）→ 轮询确认显存安全 → 启动 TTS |
| Mac M4 Pro 24G | 全员常驻 | VLM + TTS 可同时驻留（统一内存约 <10G），但仍保留 `release_after_stage` 配置作为安全机制。**不默认认为统一内存一定安全** |

总控通过 `torch.cuda.is_available()` 自动选择路线。

**跨平台注意事项**：
- 3060：进程结束后显存释放有 1-2 秒延迟（NVML 驱动回收慢），必须用 pynvml 轮询确认
- Mac：统一内存由系统托管，进程死后可能残留 Page Cache，必须用 psutil 确认物理内存真正释放

### 8.2 降级链条（3060 超阈值时）

降帧数（12→8→5→3）→ 降分辨率（600px→480px→360px）→ 切分窗口 → 换更小模型 → 切换云端

### 8.3 运行记录

每次任务后写入 run_metrics.json：各 Stage 耗时、显存峰值、解析失败次数、人工修改次数。

---

## 9. 技术路线图

### P0 — MVP（2-3 周）
目标：端到端链路跑通。VLM 可先用云端 API（如 GPT-4V）快速验证流程，TTS 暂用任意可用方案先跑通。对齐仅靠顺序+时长匹配。  
验收：3 分钟视频完整跑出配音 mp4。

### P1 — 本地 VLM（2-3 周）
- 3060：部署 Qwen3-VL-8B Q6_K，硬锁 num_ctx=4096 / num_predict=512，抽帧预算控制
- Mac：部署 Qwen3-VL-8B Q8_0
- 共通的：动态帧差抽帧、JSON 解析容错、schema 校验、显存监控与自动降级  
验收：3060 上不 OOM 或能自动降级；本地 VLM 输出可被 Stage 3 正确消费。

### P2 — 对齐质量（2-3 周）
7 维评分，滑窗组合，变速/B-roll/定格优化。  
验收：长句不再尾帧长定格。

### P3 — TTS 质量（2-3 周）
- 3060：VoxCPM2 BF16 部署；Mac：VoxCPM2 FP16 部署
- 完成压测验证清单（详见 5.3 节），发音词典 ≥ 50 条目
- 单段重跑、批处理优化、长文本分段语气连贯性  
验收：读错词可通过词典稳定修正；单段重跑不影响其他段。

### P4 — 持续优化
审核 UI、评测集、few-shot 池、规则库。微调条件满足后启动。

---

## 10. 风险与兜底

| 风险 | 兜底 |
|------|------|
| VLM JSON 解析失败 | 多级容错 → 标记 needs_review → 不阻断后续 |
| 3060 OOM | 自动降级链条 |
| TTS 读错词 | 发音词典 + 单段重跑 |
| 文稿长于视频素材 | 压缩建议 + B-roll + 人工 review |
| FFmpeg 编码失败 | 硬编失败切软编，重试 3 次 |
| Mac/PC 不一致 | 各 Stage 内做平台判断 |

---

## 附录 — 技术选型理由

**A.1 为什么用 PySceneDetect 切片 + 抽帧而非 Video Caption**：配音对齐需要精确到帧的绝对时间戳，不是笼统的视频描述。PySceneDetect 精确切出镜头边界后，每段只需 1-3 帧即可识别内容。5 分钟视频仅处理 50-150 张图，Video Caption 则是天文数字的 Token 消耗。

**A.2 为什么选 Qwen3-VL-8B**：Interleaved-MRoPE 三维空间融合捕获帧间运动。Thinking 模式输出文学级意境。Q6_K（~6.9G）可在 3060 跑。

**A.3 为什么 VoxCPM2 是首选 TTS**：2B/4.5G/48kHz，Mac MPS+PC CUDA+BF16 原生兼容。

**A.4 参考项目**：VisionCaptioner（审核 GUI）、simple-captioner（Headless 脚本）、Subdub（Stage 模块化总控）、AutoDub（FFmpeg 混音）、force-align-wordstamps（词级时间戳）、TimeChat-Captioner（六维打标 Schema）。

---

*版本 Final V2.0 | 2026-06-11*
