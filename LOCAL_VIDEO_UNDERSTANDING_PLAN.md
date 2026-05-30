# 📄 NarrateFlow 工业级系统设计方案书

## 一、 项目定位与硬核技术底座

NarrateFlow 是一款面向硬核技术演示、代码运行及 SaaS 录屏的通用视频全自动配音与非线性重定时（Retiming）桌面端软件。它通过全栈量化、内存分页虚拟化、跨进程熔断级联架构，攻克了长视频多模态推理中“KV Cache 溢出、OCR 过载、显存碎片化崩溃”三大工业级死穴，在 3060 显卡上实现商业级的极致声画卡点。

## 二、 核心技术优化方案（量化、KV Cache 与 vLLM 垂直调优）

### 1. 多模态全栈量化策略（VRAM 瘦身）

为了让 4B 视觉模型、Qwen-TTS、Whisper-X 及文本反思模型在 12G 显存内和平共处，必须对权重进行精细化量化剪裁：

- **视觉与文本 LLM 层**： 放弃标准的 FP16 推理，全面采用 AWQ（Activation-aware Weight Quantization）或 GPTQ 4-bit 量化。量化后的 Qwen3.5-4B 视觉模型权重文件直降至 3.2GB 左右。

    - 技术优势： AWQ 在保持量化精度的同时，相比大块头的 GGUF，在 Linux/Windows 的 CUDA 环境下具备极强的高吞吐推理吞吐量。

- **语音与对齐模型层**： 接入经过 CTranslate2 深度优化的 Faster-Whisper，使用 INT8 / FLOAT16 混合量化。其常驻显存仅需 350MB，解码速度提升 4 倍以上。

### 2. 基于 vLLM 的 KV Cache 动态管理优化（根治长视频 VRAM 溢出）

录屏长视频伴随着海量的视频帧（Tokens 输入），其长上下文（Long Context）会导致大模型的 KV Cache 呈几何级数爆炸，这是 3060 爆显存的头号杀手。

- **调优策略一**：PagedAttention 虚拟内存管理

    - 引入 vLLM 的核心思想，将 KV Cache 存储在不连续的物理显存块中（类似于操作系统的虚拟内存分页）。彻底消除显存碎片化（Memory Fragmentation），将显存利用率无限逼近 96% 以上。

- **调优策略二**：自适应 KV Cache 驱逐与流式上下文滚动（Cache Eviction）

    - 技术录屏存在极强的局部相关性。算法采用动态滑窗机制（Sliding Window Attention）：当大模型观看完视频前 1 分钟的代码演示并吐出对应的剧本文本后，后台立刻释放这 1 分钟对应的视频特征 KV Cache，仅保留最新 15 秒的“语义锚点”作为上下文提示。通过“边看、边忘、边写”的流水线，将推理期间的 KV Cache 强行控制在 2GB 阈值以内。

### 3. 跨进程显存熔断机制（VRAM Guard）

哪怕经过量化和 vLLM 调优，多模型并发依然有物理风险。NarrateFlow 采用进程级硬隔离架构：

1. 视觉大模型（AWQ 4-bit）利用 vLLM 推理完毕，将 JSON 剧本序列化存入本地 SQLite。

2. 主控脚本直接执行 os.kill(vllm_pid, 9) 强行物理毁灭该进程。100% 显存秒级归还系统。

3. 随后干净、空仓地拉起 Qwen-TTS（CosyVoice）和 Whisper-X 进行音频合成，从物理底层断绝了爆显存的可能性。

## 三、 前后端算法与论文工程落地

### 1. 前端视觉级联切片引擎

结合学术界 《Token-Efficient Video Understanding for High-Density Text Screens》(CVPR) 的降维思想，拒绝盲目全屏 OCR：

- **L1 pHash 粗筛（CPU 密集型）**： 利用感知哈希算法过滤 90% 的连续滚动日志帧与静态死寂帧，每秒跑上百帧，不占 1MB 显存。

- **L2 局部 ROI OCR 比对（轻量级）**： 结合 《ScreenParsing》 论文，仅切片扫描屏幕顶部 15%（软件标题/文件名）和左侧 15%（项目结构树）。当区域内字符编辑距离发生突变（Jaccard 相似度 < 0.4）时，才判定为“业务场景彻底切换”，触发精准视频分镜切片点。

### 2. 后端音时滑动窗口与智能体字数反思

结合 《Aligning Audio-Visual... with an Agentic Workflow》 的多模态对齐思想：

- **声音连贯性调优**： 连续流式生成音频 Chunk（每段 35~50 字）时，将上一句尾部的 8 个字和 4 秒音频作为 Prompt 滚动喂给 Qwen-TTS，利用流匹配声学解码器完美承接呼吸惯性和麦克风底噪。

- **智能体字数反思（Agentic Constraint）**： 当 Whisper-X 强制对齐算子计算出“人类剧本字数过于膨胀、无法塞进当期视频片段”时，触发轻量文本模型的自反思机制（Self-Reflective Compression），在不改变技术核心原意的前提下，强制精简字数至 物理秒数 × 4.2 字以内，随后调用 FFmpeg 局部静态冻结（Smart Frame Freezing）算法微调画面节奏，实现完美卡点。

## 三、 工业级风险控制与兜底预案

1. 风险：AWQ 4-bit 量化大模型对技术录屏中的极微小 OCR 字符（如代码中的一个分号 ;）理解出现严重退化或幻觉。

    - 预案： 立即启用级联切片中的 L2 局部 ROI OCR。不要让大模型去硬看高清视频帧来猜字符，而是将 L2 OCR 提取到的高精纯文本字符串，作为 Text-Context-Embedding 以硬编码的形式动态注入到 Qwen3.5-4B 的 Prompt 提示词中（即“借尸还魂”策略），强行纠正量化模型的幻觉。

2. 风险：vLLM 的 PagedAttention 在 Windows 原生环境（非 WSL2）下由于 CUDA 算子编译冲突导致桌面端启动失败。

    - 预案： 在 Windows 端的打包分支中，如果 vLLM 核心算子跑不通，自动降级降维为基于 OnnxRuntime-GenAI 或 llama.cpp 的 4-bit 量化推理引擎。虽然牺牲了微量的大并发吞吐量，但由于我们的 Pipeline 本身就是串行视频块消费，其单批次推理速度依然能够保持在较高水准，从而 100% 确保 Windows 桌面客户端的绝对稳定性。

## 四、结合已有项目的重构与工业化增强

### 1. 核心架构补丁：从“串行脚本”到“状态机流水线”

你目前的仓库可能倾向于脚本化执行，为了实现“路线A（精准排版）”和“路线B（AI自动）”，你需要重构核心流程为事件驱动状态机。

- 重构输入层 (Input Layer)：
    - 工作量： 编写 text_processor.py。
    - 优化点： 增加“智能切片适配器”，统一处理用户输入（一坨长文 vs 已分段文本）。实现基于字数窗口（35-50字/Chunk）的聚类逻辑，确保送入 TTS 的文本永远处于“最佳拟真长度”。
- 重构推理调度层 (Orchestrator)：
    - 工作量： 实现基于 ProcessPoolExecutor 的多进程管理。
    - 优化点： 这是处理 Qwen3.5-4B (视觉模型) 和 Qwen-TTS (语音模型) 显存竞争的核心。你需要一个全局协调器，在视觉模型跑完 JSON 剧本后，通过 os.kill() 彻底回收显存。

### 2. 性能与效果优化点 (Performance & Quality Tuning)

这是让你的工具从“能用”变成“好用”的关键，也是 3060 显存的生命线。

1. 量化与 KV Cache 优化 (重中之重)：
    - 工作量： 在推理加载环节引入 vLLM 的 PagedAttention 配置或者 AutoAWQ 加载器。
    - 优化点： 显存水位控制在 2GB 以内。确保在处理 10 分钟以上的录屏视频时，KV Cache 不会吃满 12G。

2. 视觉感知切片级联引擎 (Visual Segmentation)：
    - 工作量： 在仓库增加 vision_seg.py 模块。
    - 优化点： 现在的录屏视频切片如果是全自动的，很容易误切。必须加入 pHash (感知哈希) 过滤掉那些疯狂跳动的日志日志窗口，只在界面宏观布局发生跳变（如窗口切换、PPT翻页）时切片。

3. 音频情感连贯性 (Audio Continuity)：
    - 工作量： 在 tts_engine.py 中引入“上一段音频尾部滚动滑窗”。
    - 优化点： 实现 Qwen-TTS 的动态 prompt_wav 注入，解决多段音频拼接时的机械断层感。
    
### 3. 详细工作量拆解表

我将这些优化任务分为了三个等级，你可以根据你目前的时间精力进行排期：

| 优先级 | 任务模块 | 具体内容与优化目标 | 预估工时 |
| --- | --- | --- | --- | 
| P0 | 音频卡点对齐 (Whisper-X) | 在仓库中集成 whisperx。实现强制对齐模式，确保每一段音频都有精确到毫秒的时间轴。| 2天 |
| P0 | 显存安全管理 (Process Isolation) | 将视觉分析和 TTS 合成隔离为两个独立进程，实现视觉任务完成后物理销毁进程以回收显存。| 2天 |
| P1 | 视频重定时 (FFmpeg Retiming) | 编写 video_retimer.py，实现依据 Whisper-X 时间轴，对视频片段进行局部变速/冻结拉伸的逻辑。| 3 天 | 
| P1 | 文本输入自适应 (Text Adapter) | 实现针对一坨长文的语义自动切片，以及针对已分段文本的上下文重叠（Context-Overlap）拼接。| 2 天 | 
| P2 | 桌面端交互封装 (Tauri+Vue3) | 搭建 Tauri 壳程序，将现有的 Python 核心代码封装为本地 API 服务。| 5 天 | 
| P2 | 性能调优 (AWQ 量化) | 对所有模型进行 AWQ 4-bit 量化，测试在 3060 上的推理效率与准确度极限。| 3 天 | 

### 4. 代码结构调整
建议建议你的 narrateflow 目录结构调整为以下“工业级”形态：
```Plaintext
narrateflow/
├── core/
│   ├── vision_seg.py      # pHash+OCR 级联切片引擎
│   ├── tts_engine.py      # 带滑动窗口 Prompt 的 Qwen-TTS 包装
│   ├── alignment.py       # Whisper-X 强制对齐算子
│   └── retimer.py         # FFmpeg 局部拉伸/冻结算法
├── utils/
│   ├── memory_guard.py    # 跨进程显存熔断逻辑
│   └── text_utils.py      # 正则清洗与语义切片
├── api/
│   └── service.py         # FastAPI 接口，供前端调用
├── gui/                   # Tauri + Vue3 前端代码
└── main.py                # 项目启动入口
```

## 五、现实落地开发计划（结合当前仓库）

本节不替换前面的工业级系统设计，而是把它拆成当前仓库可以逐步执行的工程路线。当前仓库已经具备 `run_pipeline.py` 主入口、`--project-dir` 任务目录、`task.json` 状态记录、`video_understanding.json` 共享中间产物，以及 Gemini API 版本的视频理解 baseline。下一阶段的目标，是把 Gemini 从“当前主实现”逐步降级为 fallback/debug baseline，并把本地视频理解模型接进同一套 pipeline。

### 5.1 当前阶段判断

当前不建议一次性同时推进 vLLM、KV Cache 驱逐、ROI OCR、Whisper-X、GUI、打包和本地大模型接入。更稳妥的路线是：先固定数据契约和 provider 接口，再接入本地 VLM，最后逐步做显存、OCR、对齐和产品化增强。

当前已经具备的基础：

- `--project-dir` 已作为单个任务的目录边界。
- 原始输入会尽量复制到 `<project-dir>/source/`。
- `task.json` 会记录输入、产物和部分运行选项。
- 视频理解统一输出到 `<project-dir>/understanding/video_understanding.json`。
- `script-align` 和 `video-auto` 都围绕 `spoken.json` 作为人工 review 点。
- Gemini API 已经可以作为视频理解 baseline。

当前最需要补齐的能力：

- `understand` provider 抽象。
- 本地 VLM 输出兼容 `video_understanding.json`。
- `video-auto` 讲解稿连续性优化。
- `script-align` 语义匹配质量优化。
- 本地模型运行时的显存隔离、失败恢复和缓存复用。

### 5.2 P0：冻结中间产物契约

目标：先让多个开发窗口不会互相踩踏，确保无论上游使用 Gemini、mock provider 还是本地 VLM，下游都读取同一份稳定 JSON。

需要稳定的 `video_understanding.json` 字段：

- `window_id`
- `start_time`
- `end_time`
- `visual_summary`
- `actions`
- `visible_text` 或 `onscreen_text`
- `objects` 或 `entities`
- `source_frames`

执行原则：

- 允许 provider 增加新字段。
- 不允许下游强依赖尚未稳定的新字段。
- `script`、`timeline`、`voice`、`compose` 不应该关心理解结果来自 Gemini 还是本地模型。

验收标准：

- Gemini 生成的 `video_understanding.json` 继续可被 `script` 和 `timeline` 使用。
- 本地 VLM 只要输出同样结构，就可以接入下游。
- `compose` 不需要感知视频理解来源。

### 5.3 P1：抽象视频理解 Provider

目标：把当前 Gemini-only 的 `understand` 阶段改成 provider 架构。

建议实现：

- 在 `timeline_align/video_understanding.py` 中保留窗口构建、缓存、写文件等公共逻辑。
- 新增或拆出 `timeline_align/understanding_providers.py`。
- 在 `run_pipeline.py` / `pipeline/stages.py` 中增加 provider 参数，例如：
  - `--vl-provider gemini`
  - `--vl-provider mock`
  - `--vl-provider local`

第一步不需要立刻接真实本地大模型，可以先做 `mock` provider：

- 输入同样的 window/keyframe manifest。
- 输出兼容的 `video_understanding.json`。
- 不调用网络。
- 用于验证 provider 抽象和下游链路是否解耦。

验收标准：

- 默认 `gemini` provider 不回退。
- `mock` provider 能在无 API key 情况下生成结构合法的理解文件。
- `script` 阶段不关心理解结果来自 Gemini、mock 还是本地模型。

### 5.4 P2：接入第一个本地 VLM 基线

目标：先跑通本地视频理解最小闭环，而不是一开始追求最高效果。

模型选择建议：

- 主方向可以保留 Qwen3.5-4B / Qwen3.5-VL 作为重点评估对象。
- 如果 RTX 3060 12GB 上 Qwen3.5-4B 压力过高，应准备一个轻量 fallback，例如 MiniCPM-V 4.x int4 或 Qwen2.5-VL-3B/7B 量化版。
- Gemini 保留为质量对照组和 fallback，不作为长期主线。

实现策略：

- 使用短窗口、多帧输入，不一次性喂完整长视频。
- 每个 window 独立推理或小 batch 推理。
- 输出严格 normalize 到 `video_understanding.json`。
- 本地模型加载、推理、释放先保持串行稳定。

验收标准：

- `python run_pipeline.py --only-stage understand --vl-provider local --project-dir ...` 可以生成 `video_understanding.json`。
- 后续 `script` / `timeline` / `voice` / `compose` 可以继续跑。
- 记录显存峰值、单窗口耗时、失败率和输出质量问题。

说明：上面这条命令是目标形态。当前仓库还没有 `--vl-provider` 参数，实现前应先完成 provider 抽象。

### 5.5 P3：提升 video-auto 文稿质量

目标：让视频自动生成的讲解稿从“画面描述”升级为“连续产品/教程解说”。

重点优化：

- 全局摘要：先理解视频整体任务和场景。
- 上下文滚动：每个 window 生成时知道前一个 window 已经讲了什么。
- 避免重复句式：减少“画面显示”“此时可以看到”等机械开头。
- 控制字数：根据 window 时长估算可讲文本长度。
- 风格模板：区分 SaaS 操作讲解、技术教程、产品演示。

验收标准：

- `script/page_01.spoken.json` 更像连续讲稿，而不是分帧 caption。
- 人工只需要 review `spoken.json`，不强制 review keyframe 或 understanding。
- 不引入运行时 LLM 评估。

### 5.6 P4：提升 script-align 时间轴匹配质量

目标：已有文字稿 + 视频的模式下，提升段落与视频窗口的匹配质量。

当前轻量匹配可以逐步升级为：

- 关键词 / token overlap。
- 屏幕文字 / 可见 UI 文本加权。
- 操作词和实体词加权。
- 单调顺序约束。
- 邻近窗口平滑。
- 匹配 debug 输出。

后续 OCR 字段加入后，可以把 OCR 文本作为额外匹配特征，而不是重写整个 timeline 逻辑。

验收标准：

- `timeline/page_01.timeline.final.json` 保持 compose 兼容。
- debug 信息能解释每段为什么匹配某个 window。
- 需要人工调整的段落数量减少。

### 5.7 P5：显存安全与本地运行时

目标：让本地 VLM、Qwen-TTS、未来 Whisper-X 不在 3060 12GB 上互相抢显存导致流程崩溃。

优先做工程稳定方案：

- 阶段串行执行。
- VLM 与 TTS 分进程运行。
- 单阶段结束后释放进程。
- 小 batch 推理。
- 失败时保留已有 project-dir 产物。
- 可恢复重跑未完成窗口。

暂不作为 P0 的能力：

- vLLM 深度集成。
- PagedAttention 调参。
- KV Cache 驱逐策略。
- 多模型常驻并发。

这些应在本地 VLM baseline 跑通后再评估收益。

验收标准：

- 本地 understand 失败不会破坏已有 `task.json` 和历史 artifact。
- 重跑可以复用已经完成的窗口。
- 能记录显存峰值和耗时。

### 5.8 P6：ROI OCR 与视觉切片增强

目标：提升代码录屏、SaaS 界面、小字密集场景的理解准确率。

落地顺序：

1. 先把 OCR 作为 debug artifact 输出，不影响主流程。
2. 再将 OCR 文本作为 VLM prompt 的辅助上下文。
3. 最后将 OCR 文本纳入 timeline 匹配评分。

建议优先区域：

- 顶部标题栏 / 路径栏。
- 左侧目录树 / 导航栏。
- 主操作按钮区域。
- 当前焦点窗口或弹窗。

验收标准：

- OCR 不可用时主流程仍可运行。
- OCR 可用时，小字场景的理解和匹配质量提升。
- OCR 结果写入 `understanding/` 下的 debug 文件或 window 字段中。

### 5.9 P7：Whisper-X / MFA 对齐增强

目标：在音频精对齐或原视频有人声时提供更强的时间校准能力。

推荐定位：

- 不作为当前主路径 P0。
- 作为 timeline/voice 后的增强分支。
- 对“原视频有人声，需要分析原讲解时间轴”的场景更有价值。
- 对“无讲解视频自动配音”的主场景，优先级低于本地 VLM 和文稿质量。

验收标准：

- Whisper-X 不安装时主流程可继续运行。
- 安装后可作为可选 alignment provider。
- 输出仍能转换到现有 `timeline.final.json` 或兼容结构。

### 5.10 P8：GUI 与打包

目标：在核心 pipeline 稳定后，再封装桌面端体验。

推荐时机：

- `--project-dir` 稳定。
- provider 架构稳定。
- 本地 VLM baseline 可运行。
- `spoken.json` review 流程稳定。
- 常见失败可恢复。

GUI 初版重点：

- 创建 / 打开项目目录。
- 选择视频、文字稿、音色。
- 执行阶段。
- 查看 `spoken.json`。
- 查看最终视频。
- 展示失败日志和可重跑阶段。

### 5.11 多窗口并行建议

多 Claude 窗口并行时建议按以下边界拆分，减少冲突：

- 窗口 A：provider 抽象和本地 VLM 接入口。
- 窗口 B：`video_script.py`，提升 video-auto 文稿质量。
- 窗口 C：`run_timeline_align.py` 或新增 `matching.py`，提升 script-align 匹配。
- 窗口 D：`pipeline/local_runtime.py` / `process_guard.py`，处理本地推理运行时和显存安全。
- 窗口 E：文档、验证脚本、CLI 对齐。

详细任务单见 `DEVELOPMENT_PARALLEL_TASKS.md`。

### 5.12 推荐近期提交顺序

1. `feat: add understanding provider selection`
2. `feat: add mock understanding provider`
3. `feat: improve video-auto narration drafting`
4. `feat: improve understanding-based timeline matching`
5. `feat: add local runtime process guard`
6. `feat: add first local VLM understanding provider`
7. `doc: add local model setup and validation guide`

每个阶段完成后都应跑一次最小验证，并提交/推送，避免多个窗口长期积压大改动。

### 5.13 当前最推荐的下一步

最推荐先做：`understand` provider 抽象 + `mock` provider。

原因：

- 它不要求马上解决本地模型安装和显存问题。
- 它可以立刻把 Gemini 从硬编码主路径变成一个 provider。
- 它能让后续本地 Qwen3.5-4B、MiniCPM-V、OCR 增强都沿同一接口接入。
- 它能保护下游 `script`、`timeline`、`voice`、`compose` 不被模型替换反复牵动。

完成这一步后，再进入真实本地 VLM 接入会稳很多。
