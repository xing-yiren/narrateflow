# NarrateFlow 开发 TODO — Mac M4 Pro 路径

> 基于 NarrateFlow_Final.md 技术方案，按 P0→P1→P2→P3→P4 路线图推进

---

## P0 — MVP：端到端链路跑通（目标 2-3 周）

- [ ] **[P0.1]** 项目骨架搭建：目录结构、config.yaml、requirements.txt、__init__.py
- [ ] **[P0.2]** Stage 0：资产探测与文稿解析 (stage0_manifest.py)
  - [ ] ffprobe 封装 (utils/ffprobe.py)
  - [ ] 文稿分句 + 规范化
  - [ ] project_manifest.json schema + 输出
- [ ] **[P0.3]** Stage 1：视频结构化理解 (stage1_vision.py) — P0 用云端 API 快速验证
  - [ ] PySceneDetect 物理切片
  - [ ] 抽帧逻辑（动态帧差 + 硬上限 12 帧 + 长边 600px）
  - [ ] VLM 推理 + JSON 解析（容错）
  - [ ] Schema 校验 (utils/schema_validator.py)
- [ ] **[P0.4]** Stage 2：旁白音频生成 (stage2_tts.py) — P0 暂用 edge-tts 跑通
  - [ ] TTS 合成
  - [ ] 真实音频时长读取 (utils/audio.py)
- [ ] **[P0.5]** Stage 3：时间轴对齐 (stage3_align.py) — 简化版顺序+时长
  - [ ] 顺序匹配 + 时长匹配
  - [ ] 原速播放 / 轻微变速策略
- [ ] **[P0.6]** Stage 4：FFmpeg 渲染合成 (stage4_render.py)
  - [ ] Mac: h264_videotoolbox 编码
  - [ ] 音频拼接 + 混音
  - [ ] 自动质检
- [ ] **[P0.7]** main.py 总控脚本
- [ ] **[P0.8]** 端到端集成测试

---

## P1 — 本地 VLM 部署（目标 2-3 周）

- [ ] **[P1.1]** Mac 部署 Qwen3-VL-8B Q8_0 到 Ollama
- [ ] **[P1.2]** VLM Provider 层：vlm_base.py + vlm_ollama.py
- [ ] **[P1.3]** 动态帧差抽帧优化
- [ ] **[P1.4]** JSON 解析多级容错
- [ ] **[P1.5]** Schema 校验完善
- [ ] **[P1.6]** Mac 内存监控 (psutil)
- [ ] **[P1.7]** 系统 Prompt 优化（六维打标）
- [ ] **[P1.8]** P1 集成测试 + 性能记录

---

## P2 — 对齐质量提升（目标 2-3 周）

- [ ] **[P2.1]** 7 维匹配评分系统
- [ ] **[P2.2]** 语义相似度
- [ ] **[P2.3]** 滑窗组合（1-3 连续窗口）
- [ ] **[P2.4]** 完整重定时策略
- [ ] **[P2.5]** 定格选点算法
- [ ] **[P2.6]** 人工锁定点支持

---

## P3 — TTS 质量提升（目标 2-3 周）

- [ ] **[P3.1]** VoxCPM2 FP16 部署到 Mac
- [ ] **[P3.2]** TTS Provider 层：tts_base.py + tts_voxcpm2.py
- [ ] **[P3.3]** Mac MPS 确认（不用 BF16）
- [ ] **[P3.4]** 压测验证清单
- [ ] **[P3.5]** 发音词典 ≥ 50 条目
- [ ] **[P3.6]** 单段重跑机制
- [ ] **[P3.7]** 长文本分段语气连贯性

---

## P4 — 持续优化

- [ ] 审核 UI / 评测集 / few-shot 池 / 规则库 / 微调

---

## 开发记录

| 日期 | 完成事项 | 测试结果 | 备注 |
|------|---------|---------|------|
| | | | |
