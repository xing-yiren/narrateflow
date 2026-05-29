# NarrateFlow 开发日志

## 2026-05-29

### 背景
- 目标：在保持原有 CUDA 部署能力的同时，补齐在当前设备上的可运行部署能力。
- 当前设备实测信息：
  - OS: macOS 26.0.1
  - 当前 Python: 3.8.13
  - 当前解释器架构：x86_64
  - 机器物理架构：Apple Silicon（M4 Pro）
  - ffmpeg / ffprobe：未安装
- 初步判断：
  - 原仓库的 TTS 运行时主要围绕 CUDA 编写；
  - 主入口 `run_pipeline.py` 没有把 `device` / `dtype` / `voice_batch_size` 贯穿到完整 pipeline；
  - 在当前 x86_64 Python 环境下，优先保证 `cpu` 可跑通；若切换为原生 arm64 Python，再启用 `mps`；CUDA 保持原逻辑。

### 今日目标
1. 增加统一运行时设备选择：CUDA / MPS / CPU。
2. 在 pipeline 主入口暴露并传递 `device` / `dtype` / `voice_batch_size`。
3. 更新配置文件与 README，补充 macOS / CUDA 双部署说明。
4. 准备本机部署验证脚本或命令。
5. 做至少一轮静态检查/编译检查；若本机依赖缺失，则记录阻塞项。
6. 用 git commit 记录阶段性修改；必要时再 push 独立分支。

### 进展记录
- [x] 建立开发分支：`feat/device-compat-macos`
- [x] 建立开发日志 `docs/DEVLOG.md`
- [x] 增加统一 device / dtype 解析逻辑
- [x] 将 device / dtype / voice_batch_size 贯穿到 pipeline 配置
- [x] 将 voice 生成阶段接入 device / dtype / batch size
- [x] 将 outro 语音生成接入 device / dtype
- [x] 更新 `config/video_mode.toml`
- [x] 更新 README 的设备支持说明
- [x] 本机部署环境创建完成：`conda create -n narrateflow python=3.11 ffmpeg pip`
- [x] 公共 Python 依赖安装完成
- [x] `qwen-tts==0.1.1` 安装完成
- [x] `scripts/deploy_check.py --device cpu/auto/mps` 依赖与设备检查通过
- [ ] 创建 git commit
- [ ] push 到远端独立分支

### 当前结论
- base 环境（Python 3.8.13）不满足项目部署需要，不建议直接使用。
- 新建 `narrateflow` conda 环境后，CPU 部署检查已通过。
- 在该环境中，`torch.backends.mps.is_available()` 返回 `True`，说明当前设备后续可进一步使用 MPS。
- 仍未做完整 TTS 生成 smoke test 的唯一剩余前提，是本地 Qwen-TTS 模型目录需准备到 `models/Qwen/...`；当前脚本会明确报告模型缺失。

### TODO
- [x] 确认 qwen-tts 可在新环境安装
- [x] 安装 ffmpeg / ffprobe 并完成部署检查
- [x] 补一个最小部署检查脚本
- [x] 增加 `scripts/prepare_qwen_tts_model.py`，用于下载本地 Qwen-TTS 模型
- [ ] 准备本地 Qwen-TTS 模型到 `models/Qwen/...`
- [ ] 用真实 profile / spoken json 执行一次 voice stage smoke test
- [ ] 用真实视频执行一次 full 或 from-stage smoke test
- [ ] push 到远端独立分支
