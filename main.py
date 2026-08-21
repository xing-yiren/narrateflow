"""
NarrateFlow 总控脚本
按 Stage 0→1→2→3→4 顺序执行，支持从任意 Stage 重新执行。
"""
import sys
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import yaml

from narrateflow.stage0_manifest import run_stage0
from narrateflow.stage1_vision import run_stage1, MockVLMProvider
from narrateflow.stage2_tts import run_stage2, EdgeTTSProvider
from narrateflow.stage3_align import run_stage3
from narrateflow.stage4_render import run_stage4


def load_config(config_path: str = "config.yaml") -> Dict:
    """加载全局配置"""
    config_paths = [
        Path(config_path),
        Path(__file__).parent / "config.yaml",
    ]
    for p in config_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    logging.warning("未找到 config.yaml，使用默认配置")
    return {}


def setup_logging(config: Dict):
    """配置日志"""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("file")
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def run_pipeline(
    video_path: str = "input/video.mp4",
    narration_path: str = "input/narration.txt",
    output_dir: str = "output/default",
    config_path: str = "config.yaml",
    start_stage: int = 0,
    end_stage: int = 4,
    use_mock_vlm: bool = True,
) -> Dict:
    """
    执行完整流水线 or 部分 Stages。
    
    Args:
        video_path: 输入视频路径
        narration_path: 输入文稿路径
        output_dir: 输出目录
        config_path: 配置文件路径
        start_stage: 起始 Stage (0-4)
        end_stage: 结束 Stage (0-4)
        use_mock_vlm: 是否使用 Mock VLM (P0 快速验证)
    
    Returns:
        run_metrics dict
    """
    config = load_config(config_path)
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    
    # 确定项目基础路径
    project_root = Path(__file__).parent
    video_path = str(project_root / video_path)
    narration_path = str(project_root / narration_path)
    output_dir = str(project_root / output_dir)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "pipeline": "narrateflow",
        "started_at": datetime.now().isoformat(),
        "stages": {},
        "overall_elapsed": 0,
    }
    
    t_overall = time.time()
    
    logger.info("=" * 60)
    logger.info("NarrateFlow 视频配音系统")
    logger.info(f"  视频: {video_path}")
    logger.info(f"  文稿: {narration_path}")
    logger.info(f"  输出: {output_dir}")
    logger.info(f"  Range: Stage {start_stage} → Stage {end_stage}")
    logger.info("=" * 60)
    
    manifest_path = f"{output_dir}/project_manifest.json"
    video_meta_path = f"{output_dir}/video_meta.ai.json"
    audio_manifest_path = f"{output_dir}/narration_audio_manifest.json"
    timeline_path = f"{output_dir}/timeline_plan.ai.json"
    final_path = f"{output_dir}/final_video.mp4"
    
    # ── Stage 0: 资产探测 ──────────────────────────
    if start_stage <= 0 <= end_stage:
        t0 = time.time()
        try:
            run_stage0(video_path, narration_path, output_dir, config)
            metrics["stages"]["stage0"] = {"status": "ok", "elapsed": round(time.time() - t0, 1)}
            logger.info("✓ Stage 0 完成")
        except Exception as e:
            metrics["stages"]["stage0"] = {"status": "error", "error": str(e)}
            logger.error(f"✗ Stage 0 失败: {e}")
            if start_stage == 0:
                raise
    
    # ── Stage 1: 视频理解 ──────────────────────────
    if start_stage <= 1 <= end_stage:
        t1 = time.time()
        try:
            if use_mock_vlm:
                vlm_provider = MockVLMProvider()
                logger.info("⚠ 使用 Mock VLM Provider（非真实 AI 输出）")
            else:
                # 自动选择: 优先 Ollama 本地，不支持则降级
                vlm_provider = None  # run_stage1 will auto-detect from config

            run_stage1(
                video_path, manifest_path, output_dir, config,
                vlm_provider=vlm_provider,
            )
            metrics["stages"]["stage1"] = {"status": "ok", "elapsed": round(time.time() - t1, 1)}
            logger.info("✓ Stage 1 完成")
        except Exception as e:
            metrics["stages"]["stage1"] = {"status": "error", "error": str(e)}
            logger.error(f"✗ Stage 1 失败: {e}")
            if start_stage == 1:
                raise
    
    # ── Stage 2: TTS (P3: auto-detect best) ─────────
    if start_stage <= 2 <= end_stage:
        t2 = time.time()
        try:
            stage2_cfg = config.get("stage2", {}) if config else {}
            provider_type = stage2_cfg.get("tts_provider", "edge_tts")

            tts_provider = None
            if provider_type == "voxcpm2":
                try:
                    from narrateflow.providers.tts_voxcpm2 import VoxCPM2Provider
                    vox_cfg = stage2_cfg.get("voxcpm2", {})
                    tts_provider = VoxCPM2Provider(
                        model_path=vox_cfg.get("model_path"),
                        dtype=vox_cfg.get("dtype"),  # Mac auto fp16
                    )
                    if not tts_provider.check_available():
                        logger.warning("VoxCPM2 不可用，降级为 edge-tts")
                        tts_provider = None
                except Exception as e:
                    logger.warning(f"VoxCPM2 初始化失败: {e}，降级为 edge-tts")

            if tts_provider is None:
                edge_cfg = stage2_cfg.get("edge_tts", {})
                tts_provider = EdgeTTSProvider(
                    voice=edge_cfg.get("voice", "zh-CN-XiaoxiaoNeural"),
                    rate=edge_cfg.get("rate", "+0%"),
                    pitch=edge_cfg.get("pitch", "+0Hz"),
                )

            run_stage2(manifest_path, output_dir, config, tts_provider=tts_provider)
            metrics["stages"]["stage2"] = {"status": "ok", "elapsed": round(time.time() - t2, 1)}
            logger.info("✓ Stage 2 完成")
        except Exception as e:
            metrics["stages"]["stage2"] = {"status": "error", "error": str(e)}
            logger.error(f"✗ Stage 2 失败: {e}")
            if start_stage == 2:
                raise
    
    # ── Stage 3: 时间轴对齐 ────────────────────────
    if start_stage <= 3 <= end_stage:
        t3 = time.time()
        try:
            run_stage3(
                manifest_path, video_meta_path, audio_manifest_path,
                output_dir, config,
            )
            metrics["stages"]["stage3"] = {"status": "ok", "elapsed": round(time.time() - t3, 1)}
            logger.info("✓ Stage 3 完成")
        except Exception as e:
            metrics["stages"]["stage3"] = {"status": "error", "error": str(e)}
            logger.error(f"✗ Stage 3 失败: {e}")
            if start_stage == 3:
                raise
    
    # ── Stage 4: 渲染 ──────────────────────────────
    if start_stage <= 4 <= end_stage:
        t4 = time.time()
        try:
            run_stage4(timeline_path, output_dir, config)
            metrics["stages"]["stage4"] = {"status": "ok", "elapsed": round(time.time() - t4, 1)}
            logger.info("✓ Stage 4 完成")
        except Exception as e:
            metrics["stages"]["stage4"] = {"status": "error", "error": str(e)}
            logger.error(f"✗ Stage 4 失败: {e}")
            if start_stage == 4:
                raise
    
    # ── 汇总 ──────────────────────────────────────
    t_total = time.time() - t_overall
    metrics["overall_elapsed"] = round(t_total, 1)
    metrics["finished_at"] = datetime.now().isoformat()
    metrics["output_file"] = final_path
    
    # 保存指标
    metrics_path = Path(output_dir) / "run_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"流水线完成! 总耗时: {t_total:.1f}s")
    logger.info(f"  输出: {final_path}")
    logger.info(f"  指标: {metrics_path}")
    logger.info("=" * 60)
    
    # 打印各 Stage 耗时
    for name, info in metrics["stages"].items():
        status = info["status"]
        elapsed = info.get("elapsed", "?")
        logger.info(f"  {name}: {status} ({elapsed}s)")
    
    return metrics


def main():
    """CLI 入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NarrateFlow — 视频配音系统 (Mac M4 Pro)",
    )
    parser.add_argument("--video", default="input/video.mp4", help="输入视频路径")
    parser.add_argument("--narration", default="input/narration.txt", help="旁白文稿路径")
    parser.add_argument("--output", default="output/default", help="输出目录")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--start", type=int, default=0, help="起始 Stage (0-4)")
    parser.add_argument("--end", type=int, default=4, help="结束 Stage (0-4)")
    parser.add_argument("--mock-vlm", action="store_true", default=True,
                       help="使用 Mock VLM (P0 快速验证)")
    parser.add_argument("--real-vlm", dest="mock_vlm", action="store_false",
                       help="使用真实 VLM")
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            video_path=args.video,
            narration_path=args.narration,
            output_dir=args.output,
            config_path=args.config,
            start_stage=args.start,
            end_stage=args.end,
            use_mock_vlm=args.mock_vlm,
        )
    except Exception as e:
        logging.error(f"流水线失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
