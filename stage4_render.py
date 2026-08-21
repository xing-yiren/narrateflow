"""
Stage 4: FFmpeg 渲染合成
读取 timeline_plan，生成 FFmpeg 命令，渲染输出并质检。
Mac: h264_videotoolbox 硬编
输出: final_video.mp4 + qc_report.json
"""
import json
import logging
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from narrateflow.utils.ffprobe import get_video_info, get_audio_info

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# FFmpeg 命令生成
# ═══════════════════════════════════════════════════════════════

def _build_video_concat_filter(
    segments: List[Dict],
    video_path: str,
    output_fps: int = 30,
) -> str:
    """
    根据 timeline_plan segments 构建 FFmpeg video filter_complex。
    每个 segment 按 retime_strategy 生成对应的滤镜：
    - normal: trim + setpts=PTS
    - speed_change: trim + setpts=PTS*speed_factor
    - freeze: trim + tpad (最后一帧定格)
    - merge/broll/manual_review: 简单 trim（同 normal）
    """
    filters = []
    input_count = len(segments)
    
    for i, seg in enumerate(segments):
        v_start = seg["video_start"]
        v_end = seg["video_end"]
        strategy = seg["retime_strategy"]
        speed = seg.get("speed_factor", 1.0)
        freeze_pt = seg.get("freeze_point")
        
        if strategy == "freeze" and freeze_pt is not None:
            # 正常播放到 freeze_pt，然后定格
            # 计算需要定格多少秒
            audio_dur = seg["audio_end"] - seg["audio_start"]
            play_dur = freeze_pt - v_start
            freeze_dur = max(0, audio_dur - play_dur)
            
            filters.append(
                f"[0:v]trim=start={v_start}:end={freeze_pt},setpts=PTS-STARTPTS,"
                f"tpad=stop_mode=clone:stop_duration={freeze_dur},"
                f"fps={output_fps}[v{i}]"
            )
        elif strategy == "speed_change" and speed != 1.0:
            filters.append(
                f"[0:v]trim=start={v_start}:end={v_end},"
                f"setpts=PTS*{speed:.3f}/1-STARTPTS,"
                f"fps={output_fps}[v{i}]"
            )
        else:
            # normal / merge / broll / manual_review
            filters.append(
                f"[0:v]trim=start={v_start}:end={v_end},"
                f"setpts=PTS-STARTPTS,"
                f"fps={output_fps}[v{i}]"
            )
    
    # Concat all segments
    concat_inputs = "".join(f"[v{i}]" for i in range(input_count))
    filters.append(
        f"{concat_inputs}concat=n={input_count}:v=1:a=0[outv]"
    )
    
    return ";\n".join(filters)


def _build_audio_concat_filter(segments: List[Dict]) -> str:
    """
    构建音频拼接滤镜。每个段落的音频按 audio_start/audio_end trim 后 concat。
    所有音频来自同一个输入文件（由 Stage 4 预先 concat 所有分段音频）。
    """
    filters = []
    input_count = len(segments)
    
    for i, seg in enumerate(segments):
        a_start = seg["audio_start"]
        a_end = seg["audio_end"]
        filters.append(
            f"[1:a]atrim=start={a_start}:end={a_end},asetpts=PTS-STARTPTS[a{i}]"
        )
    
    concat_inputs = "".join(f"[a{i}]" for i in range(input_count))
    filters.append(
        f"{concat_inputs}concat=n={input_count}:v=0:a=1[outa]"
    )
    
    return ";\n".join(filters)


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def run_stage4(
    timeline_path: str = "output/default/timeline_plan.ai.json",
    output_dir: str = "output/default",
    config: Optional[Dict] = None,
) -> str:
    """
    Stage 4 主入口。
    
    Returns:
        final_video.mp4 的路径
    """
    output_dir = Path(output_dir)
    
    # 优先读取人工修正版
    human_timeline = output_dir / "timeline_plan.human.json"
    if human_timeline.exists():
        timeline_path = str(human_timeline)
        logger.info("使用人工修正 timeline_plan.human.json")
    
    # 1. 加载 timeline
    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline = json.load(f)
    
    segments = timeline["segments"]
    
    stage4_cfg = config.get("stage4", {}) if config else {}
    video_cfg = stage4_cfg.get("video", {})
    audio_cfg = stage4_cfg.get("audio", {})
    qc_cfg = stage4_cfg.get("qc", {})
    
    output_fps = video_cfg.get("fps", 30)
    
    # 查找输入视频和音频
    video_path = output_dir.parent.parent / "input" / "video.mp4"
    if not video_path.exists():
        video_path = Path("input/video.mp4")
    
    audio_dir = output_dir / "audio"
    
    # 预先 concat 所有音频段为单一文件
    concat_audio_path = output_dir / "audio_combined.wav"
    _concat_audio_files(segments, audio_dir, concat_audio_path)
    
    logger.info("=" * 50)
    logger.info("Stage 4: FFmpeg 渲染合成")
    logger.info("=" * 50)
    
    # 2. 构建 FFmpeg 命令
    encoder = stage4_cfg.get("video_encoder", "h264_videotoolbox")
    
    # 检测可用编码器
    if not _encoder_available(encoder):
        logger.warning(f"{encoder} 不可用，降级为 libx264")
        encoder = "libx264"
    
    logger.info(f"视频编码器: {encoder}")
    
    video_filter = _build_video_concat_filter(segments, str(video_path), output_fps)

    # 3. 分步渲染：先渲染视频，再加音频
    video_only_path = output_dir / "video_only.mp4"
    final_path = output_dir / "final_video.mp4"
    
    t_start = time.time()
    
    # 3a. 渲染视频部分
    vf_file = output_dir / "video_filter.txt"
    with open(vf_file, "w") as f:
        f.write(video_filter)
    
    if encoder == "h264_videotoolbox":
        vf_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-filter_complex_script", str(vf_file),
            "-map", "[outv]",
            "-c:v", "h264_videotoolbox",
            "-b:v", video_cfg.get("bitrate", "5M"),
            "-r", str(output_fps),
            str(video_only_path)
        ]
    else:
        vf_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-filter_complex_script", str(vf_file),
            "-map", "[outv]",
            "-c:v", "libx264",
            "-crf", str(video_cfg.get("crf", 23)),
            "-preset", video_cfg.get("preset", "medium"),
            "-r", str(output_fps),
            str(video_only_path)
        ]
    
    logger.info("渲染视频...")
    result_v = subprocess.run(vf_cmd, capture_output=True, text=True, timeout=600)
    if result_v.returncode != 0:
        logger.error(f"视频渲染失败:\n{result_v.stderr[-500:]}")
        raise RuntimeError("FFmpeg 视频渲染失败")
    
    logger.info(f"✓ 视频渲染完成: {video_only_path}")
    
    # 3b. 合成音频：视频已按 timeline 渲染，直接 mux 音频
    # -shortest: 视频比音频长时，以音频长度为准截断
    audio_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_only_path),
        "-i", str(concat_audio_path),
        "-c:v", "copy",
        "-c:a", audio_cfg.get("codec", "aac"),
        "-b:a", audio_cfg.get("bitrate", "192k"),
        "-ar", str(audio_cfg.get("sample_rate", 48000)),
        "-shortest",
        str(final_path)
    ]

    logger.info("合成音频...")
    result_a = subprocess.run(audio_cmd, capture_output=True, text=True, timeout=300)
    if result_a.returncode != 0:
        logger.error(f"音频合成失败:\n{result_a.stderr[-500:]}")
        raise RuntimeError("FFmpeg 音频合成失败")
    
    logger.info(f"✓ 音频合成完成: {final_path}")
    
    # 清理中间文件
    video_only_path.unlink(missing_ok=True)
    concat_audio_path.unlink(missing_ok=True)
    vf_file.unlink(missing_ok=True)
    
    t_elapsed = time.time() - t_start
    
    # 4. 质检
    qc_report = _run_qc(
        str(final_path),
        timeline,
        qc_cfg,
        t_elapsed,
    )
    
    qc_path = output_dir / "qc_report.json"
    with open(qc_path, "w", encoding="utf-8") as f:
        json.dump(qc_report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ 质检报告: {qc_path}")
    logger.info(f"✓ final_video.mp4 已输出: {final_path}")
    logger.info(f"  总耗时: {t_elapsed:.1f}s")
    
    return str(final_path)


def _concat_audio_files(segments: List[Dict], audio_dir: Path, output_path: Path):
    """将所有段落的音频文件拼接为单一的 WAV 文件"""
    # 生成 concat 列表
    concat_list = audio_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for seg in segments:
            audio_file = audio_dir / Path(seg.get("audio_file", f"seg_{seg['narration_index']:03d}.wav")).name
            if audio_file.exists():
                f.write(f"file '{audio_file.absolute()}'\n")
    
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"音频拼接失败: {result.stderr}")
    concat_list.unlink(missing_ok=True)


def _encoder_available(encoder: str) -> bool:
    """检查 FFmpeg 编码器是否可用"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10
        )
        return encoder in result.stdout
    except Exception:
        return False


def _run_qc(
    final_path: str,
    timeline: Dict,
    qc_cfg: Dict,
    elapsed: float,
) -> Dict:
    """自动质检"""
    duration_tolerance = qc_cfg.get("duration_tolerance", 2.0)
    black_threshold = qc_cfg.get("black_frame_threshold", 16)
    silence_threshold = qc_cfg.get("silence_threshold_db", -40)
    
    issues = []
    
    # 1. 时长校验
    try:
        info = get_video_info(final_path)
        actual_duration = info["duration"]
        expected_duration = timeline.get("total_duration", 0)
        delta = abs(actual_duration - expected_duration)
        
        if delta > duration_tolerance:
            issues.append(f"时长偏差 {delta:.1f}s (预期 {expected_duration:.1f}s, 实际 {actual_duration:.1f}s)")
        
        logger.info(f"  时长校验: 预期 {expected_duration:.1f}s, 实际 {actual_duration:.1f}s, 偏差 {delta:.1f}s")
    except Exception as e:
        issues.append(f"时长校验失败: {e}")
        actual_duration = 0
    
    # 2. 黑帧检测
    try:
        cmd = [
            "ffmpeg", "-v", "quiet",
            "-i", final_path,
            "-vf", f"blackframe=threshold={black_threshold}:amount=98",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        black_lines = [l for l in result.stderr.split("\n") if "blackframe" in l.lower()]
        black_count = len(black_lines)
        if black_count > 3:
            issues.append(f"检测到 {black_count} 个黑帧")
        logger.info(f"  黑帧检测: {black_count} 个")
    except Exception:
        black_count = 0
        issues.append("黑帧检测失败")
    
    # 3. 静音检测
    try:
        cmd = [
            "ffmpeg", "-v", "quiet",
            "-i", final_path,
            "-af", f"silencedetect=n={silence_threshold}dB:d=0.5",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        silence_lines = [l for l in result.stderr.split("\n") if "silence" in l.lower()]
        silence_count = len([l for l in silence_lines if "silence_start" in l])
        if silence_count > 3:
            issues.append(f"检测到 {silence_count} 段长静音")
        logger.info(f"  静音检测: {silence_count} 段")
    except Exception:
        silence_count = 0
        issues.append("静音检测失败")
    
    return {
        "generated_at": datetime.now().isoformat(),
        "file": final_path,
        "expected_duration": timeline.get("total_duration", 0),
        "actual_duration": round(actual_duration, 3),
        "elapsed_seconds": round(elapsed, 1),
        "black_frames": black_count if "black_count" in dir() else -1,
        "silence_segments": silence_count if "silence_count" in dir() else -1,
        "issues": issues,
        "passed": len(issues) == 0,
    }


# ═══════════════════════════════════════════════════════════════
# CLI 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    timeline = sys.argv[1] if len(sys.argv) > 1 else "output/default/timeline_plan.ai.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "output/default"
    
    try:
        result = run_stage4(timeline, output)
        print(f"\n✓ Stage 4 完成: {result}")
    except Exception as e:
        logger.error(f"Stage 4 失败: {e}", exc_info=True)
        sys.exit(1)
