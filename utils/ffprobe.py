"""
ffprobe 命令封装
所有视频元数据必须来自 ffprobe 实测，禁止估算。
"""
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_video_info(video_path: str, ffprobe_path: str = "ffprobe") -> Dict[str, Any]:
    """
    用 ffprobe 获取视频真实元信息。
    
    Args:
        video_path: 视频文件路径
        ffprobe_path: ffprobe 命令路径
        
    Returns:
        dict with keys: path, duration, fps, width, height, codec, bitrate, duration_frames
        
    Raises:
        FileNotFoundError: 视频文件不存在
        RuntimeError: ffprobe 执行失败
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 获取视频流信息
    cmd = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe 执行失败: {result.stderr}")
        
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe 输出解析失败: {e}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe 超时 (30s): {video_path}")
    
    # 查找视频流
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    
    if video_stream is None:
        raise RuntimeError(f"未找到视频流: {video_path}")
    
    # 提取关键信息（全部来自 ffprobe 实测值）
    # FPS: 优先 r_frame_rate，其次 avg_frame_rate
    fps_str = video_stream.get("r_frame_rate", video_stream.get("avg_frame_rate", "0/1"))
    fps = _parse_fraction(fps_str)
    
    # Duration: stream.duration 或 format.duration
    duration = float(video_stream.get("duration") or data.get("format", {}).get("duration", 0))
    
    # Duration in frames
    duration_frames = int(video_stream.get("nb_frames") or 0)
    if duration_frames == 0 and fps > 0:
        duration_frames = int(duration * fps)
    
    info = {
        "path": str(video_path.absolute()),
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "width": video_stream.get("width", 0),
        "height": video_stream.get("height", 0),
        "codec": video_stream.get("codec_name", "unknown"),
        "bitrate": data.get("format", {}).get("bit_rate", "unknown"),
        "duration_frames": duration_frames,
    }
    
    logger.info(f"视频信息: {info['width']}x{info['height']}, "
                f"{info['fps']}fps, {info['duration']}s, {info['codec']}")
    
    return info


def get_audio_info(audio_path: str, ffprobe_path: str = "ffprobe") -> Dict[str, Any]:
    """
    获取音频文件的真实时长（不可用字数估算）。
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    cmd = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(audio_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe 执行失败: {result.stderr}")
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe JSON 解析失败: {e}")
    
    duration = float(data.get("format", {}).get("duration", 0))
    sample_rate = 0
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            sample_rate = int(stream.get("sample_rate", 0))
            break
    
    return {
        "path": str(audio_path.absolute()),
        "duration": round(duration, 3),
        "sample_rate": sample_rate,
    }


def get_audio_duration_wave(audio_path: str) -> float:
    """
    用 Python wave 模块读取 WAV 文件时长（比 ffprobe 更快，无外部依赖）。
    """
    import wave
    
    with wave.open(str(audio_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate > 0:
            return round(frames / rate, 3)
    return 0.0


def _parse_fraction(fps_str: str) -> float:
    """解析 '30000/1001' 或 '30' 这类帧率字符串"""
    fps_str = fps_str.strip()
    if not fps_str:
        return 0.0
    if "/" in fps_str:
        try:
            num, den = fps_str.split("/")
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            logger.warning(f"无法解析帧率: {fps_str}")
            return 0.0
    try:
        return float(fps_str)
    except ValueError:
        logger.warning(f"无法解析帧率: {fps_str}")
        return 0.0
