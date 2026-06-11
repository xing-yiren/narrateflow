"""
音频工具：时长读取、音频拼接信息
所有音频时长必须实测，禁止用字数估算。
"""
import wave
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def get_wav_duration(wav_path: str) -> float:
    """
    用 wave 模块读取 WAV 文件时长（毫秒级精度）。
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return round(frames / rate, 3)
    except (wave.Error, FileNotFoundError) as e:
        logger.error(f"读取 WAV 时长失败: {wav_path}, {e}")
    
    # 兜底: 使用 ffprobe
    from narrateflow.utils.ffprobe import get_audio_info
    info = get_audio_info(wav_path)
    return info["duration"]


def batch_get_durations(wav_paths: List[str]) -> List[float]:
    """批量获取 WAV 文件时长"""
    return [get_wav_duration(p) for p in wav_paths]


def estimate_audio_duration_never_use(text: str, chars_per_second: float = 4.5) -> float:
    """
    ⚠️ 禁止使用！仅作为参考对照存在。
    此函数仅用于日志对比"估算值 vs 实测值"的差距。
    """
    import re
    # 只计算中文字符和标点
    chinese_chars = len(re.findall(r'[一-鿿　-〿＀-￯]', text))
    return round(chinese_chars / chars_per_second, 2)


def concat_audio_plan(segments: List[Dict]) -> List[Dict]:
    """
    生成 FFmpeg concat 所需的音频片段计划。
    每段包含音频文件路径、start/end时间到全局时间轴。
    
    Args:
        segments: [{"audio_file": str, "duration": float, ...}, ...]
    
    Returns:
        [{"audio_file": str, "global_start": float, "global_end": float, ...}, ...]
    """
    plan = []
    current_time = 0.0
    
    for seg in segments:
        duration = seg["duration"]
        plan.append({
            "audio_file": seg["audio_file"],
            "global_start": round(current_time, 3),
            "global_end": round(current_time + duration, 3),
            "duration": duration,
            "index": seg.get("index", -1),
        })
        current_time += duration
    
    return plan
