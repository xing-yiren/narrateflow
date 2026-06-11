"""
Stage 2: 旁白音频生成（TTS 独占）
P0: 使用 edge-tts (轻量方案快速验证)
P3: 切换为 VoxCPM2
输出: output/{project_id}/audio/*.wav + narration_audio_manifest.json
"""
import json
import logging
import time
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from narrateflow.utils.audio import get_wav_duration, concat_audio_plan
from narrateflow.utils.schema_validator import validate_json

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# TTS Provider 接口 (可插拔)
# ═══════════════════════════════════════════════════════════════

class TTSProvider:
    """TTS 抽象基类"""
    
    def synthesize(self, text: str, output_path: str, **kwargs) -> float:
        """
        合成一段音频。
        
        Returns:
            真实音频时长（秒），来自实测，不可估算
        """
        raise NotImplementedError


class EdgeTTSProvider(TTSProvider):
    """Edge TTS Provider — P0 快速验证"""
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", 
                 rate: str = "+0%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
    
    def synthesize(self, text: str, output_path: str, **kwargs) -> float:
        """使用 edge-tts 合成音频"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # edge-tts 输出 MP3，需要转 WAV 以获得精确时长
        mp3_path = output_path.with_suffix(".mp3")
        
        async def _run():
            import edge_tts
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch,
            )
            await communicate.save(str(mp3_path))
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有 event loop 中创建 task
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(_run())
            else:
                loop.run_until_complete(_run())
        except RuntimeError:
            asyncio.run(_run())
        
        # 转换为 WAV (PCM 16-bit) 以确保精确时长读取
        _convert_to_wav(str(mp3_path), str(output_path))
        
        # 删除临时 MP3
        mp3_path.unlink(missing_ok=True)
        
        # 实测时长
        duration = get_wav_duration(str(output_path))
        logger.debug(f"  TTS: {duration:.2f}s ← \"{text[:40]}...\"")
        return duration


def _convert_to_wav(input_path: str, output_path: str, sample_rate: int = 24000):
    """用 ffmpeg 将音频转为 PCM WAV"""
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-i", input_path,
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"FFmpeg 转换失败: {result.stderr}")


# ═══════════════════════════════════════════════════════════════
# 发音词典
# ═══════════════════════════════════════════════════════════════

def load_pronunciation_dict(dict_path: str) -> Dict[str, str]:
    """加载发音词典"""
    dict_path = Path(dict_path)
    if not dict_path.exists():
        logger.warning(f"发音词典不存在: {dict_path}")
        return {}
    with open(dict_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"加载发音词典: {len(data)} 条")
    return data


def apply_pronunciation_dict(text: str, p_dict: Dict[str, str]) -> str:
    """长词优先替换"""
    sorted_keys = sorted(p_dict.keys(), key=len, reverse=True)
    result = text
    for key in sorted_keys:
        if key in result:
            result = result.replace(key, p_dict[key])
    return result


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def run_stage2(
    manifest_path: str = "output/default/project_manifest.json",
    output_dir: str = "output/default",
    config: Optional[Dict] = None,
    tts_provider: Optional[TTSProvider] = None,
    regenerate_indices: Optional[List[int]] = None,
) -> str:
    """
    Stage 2 主入口。
    
    Args:
        manifest_path: Stage 0 输出的 project_manifest.json
        output_dir: 输出目录
        config: 全局配置
        tts_provider: TTS 实现（不提供则用 edge-tts 默认）
        regenerate_indices: 仅重跑指定段落（用于人工修正后重跑）
    
    Returns:
        narration_audio_manifest.json 路径
    """
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 读取 manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    project_id = manifest["project_id"]
    segments = manifest["narration_segments"]
    
    # 确定要处理的段落
    if regenerate_indices:
        target_segments = [s for s in segments if s["index"] in regenerate_indices]
        logger.info(f"单段重跑: {regenerate_indices}")
    else:
        target_segments = segments
    
    # 2. 初始化 TTS Provider
    if tts_provider is None:
        tts_cfg = config.get("stage2", {}).get("edge_tts", {}) if config else {}
        tts_provider = EdgeTTSProvider(
            voice=tts_cfg.get("voice", "zh-CN-XiaoxiaoNeural"),
            rate=tts_cfg.get("rate", "+0%"),
            pitch=tts_cfg.get("pitch", "+0Hz"),
        )
    
    provider_name = tts_provider.__class__.__name__
    logger.info("=" * 50)
    logger.info(f"Stage 2: 旁白音频生成 (TTS: {provider_name})")
    logger.info("=" * 50)
    
    # 3. 加载发音词典
    dict_cfg = config.get("stage2", {}).get("pronunciation_dict", "dicts/pronunciation.json") if config else "dicts/pronunciation.json"
    p_dict = load_pronunciation_dict(dict_cfg)
    
    # 4. 逐段合成
    audio_segments = []
    total_duration = 0.0
    t_start = time.time()
    max_retries = config.get("stage2", {}).get("max_retries_per_segment", 3) if config else 3
    
    # 先加载已有 manifest（如果存在）用于单段重跑时保留其他段
    existing_manifest = None
    existing_path = output_dir / "narration_audio_manifest.json"
    if existing_path.exists() and regenerate_indices:
        with open(existing_path, "r", encoding="utf-8") as f:
            existing_manifest = json.load(f)
    
    for seg in target_segments:
        idx = seg["index"]
        original_text = seg["original_text"]
        tts_text = seg["tts_text"]
        
        # 再次应用发音词典（确保词典更新后生效）
        tts_text = apply_pronunciation_dict(tts_text, p_dict)
        
        audio_path = audio_dir / f"seg_{idx:03d}.wav"
        
        logger.info(f"  合成段 {idx}: [{len(tts_text)}字] {tts_text[:50]}...")
        
        success = False
        duration = 0.0
        
        for attempt in range(max_retries):
            try:
                duration = tts_provider.synthesize(tts_text, str(audio_path))
                success = True
                break
            except Exception as e:
                logger.warning(f"    TTS 失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"    段 {idx} TTS 彻底失败")
        
        audio_segments.append({
            "index": idx,
            "audio_file": f"audio/seg_{idx:03d}.wav",
            "duration": round(duration, 3),
            "tts_text": tts_text,
            "original_text": original_text,
            "needs_regenerate": not success,
            "issue": "" if success else f"TTS 失败 ({max_retries}次重试后)",
        })
        
        total_duration += duration
    
    # 合并已有段落（单段重跑场景）
    if existing_manifest and regenerate_indices:
        # 用新的替换旧的
        existing_map = {s["index"]: s for s in existing_manifest["segments"]}
        for seg in audio_segments:
            existing_map[seg["index"]] = seg
        audio_segments = sorted(existing_map.values(), key=lambda x: x["index"])
        total_duration = sum(s["duration"] for s in audio_segments)
    
    t_elapsed = time.time() - t_start
    
    # 5. 实测 vs 估算对照
    from narrateflow.utils.audio import estimate_audio_duration_never_use
    estimated_total = sum(
        estimate_audio_duration_never_use(s["original_text"])
        for s in segments
    )
    
    # 6. 组装 manifest
    narration_manifest = {
        "project_id": project_id,
        "tts_model": provider_name,
        "sample_rate": 24000,  # edge-tts 默认
        "generated_at": datetime.now().isoformat(),
        "stats": {
            "total_segments": len(segments),
            "generated_segments": len([s for s in audio_segments if not s["needs_regenerate"]]),
            "failed_segments": len([s for s in audio_segments if s["needs_regenerate"]]),
            "total_duration": round(total_duration, 3),
            "estimated_duration": round(estimated_total, 3),
            "duration_ratio": round(total_duration / estimated_total, 3) if estimated_total > 0 else 0,
            "elapsed_seconds": round(t_elapsed, 1),
        },
        "segments": audio_segments,
    }
    
    # 7. Schema 校验
    is_valid, error = validate_json(narration_manifest, "narration_audio_manifest")
    if not is_valid:
        logger.warning(f"⚠ Schema 校验未通过: {error}")
    
    # 8. 输出
    output_path = output_dir / "narration_audio_manifest.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(narration_manifest, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ narration_audio_manifest.json 已输出: {output_path}")
    logger.info(f"  音频总时长: {total_duration:.1f}s (估算: {estimated_total:.1f}s)")
    logger.info(f"  实测/估算比: {narration_manifest['stats']['duration_ratio']}")
    logger.info(f"  成功: {narration_manifest['stats']['generated_segments']}/{len(segments)}")
    
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
# CLI 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    manifest = sys.argv[1] if len(sys.argv) > 1 else "output/default/project_manifest.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "output/default"
    
    try:
        result = run_stage2(manifest, output)
        print(f"\n✓ Stage 2 完成: {result}")
    except Exception as e:
        logger.error(f"Stage 2 失败: {e}", exc_info=True)
        sys.exit(1)
