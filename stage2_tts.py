"""
Stage 2: 旁白音频生成（TTS 独占）
P3: 发音词典 + 单段重跑 + VoxCPM2 接口预留
输出: output/{project_id}/audio/*.wav + narration_audio_manifest.json
"""
import json
import logging
import time
import asyncio
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from narrateflow.utils.audio import get_wav_duration, concat_audio_plan
from narrateflow.utils.schema_validator import validate_json
from narrateflow.providers.tts_base import TTSProvider

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# EdgeTTS Provider (实现 TTSProvider 接口)
# ═══════════════════════════════════════════════════════════════

class EdgeTTSProvider(TTSProvider):
    """Edge TTS Provider — 通过 Microsoft Edge TTS API 合成"""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural",
                 rate: str = "+0%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    @property
    def name(self) -> str:
        return f"edge-tts:{self.voice}"

    @property
    def sample_rate(self) -> int:
        return 24000  # edge-tts 默认输出

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
                voice=kwargs.get("voice", self.voice),
                rate=kwargs.get("rate", self.rate),
                pitch=kwargs.get("pitch", self.pitch),
            )
            await communicate.save(str(mp3_path))

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
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
# 发音词典 (P3: 75 条目)
# ═══════════════════════════════════════════════════════════════

def load_pronunciation_dict(dict_path: str) -> Tuple[Dict[str, str], str]:
    """加载发音词典，返回 (词典, 内容哈希) 用于缓存失效"""
    dict_path = Path(dict_path)
    if not dict_path.exists():
        # 尝试相对于 narrateflow 包的路径
        alt_path = Path(__file__).parent / dict_path
        if alt_path.exists():
            dict_path = alt_path
        else:
            logger.warning(f"发音词典不存在: {dict_path} (also tried {alt_path})")
            return {}, ""
    with open(dict_path, "r", encoding="utf-8") as f:
        raw = f.read()
    data = json.loads(raw)
    # 去除 _comment 等元数据键
    entries = {k: v for k, v in data.items() if not k.startswith("_")}
    hash_val = hashlib.md5(raw.encode()).hexdigest()[:8]
    logger.info(f"加载发音词典: {len(entries)} 条 (hash={hash_val})")
    return entries, hash_val


def apply_pronunciation_dict(text: str, p_dict: Dict[str, str]) -> str:
    """长词优先替换，避免短词误匹配"""
    sorted_keys = sorted(p_dict.keys(), key=len, reverse=True)
    result = text
    for key in sorted_keys:
        if key in result:
            result = result.replace(key, p_dict[key])
    return result


# ═══════════════════════════════════════════════════════════════
# 发音缓存 — 词典更新后自动重跑受影响段落
# ═══════════════════════════════════════════════════════════════

def _load_dict_cache(cache_path: Path) -> Optional[str]:
    """读取上次使用的词典哈希"""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f).get("dict_hash", "")
    return None


def _save_dict_cache(cache_path: Path, dict_hash: str):
    """保存词典哈希"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"dict_hash": dict_hash, "updated_at": datetime.now().isoformat()}, f)


def _find_affected_segments(
    segments: List[Dict], p_dict: Dict[str, str]
) -> List[int]:
    """找出受发音词典影响的段落索引（词典更新后自动检测）"""
    affected = []
    for seg in segments:
        text = seg.get("tts_text", seg.get("original_text", ""))
        for key in p_dict:
            if key in text:
                affected.append(seg["index"])
                break
    return affected


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
    Stage 2 主入口 — P3 增强版。

    新增 P3 功能:
    - 发音词典（75条）+ 缓存哈希，词典更新后自动重跑受影响段落
    - 单段重跑（regenerate_indices），其他段落不受影响
    - 长文本语气一致性（同一次 provider 会话内保持音色）

    Args:
        manifest_path: Stage 0 输出的 project_manifest.json
        output_dir: 输出目录
        config: 全局配置
        tts_provider: TTS 实现（不提供则自动选择）
        regenerate_indices: 仅重跑指定段落（人工修正后重跑）

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

    # 2. 初始化 TTS Provider（P3: 自动选择最佳可用方案）
    if tts_provider is None:
        stage2_cfg = config.get("stage2", {}) if config else {}
        provider_type = stage2_cfg.get("tts_provider", "edge_tts")

        if provider_type == "voxcpm2":
            try:
                from narrateflow.providers.tts_voxcpm2 import VoxCPM2Provider
                vox_cfg = stage2_cfg.get("voxcpm2", {})
                tts_provider = VoxCPM2Provider(
                    model_path=vox_cfg.get("model_path"),
                    dtype=vox_cfg.get("dtype"),  # Mac 自动 fp16
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

    provider_name = tts_provider.name if hasattr(tts_provider, 'name') else tts_provider.__class__.__name__
    logger.info("=" * 60)
    logger.info(f"Stage 2: 旁白音频生成 (TTS: {provider_name})")
    logger.info("=" * 60)

    # 3. 发音词典 + 缓存管理
    dict_cfg = config.get("stage2", {}).get("pronunciation_dict", "dicts/pronunciation.json") if config else "dicts/pronunciation.json"
    p_dict, dict_hash = load_pronunciation_dict(dict_cfg)

    # 检测词典是否已更新 → 自动重跑受影响段落
    dict_cache_path = output_dir / ".dict_cache.json"
    old_hash = _load_dict_cache(dict_cache_path)

    auto_regenerate = []
    if dict_hash and old_hash and dict_hash != old_hash and not regenerate_indices:
        auto_regenerate = _find_affected_segments(segments, p_dict)
        if auto_regenerate:
            logger.info(f"发音词典已更新 (hash: {old_hash}→{dict_hash})")
            logger.info(f"  受影响段落: {auto_regenerate}")
    _save_dict_cache(dict_cache_path, dict_hash)

    # 合并手动+自动重跑
    if regenerate_indices is None:
        regenerate_indices = []
    regenerate_indices = list(set(regenerate_indices) | set(auto_regenerate))

    # 确定要处理的段落
    if regenerate_indices:
        target_segments = [s for s in segments if s["index"] in regenerate_indices]
        logger.info(f"单段重跑 (手动+自动): {sorted(regenerate_indices)}")
    else:
        target_segments = segments

    # 4. 逐段合成（P3: 单段重跑时保留其他段不受影响）
    audio_segments = []
    total_duration = 0.0
    t_start = time.time()
    max_retries = config.get("stage2", {}).get("max_retries_per_segment", 3) if config else 3

    # 先加载已有 manifest（用于单段重跑时保留其他段）
    existing_manifest = None
    existing_path = output_dir / "narration_audio_manifest.json"
    if existing_path.exists() and regenerate_indices:
        with open(existing_path, "r", encoding="utf-8") as f:
            existing_manifest = json.load(f)
        # 从已有数据中保留未重跑的段
        for seg in existing_manifest.get("segments", []):
            if seg["index"] not in regenerate_indices:
                audio_segments.append(seg)
                total_duration += seg.get("duration", 0)

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
        last_error = ""

        for attempt in range(max_retries):
            try:
                duration = tts_provider.synthesize(tts_text, str(audio_path))
                success = True
                break
            except Exception as e:
                last_error = str(e)
                logger.warning(f"    TTS 失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time as _time
                    _time.sleep(1)

        if not success:
            logger.error(f"    段 {idx} TTS 彻底失败: {last_error}")

        seg_result = {
            "index": idx,
            "audio_file": f"audio/seg_{idx:03d}.wav",
            "duration": round(duration, 3),
            "tts_text": tts_text,
            "original_text": original_text,
            "needs_regenerate": not success,
            "issue": "" if success else f"TTS 失败 ({max_retries}次重试): {last_error[:100]}",
        }

        # 单段重跑: 替换同索引旧条目
        existing_idx = next((i for i, s in enumerate(audio_segments) if s["index"] == idx), None)
        if existing_idx is not None:
            audio_segments[existing_idx] = seg_result
        else:
            audio_segments.append(seg_result)

        if success:
            total_duration += duration
        elif existing_idx is not None:
            # 重跑失败: 保留旧时长
            total_duration += audio_segments[existing_idx].get("duration", 0)

    # 按索引排序
    audio_segments.sort(key=lambda x: x["index"])

    t_elapsed = time.time() - t_start

    # 5. 实测 vs 估算对照
    from narrateflow.utils.audio import estimate_audio_duration_never_use
    estimated_total = sum(
        estimate_audio_duration_never_use(s["original_text"])
        for s in segments
    )

    # 6. 长文本语气一致性记录
    total_segments = len(segments)
    generated = len([s for s in audio_segments if not s["needs_regenerate"]])
    failed = len([s for s in audio_segments if s["needs_regenerate"]])

    # 7. 组装 manifest
    narration_manifest = {
        "project_id": project_id,
        "tts_model": provider_name,
        "sample_rate": getattr(tts_provider, 'sample_rate', 24000),
        "generated_at": datetime.now().isoformat(),
        "pronunciation": {
            "dict_path": str(dict_cfg),
            "dict_hash": dict_hash,
            "entries_count": len(p_dict),
        },
        "stats": {
            "total_segments": total_segments,
            "generated_segments": generated,
            "failed_segments": failed,
            "total_duration": round(total_duration, 3),
            "estimated_duration": round(estimated_total, 3),
            "duration_ratio": round(total_duration / estimated_total, 3) if estimated_total > 0 else 0,
            "elapsed_seconds": round(t_elapsed, 1),
            "regenerated_indices": sorted(regenerate_indices) if regenerate_indices else [],
            "auto_regenerated_from_dict_update": sorted(auto_regenerate),
        },
        "segments": audio_segments,
    }

    # 8. Schema 校验 & 输出
    is_valid, error = validate_json(narration_manifest, "narration_audio_manifest")
    if not is_valid:
        logger.warning(f"⚠ Schema 校验未通过: {error}")

    output_path = output_dir / "narration_audio_manifest.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(narration_manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ narration_audio_manifest.json 已输出: {output_path}")
    logger.info(f"  音频总时长: {total_duration:.1f}s (估算: {estimated_total:.1f}s)")
    logger.info(f"  实测/估算比: {narration_manifest['stats']['duration_ratio']}")
    logger.info(f"  成功: {generated}/{total_segments}")
    if auto_regenerate:
        logger.info(f"  词典更新自动重跑: {auto_regenerate}")
    if failed > 0:
        logger.warning(f"  失败: {failed} 段")

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
