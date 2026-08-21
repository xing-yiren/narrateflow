"""
Stage 0: 资产探测与文稿解析
输入: input/video.mp4 + input/narration.txt
输出: output/{project_id}/project_manifest.json
"""
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from narrateflow.utils.ffprobe import get_video_info
from narrateflow.utils.schema_validator import validate_json

logger = logging.getLogger(__name__)

# ─── 数字转中文 ─────────────────────────────────────────────
_NUM_MAP = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
}

_UNIT_MAP = ['', '十', '百', '千', '万', '十万', '百万', '千万', '亿']


def _num_to_chinese(num_str: str) -> str:
    """简单阿拉伯数字转中文（整数），不做复杂逻辑"""
    try:
        num = int(num_str)
    except ValueError:
        return num_str
    if num == 0:
        return '零'
    if num < 0:
        return '负' + _num_to_chinese(str(-num))
    
    result = []
    digits = list(str(num))
    length = len(digits)
    
    for i, d in enumerate(digits):
        digit = int(d)
        unit_idx = length - i - 1
        if digit == 0:
            # 避免连续的零
            if result and result[-1] != '零':
                result.append('零')
        else:
            if unit_idx < len(_UNIT_MAP):
                result.append(_NUM_MAP[d])
                if unit_idx > 0:
                    result.append(_UNIT_MAP[unit_idx])
            else:
                result.append(_NUM_MAP[d])
    
    text = ''.join(result).rstrip('零')
    # 处理 "一十" -> "十"
    if text.startswith('一十'):
        text = text[1:]
    return text


def number_to_chinese(text: str) -> str:
    """将文本中的阿拉伯数字转换为中文读法"""
    # 匹配独立的数字（不跟在字母后面）
    def repl(match):
        num = match.group(0)
        return _num_to_chinese(num)
    
    return re.sub(r'(?<![a-zA-Z])\d+(?![a-zA-Z])', repl, text)


# ─── 发音词典 ───────────────────────────────────────────────
def load_pronunciation_dict(dict_path: str) -> Dict[str, str]:
    """加载发音词典"""
    dict_path = Path(dict_path)
    if not dict_path.exists():
        logger.warning(f"发音词典不存在: {dict_path}，跳过")
        return {}
    
    with open(dict_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"加载发音词典: {len(data)} 条")
    return data


def apply_pronunciation_dict(text: str, p_dict: Dict[str, str]) -> str:
    """对文本应用发音词典替换（长词优先）"""
    # 按 key 长度降序排序，确保长词先替换
    sorted_keys = sorted(p_dict.keys(), key=len, reverse=True)
    result = text
    for key in sorted_keys:
        if key in result:
            result = result.replace(key, p_dict[key])
    return result


# ─── 文稿分句 ───────────────────────────────────────────────
def split_narration(text: str, min_chars: int = 25, max_chars: int = 60) -> List[str]:
    """
    将旁白文稿按标点拆分为适合旁白的段落。
    每段控制在 min_chars ~ max_chars 个中文字符。
    """
    # 按句末标点拆分
    sentences = re.split(r'(?<=[。！？；\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 合并短句，拆分长句
    segments = []
    buffer = ""
    
    for sentence in sentences:
        combined = buffer + sentence
        char_count = _count_chinese(combined)
        
        if char_count < min_chars:
            # 太短，累积到 buffer
            buffer = combined
        elif char_count > max_chars:
            # 如果 buffer 已足够，先输出 buffer
            if buffer and _count_chinese(buffer) >= min_chars:
                segments.append(buffer)
                buffer = ""
            # 对当前长句进一步拆分
            sub_segments = _split_long_sentence(combined, max_chars)
            # 最后一个可能不够长，留到 buffer
            if len(sub_segments) > 1:
                for sub in sub_segments[:-1]:
                    segments.append(sub)
                buffer = sub_segments[-1]
            else:
                segments.append(combined)
                buffer = ""
        else:
            # 合适长度，输出
            segments.append(combined)
            buffer = ""
    
    # 处理残留
    if buffer:
        if segments and _count_chinese(buffer) < min_chars:
            # 合并到最后一段
            segments[-1] = segments[-1] + buffer
        else:
            segments.append(buffer)
    
    logger.info(f"文稿分句: 原文 {_count_chinese(text)} 字符 → {len(segments)} 段")
    for i, seg in enumerate(segments):
        logger.debug(f"  段{i}: [{_count_chinese(seg)}字] {seg[:50]}...")
    
    return segments


def _count_chinese(text: str) -> int:
    """统计中文字符数（含中文标点）"""
    return len(re.findall(r'[一-鿿㐀-䶿豈-﫿　-〿＀-￯]', text))


def _split_long_sentence(text: str, max_chars: int) -> List[str]:
    """对长句按逗号/分号进一步拆分"""
    parts = re.split(r'(?<=[，,、\s])', text)
    segments = []
    buffer = ""
    
    for part in parts:
        if _count_chinese(buffer + part) > max_chars and buffer:
            segments.append(buffer)
            buffer = part
        else:
            buffer += part
    
    if buffer:
        segments.append(buffer)
    
    return segments


# ─── 主流程 ─────────────────────────────────────────────────
def run_stage0(
    video_path: str = "input/video.mp4",
    narration_path: str = "input/narration.txt",
    output_dir: str = "output/default",
    config: Optional[Dict] = None,
) -> str:
    """
    Stage 0 主入口。
    
    返回: project_manifest.json 的路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = output_dir / "project_manifest.json"
    
    # 1. 获取视频真实元信息
    logger.info("=" * 50)
    logger.info("Stage 0: 资产探测与文稿解析")
    logger.info("=" * 50)
    
    ffprobe_cfg = config.get("stage0", {}).get("ffprobe_path", "ffprobe") if config else "ffprobe"
    video_info = get_video_info(video_path, ffprobe_cfg)
    logger.info(f"✓ 视频探测完成: {video_info['duration']}s, {video_info['fps']}fps, "
                f"{video_info['width']}x{video_info['height']}")
    
    # 2. 读取旁白文稿
    narration_path = Path(narration_path)
    if not narration_path.exists():
        raise FileNotFoundError(f"旁白文稿不存在: {narration_path}")
    
    with open(narration_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    
    logger.info(f"✓ 文稿读取: {len(raw_text)} 字符")
    
    # 3. 加载发音词典
    stage0_cfg = config.get("stage0", {}) if config else {}
    min_chars = stage0_cfg.get("sentence_min_chars", 25)
    max_chars = stage0_cfg.get("sentence_max_chars", 60)
    
    dict_cfg = config.get("stage2", {}).get("pronunciation_dict", "dicts/pronunciation.json") if config else "dicts/pronunciation.json"
    p_dict = load_pronunciation_dict(dict_cfg)
    
    # 4. 分句
    raw_segments = split_narration(raw_text, min_chars, max_chars)
    
    # 5. 规范化：数字转中文 + 发音词典替换
    narration_segments = []
    for i, text in enumerate(raw_segments):
        # 保留 original_text
        original = text
        # 数字转中文
        tts_text = number_to_chinese(text)
        # 发音词典替换
        tts_text = apply_pronunciation_dict(tts_text, p_dict)
        # 标记是否有特殊处理
        has_special = (original != tts_text)
        
        narration_segments.append({
            "index": i,
            "original_text": original,
            "tts_text": tts_text,
            "char_count": _count_chinese(tts_text),
            "has_special_terms": has_special,
        })
    
    logger.info(f"✓ 文稿规范化: {len(narration_segments)} 段")
    
    # 6. 组装 manifest
    manifest = {
        "project_id": output_dir.name,
        "created_at": datetime.now().isoformat(),
        "video": video_info,
        "narration": {
            "source_file": str(narration_path),
            "source_text": raw_text,
            "total_chars": _count_chinese(raw_text),
            "total_segments": len(narration_segments),
        },
        "narration_segments": narration_segments,
    }
    
    # 7. Schema 校验
    is_valid, error = validate_json(manifest, "project_manifest")
    if not is_valid:
        logger.warning(f"⚠ Schema 校验未通过: {error}")
    
    # 8. 输出
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ project_manifest.json 已输出: {manifest_path}")
    logger.info(f"  Video: {video_info['duration']}s, {video_info['width']}x{video_info['height']}")
    logger.info(f"  Segments: {len(narration_segments)}")
    
    return str(manifest_path)


# ─── CLI 测试入口 ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    video = sys.argv[1] if len(sys.argv) > 1 else "input/video.mp4"
    narration = sys.argv[2] if len(sys.argv) > 2 else "input/narration.txt"
    output = sys.argv[3] if len(sys.argv) > 3 else "output/default"
    
    try:
        manifest_path = run_stage0(video, narration, output)
        print(f"\n✓ Stage 0 完成: {manifest_path}")
    except Exception as e:
        logger.error(f"Stage 0 失败: {e}")
        sys.exit(1)
