"""
Stage 3: 时间轴对齐（纯 CPU）
P0: 顺序匹配 + 时长匹配（简化版）
P2: 7 维评分 + 滑窗 + 完整重定时策略
输出: output/{project_id}/timeline_plan.ai.json
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from narrateflow.utils.schema_validator import validate_json

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# P0 简化版匹配：顺序 + 时长
# ═══════════════════════════════════════════════════════════════

def _order_score(narration_idx: int, window_idx: int, total_windows: int) -> float:
    """
    顺序匹配得分。
    旁白第 i 段配视频第 j 个窗口，位置越接近 i*N/M 得分越高。
    """
    expected_window = narration_idx * total_windows / max(1, total_windows)
    distance = abs(window_idx - expected_window) / max(1, total_windows)
    return max(0.0, 1.0 - distance)


def _duration_score(audio_duration: float, window_duration: float) -> float:
    """
    时长匹配得分。
    音频与窗口时长越接近得分越高。
    """
    if audio_duration <= 0 or window_duration <= 0:
        return 0.0
    ratio = min(audio_duration, window_duration) / max(audio_duration, window_duration)
    return ratio  # 0~1


def match_segments_simple(
    narration_segments: List[Dict],
    windows: List[Dict],
    order_weight: float = 0.5,
    duration_weight: float = 0.5,
    min_score: float = 0.3,
    video_duration: float = 0,
) -> List[Dict]:
    """
    P0 简化版匹配：按音频时长比例分配视频时间轴。

    每个旁白段获取与其音频时长成比例的视频区间。
    保持原始顺序，允许多个窗口合并。

    Returns:
        [{narration_index, matched_windows, video_start, video_end, score, retime_strategy, ...}]
    """
    if video_duration <= 0:
        video_duration = windows[-1]["end_time"] if windows else 0

    # 计算总音频时长
    total_audio = sum(seg.get("duration", 0) for seg in narration_segments)
    if total_audio <= 0:
        total_audio = 1

    plans = []
    current_video_time = 0.0

    for seg in narration_segments:
        n_idx = seg["index"]
        audio_dur = seg.get("duration", 0)

        # 按比例分配视频区间
        video_budget = video_duration * (audio_dur / total_audio)
        video_start = current_video_time
        video_end = min(video_duration, current_video_time + video_budget)

        # 找到区间覆盖的视频窗口
        matched_windows = []
        for w in windows:
            w_start = w["start_time"]
            w_end = w["end_time"]
            # 检查窗口是否与我们的区间有重叠
            if w_end > video_start and w_start < video_end:
                matched_windows.append(w["window_id"])

        if not matched_windows:
            # 找最近的窗口
            closest = min(windows, key=lambda w: abs(w["start_time"] - video_start))
            matched_windows = [closest["window_id"]]

        # 评分（简化：顺序天然匹配，时长看预算覆盖）
        order_score = 1.0  # 顺序天然正确
        duration_score = min(audio_dur, video_budget) / max(audio_dur, video_budget) if max(audio_dur, video_budget) > 0 else 1.0
        score = order_weight * order_score + duration_weight * duration_score

        # 构建段计划（传入匹配窗口信息以便确定重定时策略）
        # 将匹配窗口组装为类似单窗口的格式，用 merged duration
        matched_windows_data = [w for w in windows if w["window_id"] in matched_windows]
        if matched_windows_data:
            merged_start = min(w["start_time"] for w in matched_windows_data)
            merged_end = max(w["end_time"] for w in matched_windows_data)
            merged_window = {
                "window_id": matched_windows[0],
                "start_time": merged_start,
                "end_time": merged_end,
                "duration": merged_end - merged_start,
            }
        else:
            merged_window = windows[0] if windows else {"start_time": 0, "end_time": 0, "duration": 0}

        plan = _build_segment_plan(
            n_idx, audio_dur, [merged_window], score,
            video_start=video_start, video_end=video_end
        )
        plan["matched_windows"] = matched_windows

        current_video_time = video_end
        plans.append(plan)

    return plans


def _build_segment_plan(
    n_idx: int, audio_dur: float, matched_windows: List[Dict], score: float,
    video_start: Optional[float] = None, video_end: Optional[float] = None,
) -> Dict:
    """构建单个段落的对齐方案"""
    if not matched_windows:
        return {
            "narration_index": n_idx,
            "video_start": 0,
            "video_end": audio_dur,
            "matched_windows": [],
            "score": 0,
            "retime_strategy": "manual_review",
            "needs_review": True,
        }

    win = matched_windows[0]
    if video_start is None:
        video_start = win["start_time"]
    if video_end is None:
        video_end = win["end_time"]
    window_dur = win["duration"]
    
    # 确定重定时策略
    retime, speed_factor, freeze_point, needs_review = _determine_retime_strategy(
        audio_dur, window_dur, video_start, video_end
    )
    
    result = {
        "narration_index": n_idx,
        "video_start": video_start,
        "video_end": video_end,
        "audio_start": 0.0,  # 将在全局时间轴计算时填充
        "audio_end": audio_dur,
        "matched_windows": [win["window_id"]],
        "score": round(score, 3),
        "retime_strategy": retime,
        "needs_review": needs_review,
    }
    # 可选字段仅在非 None 时输出
    if speed_factor is not None:
        result["speed_factor"] = speed_factor
    if freeze_point is not None:
        result["freeze_point"] = freeze_point
    return result


def _determine_retime_strategy(
    audio_dur: float, window_dur: float, 
    video_start: float, video_end: float,
    duration_tolerance: float = 0.5,
    speed_range: Tuple[float, float] = (0.90, 1.10),
    freeze_max: float = 0.8,
) -> Tuple[str, Optional[float], Optional[float], bool]:
    """
    根据音视频时长差决定重定时策略。
    
    Returns:
        (strategy, speed_factor, freeze_point, needs_review)
    """
    diff = audio_dur - window_dur

    # 策略 1: 原速播放（≤0.5s 差）
    if abs(diff) <= duration_tolerance:
        return ("normal", 1.0, None, False)

    # 策略 2+3: 音频更长 → 视频需要变长（慢放/定格/合并窗口）
    if diff > 0:
        # 轻微变速（视频慢放到匹配音频长度）
        speed_factor = window_dur / audio_dur  # < 1.0 = 慢放
        if speed_range[0] <= speed_factor <= speed_range[1]:
            return ("speed_change", round(speed_factor, 3), None, False)
        # 短定格
        if diff <= freeze_max:
            freeze_point = video_end  # 最后一帧定格
            return ("freeze", 1.0, round(freeze_point, 3), False)
        # 超出范围 — 需要合并更多窗口或人工审核
        return ("manual_review", 1.0, None, True)

    # 策略 4: 视频预算 > 音频时长 → 原速播放，视频截断即可
    # 不需要变速压缩，因为旁白播完就结束了
    if diff < 0:
        # 视频更长 = 正常，只需按音频长度截断视频
        return ("normal", 1.0, None, False)

    return ("normal", 1.0, None, False)


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def run_stage3(
    manifest_path: str = "output/default/project_manifest.json",
    video_meta_path: str = "output/default/video_meta.ai.json",
    audio_manifest_path: str = "output/default/narration_audio_manifest.json",
    output_dir: str = "output/default",
    config: Optional[Dict] = None,
) -> str:
    """
    Stage 3 主入口。
    
    优先读取 *.human.json，不存在则用 *.ai.json
    
    Returns:
        timeline_plan.ai.json 路径
    """
    output_dir = Path(output_dir)
    
    # 优先读取人工修正版
    human_video_meta = output_dir / "video_meta.human.json"
    if human_video_meta.exists():
        video_meta_path = str(human_video_meta)
        logger.info("使用人工修正 video_meta.human.json")
    else:
        logger.info("使用 AI 生成 video_meta.ai.json")
    
    # 1. 加载数据
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(video_meta_path, "r", encoding="utf-8") as f:
        video_meta = json.load(f)
    with open(audio_manifest_path, "r", encoding="utf-8") as f:
        audio_manifest = json.load(f)
    
    project_id = manifest["project_id"]
    
    # 合并音频时长到 narration_segments
    audio_map = {s["index"]: s for s in audio_manifest["segments"]}
    narration_segments = manifest["narration_segments"]
    for seg in narration_segments:
        audio_info = audio_map.get(seg["index"], {})
        seg["duration"] = audio_info.get("duration", 0)
    
    windows = video_meta["windows"]
    
    stage3_cfg = config.get("stage3", {}) if config else {}
    scoring_cfg = stage3_cfg.get("scoring_weights", {})
    retime_cfg = stage3_cfg.get("retime", {})
    
    logger.info("=" * 50)
    logger.info("Stage 3: 时间轴对齐")
    logger.info("=" * 50)
    logger.info(f"  视频窗口: {len(windows)}, 旁白段: {len(narration_segments)}")
    logger.info(f"  视频时长: {video_meta['stats']['video_duration']}s")
    logger.info(f"  音频总时长: {audio_manifest['stats']['total_duration']}s")
    
    # 2. P0 简化匹配 — 按比例分配视频时间轴
    plans = match_segments_simple(
        narration_segments,
        windows,
        order_weight=scoring_cfg.get("order", 0.5),
        duration_weight=scoring_cfg.get("duration", 0.5),
        min_score=stage3_cfg.get("min_score_threshold", 0.3),
        video_duration=video_meta["stats"]["video_duration"],
    )
    
    # 3. 计算全局时间轴
    current_audio_time = 0.0
    for plan in plans:
        plan["audio_start"] = round(current_audio_time, 3)
        plan["audio_end"] = round(current_audio_time + plan["audio_end"], 3)
        current_audio_time += (plan["audio_end"] - plan["audio_start"])
    
    # 4. 统计
    strategies = {}
    needs_review_count = 0
    for plan in plans:
        s = plan["retime_strategy"]
        strategies[s] = strategies.get(s, 0) + 1
        if plan["needs_review"]:
            needs_review_count += 1
    
    # 5. 组装输出
    timeline = {
        "project_id": project_id,
        "generated_at": datetime.now().isoformat(),
        "source": {
            "video_meta": video_meta_path,
            "audio_manifest": audio_manifest_path,
        },
        "stats": {
            "total_segments": len(plans),
            "video_duration": video_meta["stats"]["video_duration"],
            "audio_total_duration": audio_manifest["stats"]["total_duration"],
            "strategies": strategies,
            "needs_review_count": needs_review_count,
        },
        "total_duration": round(
            audio_manifest["stats"]["total_duration"],
            3,
        ),
        "segments": plans,
    }
    
    # 6. Schema 校验
    is_valid, error = validate_json(timeline, "timeline_plan")
    if not is_valid:
        logger.warning(f"⚠ Schema 校验未通过: {error}")
    
    # 7. 输出
    output_path = output_dir / "timeline_plan.ai.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ timeline_plan.ai.json 已输出: {output_path}")
    logger.info(f"  策略分布: {strategies}")
    logger.info(f"  需人工审核: {needs_review_count}/{len(plans)}")
    
    # 打印对齐预览
    print_alignment_preview(plans, windows)
    
    return str(output_path)


def print_alignment_preview(plans: List[Dict], windows: List[Dict]):
    """打印对齐预览表格"""
    logger.info("-" * 80)
    logger.info(f"{'段':>3} {'视频时间':>12} {'音频时长':>8} {'窗口时长':>8} {'策略':>14} {'得分':>6} {'审核':>4}")
    logger.info("-" * 80)
    
    for plan in plans:
        n_idx = plan["narration_index"]
        audio_dur = plan["audio_end"] - plan["audio_start"]
        v_start = plan["video_start"]
        v_end = plan["video_end"]
        window_dur = v_end - v_start
        strategy = plan["retime_strategy"]
        score = plan["score"]
        review = "⚠" if plan["needs_review"] else "✓"
        
        logger.info(
            f"{n_idx:>3} {v_start:>5.1f}s-{v_end:>5.1f}s "
            f"{audio_dur:>6.1f}s {window_dur:>6.1f}s {strategy:>14} "
            f"{score:>5.2f} {review:>4}"
        )
    logger.info("-" * 80)


# ═══════════════════════════════════════════════════════════════
# CLI 测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    manifest = sys.argv[1] if len(sys.argv) > 1 else "output/default/project_manifest.json"
    video_meta = sys.argv[2] if len(sys.argv) > 2 else "output/default/video_meta.ai.json"
    audio_manifest = sys.argv[3] if len(sys.argv) > 3 else "output/default/narration_audio_manifest.json"
    output = sys.argv[4] if len(sys.argv) > 4 else "output/default"
    
    try:
        result = run_stage3(manifest, video_meta, audio_manifest, output)
        print(f"\n✓ Stage 3 完成: {result}")
    except Exception as e:
        logger.error(f"Stage 3 失败: {e}", exc_info=True)
        sys.exit(1)
