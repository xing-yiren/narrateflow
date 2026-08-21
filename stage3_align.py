"""
Stage 3: 时间轴对齐（纯 CPU）
P2: 7 维评分 + 滑窗组合 + 完整重定时策略 + 人工锁定点
输出: output/{project_id}/timeline_plan.ai.json
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from narrateflow.utils.schema_validator import validate_json
from narrateflow.utils.similarity import (
    TextMatcher, build_window_index,
    physical_tag_score, ocr_match_score,
    action_match_score, mood_match_score,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# 默认权重 (config.yaml 可覆盖)
# ══════════════════════════════════════════════════════════════
DEFAULT_WEIGHTS = {
    "order": 0.25,
    "duration": 0.25,
    "semantic": 0.20,
    "physical_tags": 0.10,
    "ocr": 0.10,
    "action": 0.05,
    "mood": 0.05,
}

# ══════════════════════════════════════════════════════════════
# 7 维单窗口评分
# ══════════════════════════════════════════════════════════════

def score_window_pair(
    narration_text: str,
    narration_idx: int,
    audio_duration: float,
    window: dict,
    window_idx: int,
    total_windows: int,
    semantic_similarity: float,
    weights: Dict[str, float],
) -> float:
    """
    对一个旁白段 + 一个视频窗口做 7 维综合评分。

    Args:
        narration_text: 旁白文本
        narration_idx: 旁白段序号
        audio_duration: 音频时长 (秒)
        window: 视频窗口 dict
        window_idx: 窗口位置序号
        total_windows: 总窗口数
        semantic_similarity: 旁白与窗口 visual_summary 的 TF-IDF 余弦相似度
        weights: 7 维权重组

    Returns:
        综合得分 [0..1]
    """

    # 1. 顺序匹配 (25%) — 保持视频原始播放顺序
    expected_win = narration_idx * total_windows / max(1, total_windows)
    order_dist = abs(window_idx - expected_win) / max(1, total_windows)
    order_score = max(0.0, 1.0 - order_dist)

    # 2. 时长匹配 (25%) — 音频时长 vs 窗口时长
    w_dur = window["duration"]
    if audio_duration > 0 and w_dur > 0:
        duration_score = min(audio_duration, w_dur) / max(audio_duration, w_dur)
    else:
        duration_score = 0.5

    # 3. 语义相似 (20%) — 来自 TF-IDF (预计算传入)
    semantic_score = semantic_similarity

    # 4. 物理标签 (10%) — 旁白关键词与画面标签重叠
    pt_score = physical_tag_score(narration_text, window)

    # 5. OCR 匹配 (10%) — 旁白提及的词出现在画面文字中
    ocr_score = ocr_match_score(narration_text, window)

    # 6. 动作匹配 (5%) — 旁白动作描述与检测动作
    act_score = action_match_score(narration_text, window)

    # 7. 意境匹配 (5%) — 情绪/氛围
    md_score = mood_match_score(narration_text, window)

    total = (
        weights.get("order", 0.25) * order_score
        + weights.get("duration", 0.25) * duration_score
        + weights.get("semantic", 0.20) * semantic_score
        + weights.get("physical_tags", 0.10) * pt_score
        + weights.get("ocr", 0.10) * ocr_score
        + weights.get("action", 0.05) * act_score
        + weights.get("mood", 0.05) * md_score
    )

    return round(total, 5)


# ══════════════════════════════════════════════════════════════
# 滑窗组合匹配
# ══════════════════════════════════════════════════════════════

def match_segments_with_sliding_windows(
    narration_segments: List[Dict],
    windows: List[Dict],
    weights: Dict[str, float],
    max_windows_per_para: int = 3,
    min_score_threshold: float = 0.4,
    lock_points: Optional[Dict[int, List[int]]] = None,
) -> List[Dict]:
    """
    P2 滑窗匹配：一个旁白段允许匹配 1~max_windows_per_para 个连续窗口。

    流程：
    1. 对所有 (narration, window) 对计算 7 维评分
    2. 对每个旁白段，尝试 1~K 连续窗口组合，取最高分
    3. 贪心分配（已用窗口不可复用）
    4. 人工锁定点拥有最高优先级

    Args:
        narration_segments: 旁白段落列表
        windows: 视频窗口列表
        weights: 7 维评分权重
        max_windows_per_para: 一个段落最多合并的连续窗口数
        min_score_threshold: 低于此分标记 needs_review
        lock_points: {narration_index: [window_id, ...]} 人工锁定

    Returns:
        [{narration_index, matched_windows, video_start, video_end, score, ...}]
    """
    lock_points = lock_points or {}

    # ━━━ 步骤 0: 人工锁定预处理 ━━━
    locked_windows = set()
    locked_plans = {}  # n_idx → plan

    for n_idx, locked_win_ids in lock_points.items():
        if not locked_win_ids:
            continue
        # 验证窗口存在
        locked_data = [w for w in windows if w["window_id"] in locked_win_ids]
        if not locked_data:
            logger.warning(f"人工锁定窗口无效: narration={n_idx}, windows={locked_win_ids}")
            continue
        for w in locked_data:
            locked_windows.add(w["window_id"])
        # 锁定段暂存，后续插入
        locked_plans[n_idx] = {
            "locked": True,
            "locked_window_ids": locked_win_ids,
            "locked_data": locked_data,
        }

    # ━━━ 步骤 1: 建 TF-IDF 语义索引 ━━━
    matcher = build_window_index(windows)

    # 计算所有 (narration, window) 对的语义相似度
    semantic_sims: Dict[Tuple[int, int], float] = {}
    for seg in narration_segments:
        n_idx = seg["index"]
        # 检查是否为锁定段（锁定段不需要语义匹配）
        if n_idx in locked_plans:
            continue
        text = seg.get("tts_text", seg["original_text"])
        sims = matcher.query_similarities(text)
        for w_idx, sim_val in enumerate(sims):
            semantic_sims[(n_idx, w_idx)] = round(float(sim_val), 4)

    # ━━━ 步骤 2: 滑窗评分 ━━━
    # 对每个旁白段，计算所有可能的连续窗口组合的得分
    segment_candidates: List[Dict] = []  # 可用的候选匹配
    locked_assignments: Dict[int, Dict] = {}  # 锁定段的最终分配

    for seg in narration_segments:
        n_idx = seg["index"]
        audio_dur = seg.get("duration", 0)
        text = seg.get("tts_text", seg.get("original_text", ""))

        # 人工锁定 → 最高优先级，直接分配
        if n_idx in locked_plans:
            lp = locked_plans[n_idx]
            plan = _build_combo_plan_from_lock(
                n_idx, audio_dur, lp["locked_data"],
                max_score=1.0, narration_text=text, windows=windows,
            )
            locked_assignments[n_idx] = plan
            continue

        total_windows = len(windows)
        candidates_for_this_seg = []

        # 尝试 1..K 连续窗口组合
        for combo_size in range(1, max_windows_per_para + 1):
            for start_w in range(total_windows - combo_size + 1):
                combo_ids = list(range(start_w, start_w + combo_size))

                # 跳过已被人工锁定的窗口
                if any(windows[cid]["window_id"] in locked_windows for cid in combo_ids):
                    continue

                combo_windows = [windows[cid] for cid in combo_ids]

                # 组合时长
                combo_duration = sum(w["duration"] for w in combo_windows)

                # 对各窗口独立评分后取平均
                scores_7d = []
                for cid in combo_ids:
                    s = score_window_pair(
                        narration_text=text,
                        narration_idx=n_idx,
                        audio_duration=audio_dur,
                        window=windows[cid],
                        window_idx=cid,
                        total_windows=total_windows,
                        semantic_similarity=semantic_sims.get((n_idx, cid), 0.0),
                        weights=weights,
                    )
                    scores_7d.append(s)

                avg_score = sum(scores_7d) / len(scores_7d)

                # ━ 多窗口组合奖励 ━
                # 音频远长于单窗口时, 组合窗口即使语义稍低也应该得分更高
                # 奖励 = (组合时长 / 音频时长改善) * 小系数
                if combo_size >= 2 and audio_dur > 0:
                    single_best_dur = max(
                        w["duration"] for w in windows
                        if w["window_id"] not in locked_windows
                    ) if windows else audio_dur
                    # 组合比最佳单窗口的时长改善比例
                    dur_improvement = max(0, (combo_duration - single_best_dur) / audio_dur)
                    combo_bonus = min(0.15, dur_improvement * 0.3)
                    avg_score += combo_bonus

                # 组合窗口合并为一个虚拟窗口
                merged = {
                    "window_id": combo_ids[0],
                    "start_time": min(w["start_time"] for w in combo_windows),
                    "end_time": max(w["end_time"] for w in combo_windows),
                    "duration": combo_duration,
                }

                candidates_for_this_seg.append({
                    "narration_index": n_idx,
                    "combo_window_ids": combo_ids,
                    "merged_window": merged,
                    "score": round(avg_score, 4),
                    "audio_duration": audio_dur,
                })

        # 按得分降序排列
        candidates_for_this_seg.sort(key=lambda x: x["score"], reverse=True)

        if candidates_for_this_seg:
            segment_candidates.append({
                "narration_index": n_idx,
                "candidates": candidates_for_this_seg,
            })

    # ━━━ 步骤 3: 贪心分配 (长时间段优先) ━━━
    assigned_windows = set(locked_windows)
    final_plans: List[Dict] = []

    # 优先处理音频长且受益于多窗口的段落（避免短段抢占长段所需窗口）
    def _priority_key(seg_cand):
        best_score = seg_cand["candidates"][0]["score"]
        best_audio = seg_cand["candidates"][0]["audio_duration"]
        best_combo_size = len(seg_cand["candidates"][0]["combo_window_ids"])
        # 长时间段 + 已有良好多窗口组合 → 高优先级
        return (best_score + 0.1 * (best_audio / 20.0) + 0.05 * best_combo_size)

    for seg_cand in sorted(segment_candidates,
                           key=_priority_key,
                           reverse=True):
        n_idx = seg_cand["narration_index"]

        # 找第一个全部窗口未被占用的候选
        best = None
        for cand in seg_cand["candidates"]:
            combo_ids = cand["combo_window_ids"]
            combo_win_ids = {windows[cid]["window_id"] for cid in combo_ids}
            if assigned_windows.isdisjoint(combo_win_ids):
                best = cand
                break

        if best is None:
            # 所有候选都被占用，取最高分 (即使窗口冲突)
            best = seg_cand["candidates"][0]
            logger.warning(
                f"段 {n_idx}: 所有候选窗口被占用, 使用最佳冲突分配 "
                f"(得分={best['score']:.3f})"
            )

        # 标记窗口已用
        for cid in best["combo_window_ids"]:
            assigned_windows.add(windows[cid]["window_id"])

        # 构建最终计划
        plan = _build_combo_plan(
            n_idx=n_idx,
            audio_dur=best["audio_duration"],
            merged_window=best["merged_window"],
            combo_window_ids=best["combo_window_ids"],
            score=best["score"],
            min_score_threshold=min_score_threshold,
            narration_text=[
                s for s in narration_segments if s["index"] == n_idx
            ][0].get("tts_text", ""),
            windows=windows,
        )
        final_plans.append(plan)

    # ━━━ 步骤 4: 合并锁定段 ━━━
    final_plans.extend(locked_assignments.values())

    # 按 narration_index 排序
    final_plans.sort(key=lambda x: x["narration_index"])

    return final_plans


def _build_combo_plan(
    n_idx: int,
    audio_dur: float,
    merged_window: dict,
    combo_window_ids: List[int],
    score: float,
    min_score_threshold: float,
    narration_text: str = "",
    windows: Optional[List[dict]] = None,
) -> dict:
    """从滑窗候选构建最终计划"""
    video_start = merged_window["start_time"]
    video_end = merged_window["end_time"]
    window_dur = merged_window["duration"]

    retime, speed_factor, freeze_point, needs_review = _determine_retime_strategy(
        audio_dur=audio_dur,
        window_dur=window_dur,
        video_start=video_start,
        video_end=video_end,
        narration_text=narration_text,
        window_ids=combo_window_ids,
        windows=windows,
    )

    needs_review = needs_review or (score < min_score_threshold)

    result = {
        "narration_index": n_idx,
        "video_start": round(video_start, 3),
        "video_end": round(video_end, 3),
        "audio_start": 0.0,
        "audio_end": round(audio_dur, 3),
        "matched_windows": combo_window_ids,
        "score": round(score, 3),
        "retime_strategy": retime,
        "needs_review": needs_review,
    }
    if speed_factor is not None:
        result["speed_factor"] = round(speed_factor, 3)
    if freeze_point is not None:
        result["freeze_point"] = round(freeze_point, 3)

    return result


def _build_combo_plan_from_lock(
    n_idx: int, audio_dur: float, locked_data: List[dict],
    max_score: float, narration_text: str, windows: List[dict],
) -> dict:
    """从人工锁定数据构建计划"""
    merged_start = min(w["start_time"] for w in locked_data)
    merged_end = max(w["end_time"] for w in locked_data)
    combo_ids = [w["window_id"] for w in locked_data]

    merged = {
        "window_id": combo_ids[0],
        "start_time": merged_start,
        "end_time": merged_end,
        "duration": merged_end - merged_start,
    }

    return _build_combo_plan(
        n_idx=n_idx, audio_dur=audio_dur, merged_window=merged,
        combo_window_ids=combo_ids, score=max_score,
        min_score_threshold=0.0, narration_text=narration_text,
        windows=windows,
    )


# ══════════════════════════════════════════════════════════════
# 完整重定时策略 (含定格选点)
# ══════════════════════════════════════════════════════════════

def _find_freeze_point(
    window_ids: List[int],
    windows: List[dict],
    audio_dur: float,
    window_dur: float,
    narration_text: str = "",
) -> Optional[float]:
    """
    定格选点算法:
    1. 优先低运动帧
    2. 优先动作完成点 (action_candidates 中高 confidence 尾帧)
    3. 优先句子停顿点
    4. 禁止默认用最后一帧 (返回 None 则为人工审核)
    """
    if not windows:
        return None

    # 锁定窗口的 VLM 输出
    target_windows = [w for w in windows if w["window_id"] in window_ids]

    # ━ 策略 1: 低运动帧 ━
    # 取最后一个窗口的 VLM action_candidates，找低 confidence 区域
    if target_windows:
        last_win = target_windows[-1]
        vlm = last_win.get("vlm_output", {})
        actions = vlm.get("action_candidates", [])

        if not actions:
            # 无动作检测 → 画面可能静止，选窗口中间点
            return last_win["start_time"] + last_win["duration"] * 0.5

        # ━ 策略 2: 动作完成点 ━
        # 找动作 confidence 最高的点，定格在动作之后
        best_action = max(
            actions, key=lambda a: a.get("confidence", 0) if isinstance(a, dict) else 0
        )
        if isinstance(best_action, dict):
            time_hint = best_action.get("time_hint", 0.5)
            # time_hint 是相对窗口位置 [0..1]
            freeze_at = last_win["start_time"] + last_win["duration"] * time_hint
            # 留 0.1s 余量 (不能在窗口末尾，避免"尾帧定格")
            max_freeze = last_win["end_time"] - 0.1
            return min(freeze_at, max_freeze)

    # ━ 兜底: 窗口 80% 位置 (避免最后一帧) ━
    if target_windows:
        last_win = target_windows[-1]
        return last_win["start_time"] + last_win["duration"] * 0.8

    return None


def _determine_retime_strategy(
    audio_dur: float,
    window_dur: float,
    video_start: float,
    video_end: float,
    duration_tolerance: float = 0.5,
    speed_range: Tuple[float, float] = (0.90, 1.10),
    freeze_max: float = 0.8,
    broll_max: float = 3.0,
    narration_text: str = "",
    window_ids: Optional[List[int]] = None,
    windows: Optional[List[dict]] = None,
) -> Tuple[str, Optional[float], Optional[float], bool]:
    """
    P2 完整重定时策略 (优先级降序)。

    Returns:
        (strategy, speed_factor, freeze_point, needs_review)
    """
    diff = audio_dur - window_dur

    # ═══ 策略 1: 原速播放 (差≤0.5s) ═══
    if abs(diff) <= duration_tolerance:
        return ("normal", 1.0, None, False)

    # ═══ 音频更长 → 视频需要延长 ═══
    if diff > 0:
        # 策略 2: 合并镜头 — 滑窗阶段已处理，此处不重复合并
        # 策略 3: 轻微变速 (慢放 0.90x ~ 1.00x)
        speed_factor = window_dur / audio_dur  # < 1.0 = 慢放
        if speed_range[0] <= speed_factor <= speed_range[1]:
            return ("speed_change", round(speed_factor, 3), None, False)

        # 策略 4: B-roll 补画面 (差值 ≤ 3s)
        #   实现上标记为 broll，Stage 4 用窗口内片段重复填充
        if diff <= broll_max:
            return ("broll", 1.0, None, False)

        # 策略 5: 短定格 (差值 ≤ 0.8s)
        if diff <= freeze_max:
            freeze_pt = _find_freeze_point(
                window_ids=window_ids or [],
                windows=windows or [],
                audio_dur=audio_dur,
                window_dur=window_dur,
                narration_text=narration_text,
            )
            if freeze_pt is None:
                return ("manual_review", 1.0, None, True)
            return ("freeze", 1.0, round(freeze_pt, 3), False)

        # 策略 6: 人工审核
        return ("manual_review", 1.0, None, True)

    # ═══ 视频更长 → 原速播放即截断 ═══
    if diff < 0:
        # 视频比音频长 = 正常，按音频长度截断
        return ("normal", 1.0, None, False)

    return ("normal", 1.0, None, False)


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def load_lock_points(output_dir: Path) -> Dict[int, List[int]]:
    """加载人工锁定点（从 timeline_plan.human.json 或 locks.json）"""
    locks = {}

    # 方式 1: timeline_plan.human.json 中的 locked_windows 字段
    human_plan = output_dir / "timeline_plan.human.json"
    if human_plan.exists():
        with open(human_plan, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
        for seg in plan_data.get("segments", []):
            if seg.get("locked"):
                n_idx = seg["narration_index"]
                locks[n_idx] = seg.get("matched_windows", seg.get("locked_window_ids", []))
                logger.info(f"  人工锁定: 段{n_idx} → 窗口 {locks[n_idx]}")

    # 方式 2: locks.json (简单格式)
    locks_file = output_dir / "locks.json"
    if locks_file.exists():
        with open(locks_file, "r", encoding="utf-8") as f:
            extra_locks = json.load(f)
        for key, val in extra_locks.items():
            locks[int(key)] = val
        logger.info(f"  locks.json 加载: {len(extra_locks)} 个锁定点")

    return locks


def run_stage3(
    manifest_path: str = "output/default/project_manifest.json",
    video_meta_path: str = "output/default/video_meta.ai.json",
    audio_manifest_path: str = "output/default/narration_audio_manifest.json",
    output_dir: str = "output/default",
    config: Optional[Dict] = None,
) -> str:
    """
    Stage 3 主入口 — P2 完整版。

    优先读取 *.human.json，不存在则用 *.ai.json
    支持人工锁定点（最高优先级覆盖自动匹配）。

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
    windows = video_meta["windows"]

    # 合并音频时长到 narration_segments
    audio_map = {s["index"]: s for s in audio_manifest["segments"]}
    narration_segments = manifest["narration_segments"]
    for seg in narration_segments:
        audio_info = audio_map.get(seg["index"], {})
        seg["duration"] = audio_info.get("duration", 0)

    # 2. 加载配置
    stage3_cfg = config.get("stage3", {}) if config else {}
    weights = dict(DEFAULT_WEIGHTS)
    weights.update(stage3_cfg.get("scoring_weights", {}))
    max_windows = stage3_cfg.get("max_windows_per_para", 3)
    min_score = stage3_cfg.get("min_score_threshold", 0.4)

    logger.info("=" * 60)
    logger.info("Stage 3: 时间轴对齐 (P2 7维评分 + 滑窗)")
    logger.info("=" * 60)
    logger.info(f"  视频窗口: {len(windows)}, 旁白段: {len(narration_segments)}")
    logger.info(f"  视频时长: {video_meta['stats']['video_duration']}s")
    logger.info(f"  音频总时长: {audio_manifest['stats']['total_duration']}s")

    # 3. 加载人工锁定点
    lock_points = load_lock_points(output_dir)
    if lock_points:
        logger.info(f"  人工锁定: {len(lock_points)} 个段")

    # 4. P2 滑窗匹配
    plans = match_segments_with_sliding_windows(
        narration_segments=narration_segments,
        windows=windows,
        weights=weights,
        max_windows_per_para=max_windows,
        min_score_threshold=min_score,
        lock_points=lock_points,
    )

    # 5. 计算全局时间轴
    current_audio_time = 0.0
    for plan in plans:
        audio_dur = plan["audio_end"] - plan["audio_start"]
        plan["audio_start"] = round(current_audio_time, 3)
        plan["audio_end"] = round(current_audio_time + audio_dur, 3)
        current_audio_time += audio_dur

    # 6. 统计
    strategies = {}
    needs_review_count = 0
    for plan in plans:
        s = plan["retime_strategy"]
        strategies[s] = strategies.get(s, 0) + 1
        if plan["needs_review"]:
            needs_review_count += 1

    # 7. 组装输出
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
            "scoring_weights": weights,
            "strategies": strategies,
            "needs_review_count": needs_review_count,
            "lock_points_used": len(lock_points),
        },
        "total_duration": round(
            audio_manifest["stats"]["total_duration"], 3
        ),
        "segments": plans,
    }

    # 8. Schema 校验 & 输出
    is_valid, error = validate_json(timeline, "timeline_plan")
    if not is_valid:
        logger.warning(f"⚠ Schema 校验未通过: {error}")

    output_path = output_dir / "timeline_plan.ai.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ timeline_plan.ai.json 已输出: {output_path}")
    logger.info(f"  策略分布: {strategies}")
    logger.info(f"  需人工审核: {needs_review_count}/{len(plans)}")

    # 打印对齐预览
    print_alignment_preview(plans)

    return str(output_path)


def print_alignment_preview(plans: List[Dict]):
    """打印对齐预览表格"""
    header = (
        f"{'段':>3} {'视频窗口':>8} {'视频时间':>14} "
        f"{'音频':>6} {'视频':>6} {'策略':>14} {'得分':>5} {'审核':>4}"
    )
    sep = "-" * (len(header) + 10)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for plan in plans:
        n_idx = plan["narration_index"]
        audio_dur = plan["audio_end"] - plan["audio_start"]
        v_start = plan["video_start"]
        v_end = plan["video_end"]
        window_dur = v_end - v_start
        strategy = plan["retime_strategy"]
        score = plan["score"]
        review = "⚠" if plan["needs_review"] else "✓"
        locked = "🔒" if plan.get("locked") else ""
        win_ids = ",".join(str(w) for w in plan.get("matched_windows", []))

        logger.info(
            f"{n_idx:>3} {win_ids:>8} {v_start:>5.1f}s-{v_end:>5.1f}s "
            f"{audio_dur:>4.1f}s {window_dur:>4.1f}s "
            f"{strategy:>14} {score:>.3f} {review:>4} {locked}"
        )
    logger.info(sep)


# ══════════════════════════════════════════════════════════════
# CLI 测试入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

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
