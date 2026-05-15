from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from timeline_align.vl_client import (
    call_vl_gemini,
    load_gemini_api_key,
    parse_gemini_batch_response,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_tts_duration(text: str, chars_per_sec: float = 4.0) -> float:
    clean = str(text or "").strip()
    if not clean:
        return 0.0
    return round(len(clean) / chars_per_sec, 3)


def build_global_summary(understanding: dict[str, Any], reference_text: str) -> str:
    if reference_text:
        snippet = " ".join(reference_text.split()).strip()
        return f"参考文稿摘要：{snippet}"
    summaries = [str(item.get("visual_summary", "")).strip() for item in understanding.get("windows", [])]
    summaries = [item for item in summaries if item]
    if summaries:
        return "视频展示流程：" + "；".join(summaries[:8])
    return "这是一个产品演示视频，请生成连贯、自然、适合配音的中文讲解。"


def build_previous_context(previous_item: dict[str, Any] | None) -> tuple[str, str]:
    if previous_item is None:
        return "[START_OF_VIDEO]", "start"
    if previous_item.get("is_silent"):
        duration = float(previous_item.get("duration", 0.0))
        return f"[SILENT_GAP: {duration:.1f}s]", "silent_gap"
    return str(previous_item.get("spoken_text", "")).strip(), "spoken_text"


def sanitize_transition_prefix(text: str, previous_context_type: str) -> str:
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return clean
    if previous_context_type not in {"start", "silent_gap"}:
        return clean
    prefixes = ["首先", "第一步", "接下来", "然后", "接着", "现在", "这里", "画面显示", "画面展示"]
    changed = True
    while changed and clean:
        changed = False
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].lstrip(" ，,。；;：:.-")
                changed = True
                break
    return clean


def calculate_window_max_chars(window: dict[str, Any], chars_per_sec: float = 4.0) -> int:
    duration = max(1.5, float(window.get("end_time", 0.0)) - float(window.get("start_time", 0.0)))
    return max(18, min(90, int(round(duration * chars_per_sec + 12))))


def build_script_prompt(
    windows: list[dict[str, Any]],
    global_summary: str,
    previous_context: str,
    reference_text: str = "",
) -> str:
    lines = [
        "你正在为一个产品演示视频撰写连续口播讲解，而不是逐帧画面描述。",
        "只返回 JSON 数组，不要输出解释文字。",
        "每个数组元素必须包含 window_id、spoken_text、is_silent、reason 和 reference_terms_used。",
        "请基于视频理解结果、参考文稿和上一段旁白，生成自然连贯的讲解稿。",
        "不要频繁使用画面显示、画面展示、我们可以看到这类描述；要像讲解员一样说明操作目的、流程和结果。",
        "visible_text 只是辅助理解，不要逐项复述。",
        "如果当前窗口只是等待、重复或没有新的讲解价值，可以 is_silent=true。",
        "尽量让 spoken_text 不超过对应字数预算。",
        f"全局主题：{global_summary}",
        f"上一段旁白：{previous_context}",
    ]
    if previous_context == "[START_OF_VIDEO]":
        lines.append("这是第一段口播，请直接进入主题，不要说首先或画面显示。")
    elif previous_context.startswith("[SILENT_GAP:"):
        lines.append("前面有静默，请自然恢复讲解，不要提到静默。")
    if reference_text:
        lines.append(f"参考文稿：{reference_text}")
    lines.append("当前窗口素材：")
    for window in windows:
        max_chars = int(window.get("max_chars") or calculate_window_max_chars(window))
        lines.append(
            json.dumps(
                {
                    "window_id": window.get("window_id"),
                    "time_range": [window.get("start_time"), window.get("end_time")],
                    "max_chars": max_chars,
                    "visual_summary": window.get("visual_summary"),
                    "visible_text": window.get("visible_text", [])[:12],
                    "actions": window.get("actions", []),
                    "objects": window.get("objects", []),
                    "is_silent_hint": window.get("is_silent", False),
                },
                ensure_ascii=False,
            )
        )
    lines.append(
        json.dumps(
            [
                {
                    "window_id": window.get("window_id"),
                    "spoken_text": "自然连贯的一小段讲解旁白",
                    "is_silent": False,
                    "reason": "生成依据",
                    "reference_terms_used": [],
                }
                for window in windows
            ],
            ensure_ascii=False,
        )
    )
    return "\n".join(lines)


def normalize_script_drafts(
    windows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result_map = {str(item.get("window_id")): item for item in results}
    drafts = []
    for window in windows:
        item = result_map.get(str(window.get("window_id")), {})
        is_silent = bool(item.get("is_silent", window.get("is_silent", False)))
        previous_context_type = str(window.get("previous_context_type", ""))
        spoken_text = "" if is_silent else sanitize_transition_prefix(
            str(item.get("spoken_text", "")).strip(), previous_context_type
        )
        max_chars = int(window.get("max_chars") or calculate_window_max_chars(window))
        drafts.append(
            {
                "window_id": window["window_id"],
                "start_time": round(float(window["start_time"]), 3),
                "end_time": round(float(window["end_time"]), 3),
                "source_frames": window.get("frames", []),
                "visual_summary": window.get("visual_summary", ""),
                "visible_text": window.get("visible_text", []),
                "spoken_text": spoken_text,
                "is_silent": is_silent,
                "reason": str(item.get("reason", "")).strip(),
                "reference_terms_used": item.get("reference_terms_used", []),
                "duration": round(float(window.get("end_time", 0.0)) - float(window.get("start_time", 0.0)), 3),
                "max_chars": max_chars,
                "previous_context_used": window.get("previous_context_used", ""),
                "previous_context_type": previous_context_type,
                "estimated_tts_duration": estimate_tts_duration(spoken_text),
                "needs_review": (not bool(spoken_text) and not is_silent) or len(spoken_text) > max_chars + 8,
            }
        )
    return drafts


def build_spoken_payload(understanding: dict[str, Any], drafts: list[dict[str, Any]]) -> dict[str, Any]:
    paragraphs = []
    segments = []
    for index, draft in enumerate(drafts, start=1):
        paragraph = {
            "index": index,
            "is_title": False,
            "is_silent": bool(draft.get("is_silent", False)),
            "source_text": draft.get("visual_summary", ""),
            "spoken_text": draft.get("spoken_text", ""),
            "segment_count": 0 if draft.get("is_silent") else 1,
            "start_time": draft["start_time"],
            "end_time": draft["end_time"],
            "source_window_id": draft["window_id"],
            "source_frames": draft.get("source_frames", []),
            "visual_summary": draft.get("visual_summary", ""),
            "visible_text": draft.get("visible_text", []),
            "duration_budget": round(float(draft.get("duration", 0.0)) + 0.8, 3),
            "max_chars": draft.get("max_chars"),
            "previous_context_used": draft.get("previous_context_used", ""),
            "previous_context_type": draft.get("previous_context_type", ""),
            "estimated_tts_duration": draft.get("estimated_tts_duration", 0.0),
            "needs_review": bool(draft.get("needs_review", False)),
            "reference_terms_used": draft.get("reference_terms_used", []),
        }
        paragraphs.append(paragraph)
        if paragraph["spoken_text"] and not paragraph["is_silent"]:
            segments.append(
                {
                    "segment_id": f"p01_s{index:03d}",
                    "paragraph_index": index,
                    "source_text": paragraph["spoken_text"],
                    "spoken_text": paragraph["spoken_text"],
                    "start_time": paragraph["start_time"],
                    "end_time": paragraph["end_time"],
                    "is_silent": False,
                }
            )
    return {
        "source_path": understanding.get("video_path"),
        "source_type": "video_auto_script",
        "video_path": understanding.get("video_path"),
        "video_understanding_path": understanding.get("understanding_path"),
        "page": 1,
        "title_text": "视频讲解",
        "global_summary": understanding.get("global_summary"),
        "paragraphs": paragraphs,
        "segments": segments,
        "combined_spoken_text": " ".join(item["spoken_text"] for item in paragraphs if item["spoken_text"]),
    }


def split_reference_script(reference_text: str) -> list[str]:
    clean = "\n".join(line.strip() for line in reference_text.splitlines())
    chunks = [item.strip() for item in re.split(r"\n\s*\n+", clean) if item.strip()]
    if len(chunks) <= 1:
        chunks = [item.strip() for item in re.split(r"(?<=[。！？；])", clean) if item.strip()]
    return chunks


def clean_reference_chunk(text: str) -> str:
    clean = " ".join(str(text or "").split()).strip()
    return re.sub(r"^\d+[.、．]\s*", "", clean).strip()


def fit_text_to_chars(text: str, max_chars: int) -> str:
    clean = clean_reference_chunk(text)
    if len(clean) <= max_chars:
        return clean
    clauses = [item.strip() for item in re.split(r"([，。；;！？])", clean) if item.strip()]
    result = ""
    for idx in range(0, len(clauses), 2):
        clause = clauses[idx]
        punct = clauses[idx + 1] if idx + 1 < len(clauses) else ""
        candidate = result + clause + punct
        if len(candidate) > max_chars and result:
            break
        result = candidate
    if result and len(result) >= min(12, max_chars):
        return result
    return clean


def build_draft_results_from_reference(
    windows: list[dict[str, Any]], reference_chunks: list[str], start_index: int
) -> tuple[list[dict[str, Any]], int]:
    results = []
    cursor = start_index
    for window in windows:
        summary = str(window.get("visual_summary", "")).strip()
        max_chars = int(window.get("max_chars") or calculate_window_max_chars(window))
        if not summary or cursor >= len(reference_chunks):
            results.append({"window_id": window.get("window_id"), "spoken_text": "", "is_silent": True, "reason": "no_reference_chunk", "reference_terms_used": []})
            continue
        spoken_text = fit_text_to_chars(reference_chunks[cursor], max(max_chars, 80))
        cursor += 1
        results.append({"window_id": window.get("window_id"), "spoken_text": spoken_text, "is_silent": False, "reason": "fallback_from_reference_text", "reference_terms_used": []})
    return results, cursor


def build_draft_results_from_understanding(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for window in windows:
        summary = str(window.get("visual_summary", "")).strip()
        is_silent = bool(window.get("is_silent", False)) or not summary
        spoken_text = "" if is_silent else summary
        for prefix in ["画面显示", "画面展示", "画面切换到", "画面中"]:
            if spoken_text.startswith(prefix):
                spoken_text = spoken_text[len(prefix):].lstrip(" ，,。；;：:.-")
                break
        results.append(
            {
                "window_id": window.get("window_id"),
                "spoken_text": spoken_text,
                "is_silent": is_silent,
                "reason": "fallback_from_video_understanding",
                "reference_terms_used": [],
            }
        )
    return results


def prepare_batch_with_context(
    batch: list[dict[str, Any]], previous_item: dict[str, Any] | None
) -> list[dict[str, Any]]:
    prepared_batch = []
    batch_previous_item = previous_item
    for window in batch:
        previous_context, previous_context_type = build_previous_context(batch_previous_item)
        prepared = dict(window)
        prepared["max_chars"] = calculate_window_max_chars(prepared)
        prepared["previous_context_used"] = previous_context
        prepared["previous_context_type"] = previous_context_type
        prepared_batch.append(prepared)
        batch_previous_item = {"spoken_text": str(window.get("visual_summary", "")).strip()}
    return prepared_batch


def generate_spoken_from_understanding(
    understanding_path: Path,
    output_dir: Path,
    gemini_api_key: str | None = None,
    reference_text_path: Path | None = None,
    batch_size: int = 3,
    model: str = "gemini-2.5-flash",
    use_gemini: bool = True,
) -> dict[str, Any]:
    understanding = load_json(understanding_path)
    understanding["understanding_path"] = str(understanding_path)
    reference_text = ""
    if reference_text_path and reference_text_path.exists():
        reference_text = reference_text_path.read_text(encoding="utf-8").strip()
    windows = understanding.get("windows", [])
    global_summary = build_global_summary(understanding, reference_text)
    request_logs = []
    response_logs = []
    drafts = []
    previous_item: dict[str, Any] | None = None
    if use_gemini:
        api_key = load_gemini_api_key(gemini_api_key)
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            previous_context, _ = build_previous_context(previous_item)
            prepared_batch = prepare_batch_with_context(batch, previous_item)
            prompt = build_script_prompt(
                prepared_batch,
                global_summary=global_summary,
                previous_context=previous_context,
                reference_text=reference_text,
            )
            request_logs.append(
                {
                    "window_ids": [item["window_id"] for item in prepared_batch],
                    "previous_context": previous_context,
                    "prompt": prompt,
                }
            )
            try:
                response = call_vl_gemini(api_key=api_key, windows=prepared_batch, prompt=prompt, model=model)
                response_text = str(response.get("content", "")).strip()
                parsed = parse_gemini_batch_response(response_text)
            except Exception as exc:
                response_text = f"error: {exc}"
                parsed = build_draft_results_from_understanding(prepared_batch)
            response_logs.append(
                {
                    "window_ids": [item["window_id"] for item in prepared_batch],
                    "raw_content": response_text,
                    "parsed_count": len(parsed),
                }
            )
            batch_drafts = normalize_script_drafts(prepared_batch, parsed)
            drafts.extend(batch_drafts)
            for draft in batch_drafts:
                previous_item = draft
    else:
        reference_chunks = split_reference_script(reference_text)
        reference_cursor = 0
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            prepared_batch = prepare_batch_with_context(batch, previous_item)
            if reference_chunks:
                parsed, reference_cursor = build_draft_results_from_reference(prepared_batch, reference_chunks, reference_cursor)
                fallback_reason = "fallback_from_reference_text"
            else:
                parsed = build_draft_results_from_understanding(prepared_batch)
                fallback_reason = "fallback_from_video_understanding"
            request_logs.append({"window_ids": [item["window_id"] for item in prepared_batch], "prompt": fallback_reason})
            response_logs.append({"window_ids": [item["window_id"] for item in prepared_batch], "raw_content": fallback_reason, "parsed_count": len(parsed)})
            batch_drafts = normalize_script_drafts(prepared_batch, parsed)
            drafts.extend(batch_drafts)
            for draft in batch_drafts:
                previous_item = draft
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "video_script_drafts.json", {"drafts": drafts})
    write_json(output_dir / "gemini_script_requests.json", {"requests": request_logs})
    write_json(output_dir / "gemini_script_responses.json", {"responses": response_logs})
    understanding["global_summary"] = global_summary
    spoken_payload = build_spoken_payload(understanding, drafts)
    spoken_path = output_dir / "page_01.spoken.json"
    write_json(spoken_path, spoken_payload)
    return {"spoken_path": spoken_path, "spoken": spoken_payload}
