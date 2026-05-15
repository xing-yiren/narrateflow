from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from timeline_align.keyframe_filter import sample_keyframes
from timeline_align.vl_client import (
    call_vl_gemini,
    load_gemini_api_key,
    parse_gemini_batch_response,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_reference_text(reference_text_path: Path | None) -> str:
    if reference_text_path is None or not reference_text_path.exists():
        return ""
    return reference_text_path.read_text(encoding="utf-8").strip()


def build_video_windows(
    keyframes_payload: dict[str, Any],
    min_window_sec: float = 2.5,
    max_window_sec: float = 6.0,
) -> list[dict[str, Any]]:
    duration = float(keyframes_payload.get("duration", 0.0) or 0.0)
    candidates = sorted(
        keyframes_payload.get("candidates", []), key=lambda item: float(item["time"])
    )
    if not candidates:
        return [
            {
                "window_id": "video_1",
                "start_time": 0.0,
                "end_time": round(duration, 3),
                "frames": [],
            }
        ]

    boundaries = [0.0]
    for item in candidates:
        time_value = round(float(item["time"]), 3)
        if time_value > boundaries[-1]:
            boundaries.append(time_value)
    if boundaries[-1] < duration:
        boundaries.append(round(duration, 3))

    raw_windows: list[dict[str, Any]] = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        is_last = idx == len(boundaries) - 2
        frames = [
            item
            for item in candidates
            if start <= float(item["time"]) < end
            or (is_last and float(item["time"]) == end)
        ]
        raw_windows.append(
            {
                "window_id": f"video_{idx + 1}",
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "frames": frames,
                "max_global_score": max(
                    (float(frame.get("global_score", 0.0)) for frame in frames),
                    default=0.0,
                ),
            }
        )

    merged: list[dict[str, Any]] = []
    for window in raw_windows:
        duration_sec = float(window["end_time"]) - float(window["start_time"])
        merged_duration = (
            float(window["end_time"]) - float(merged[-1]["start_time"])
            if merged
            else duration_sec
        )
        if (
            merged
            and duration_sec < min_window_sec
            and float(window.get("max_global_score", 0.0)) < 12.0
            and merged_duration <= max_window_sec
        ):
            merged[-1]["end_time"] = window["end_time"]
            merged[-1]["frames"].extend(window.get("frames", []))
            merged[-1]["max_global_score"] = max(
                float(merged[-1].get("max_global_score", 0.0)),
                float(window.get("max_global_score", 0.0)),
            )
            continue
        merged.append(window)

    for index, window in enumerate(merged, start=1):
        seen: set[tuple[float, str]] = set()
        frames = []
        for frame in window.get("frames", []):
            key = (round(float(frame["time"]), 3), str(frame["image_path"]))
            if key in seen:
                continue
            seen.add(key)
            frames.append(frame)
        window["window_id"] = f"video_{index}"
        window["frames"] = frames
        window["duration"] = round(float(window["end_time"]) - float(window["start_time"]), 3)
    return merged


def build_understanding_prompt(windows: list[dict[str, Any]], reference_text: str) -> str:
    lines = [
        "请根据视频关键帧理解每个时间窗口的画面内容。",
        "只返回 JSON 数组，不要输出解释文字。",
        "每个数组元素必须包含 window_id、visual_summary、visible_text、actions、objects、is_silent、reason。",
        "visual_summary 用简洁中文描述该窗口画面变化和重点信息。",
        "visible_text 返回画面中能读到的重要文字列表；没有则返回空数组。",
        "actions 返回用户操作或画面动作列表；objects 返回关键对象列表。",
        "is_silent 表示该窗口是否没有明显可讲解内容。",
    ]
    if reference_text:
        lines.append(f"参考文档：{reference_text}")
    lines.append("窗口列表：")
    for window in windows:
        lines.append(
            f"- {window['window_id']}: {float(window['start_time']):.2f}s - {float(window['end_time']):.2f}s, "
            f"frames={len(window.get('frames', []))}"
        )
    lines.append(
        json.dumps(
            [
                {
                    "window_id": window["window_id"],
                    "visual_summary": "该窗口画面内容摘要",
                    "visible_text": [],
                    "actions": [],
                    "objects": [],
                    "is_silent": False,
                    "reason": "判断依据",
                }
                for window in windows
            ],
            ensure_ascii=False,
        )
    )
    return "\n".join(lines)


def normalize_understanding_results(
    windows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result_map = {str(item.get("window_id")): item for item in results}
    normalized = []
    for window in windows:
        item = result_map.get(window["window_id"], {})
        normalized.append(
            {
                "window_id": window["window_id"],
                "start_time": round(float(window["start_time"]), 3),
                "end_time": round(float(window["end_time"]), 3),
                "duration": round(float(window["end_time"]) - float(window["start_time"]), 3),
                "frames": [
                    {
                        "time": round(float(frame["time"]), 3),
                        "image_path": str(frame["image_path"]),
                        "type": frame.get("type"),
                        "reason": frame.get("reason", []),
                        "global_score": frame.get("global_score"),
                        "text_like_score": frame.get("text_like_score"),
                    }
                    for frame in window.get("frames", [])
                ],
                "visual_summary": str(item.get("visual_summary", "")).strip(),
                "visible_text": item.get("visible_text", []) if isinstance(item.get("visible_text", []), list) else [],
                "actions": item.get("actions", []) if isinstance(item.get("actions", []), list) else [],
                "objects": item.get("objects", []) if isinstance(item.get("objects", []), list) else [],
                "is_silent": bool(item.get("is_silent", False)),
                "reason": str(item.get("reason", "")).strip(),
                "needs_review": not bool(str(item.get("visual_summary", "")).strip()),
            }
        )
    return normalized


def load_existing_understood_windows(output_dir: Path) -> dict[str, dict[str, Any]]:
    understanding_path = output_dir / "video_understanding.json"
    if not understanding_path.exists():
        return {}
    try:
        payload = load_json(understanding_path)
    except Exception:
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for window in payload.get("windows", []):
        if window.get("window_id") and not window.get("needs_review") and window.get("visual_summary"):
            existing[str(window["window_id"])] = window
    return existing


def has_quota_error(message: str) -> bool:
    return "RESOURCE_EXHAUSTED" in message or "quota" in message.lower() or "429" in message


def merge_understood_windows(
    windows: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    fresh: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = []
    for window in windows:
        window_id = str(window["window_id"])
        merged.append(fresh.get(window_id) or existing.get(window_id) or normalize_understanding_results([window], [])[0])
    return merged


def run_video_understanding(
    video: Path,
    output_dir: Path,
    gemini_api_key: str | None = None,
    reference_text_path: Path | None = None,
    frame_stride: int | None = None,
    min_gap_sec: float = 1.5,
    global_threshold: float = 8.0,
    subtitle_threshold: float = 5.5,
    detection_max_width: int = 960,
    fill_gap_sec: float = 6.0,
    batch_size: int = 3,
    model: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframes_payload = sample_keyframes(
        video_path=video,
        out_dir=output_dir / "keyframes",
        frame_stride=frame_stride,
        min_gap_sec=min_gap_sec,
        global_threshold=global_threshold,
        subtitle_threshold=subtitle_threshold,
        detection_max_width=detection_max_width,
        fill_gap_sec=fill_gap_sec,
    )
    keyframes_path = output_dir / "keyframes.json"
    write_json(keyframes_path, keyframes_payload)

    windows = build_video_windows(keyframes_payload)
    window_manifest_path = output_dir / "window_manifest.json"
    write_json(window_manifest_path, {"windows": windows})

    reference_text = read_reference_text(reference_text_path)
    api_key = load_gemini_api_key(gemini_api_key)
    existing_windows = load_existing_understood_windows(output_dir)
    fresh_windows: dict[str, dict[str, Any]] = {}
    requests_log: list[dict[str, Any]] = []
    responses_log: list[dict[str, Any]] = []
    quota_stopped = False

    pending_windows = [window for window in windows if str(window["window_id"]) not in existing_windows]
    for start in range(0, len(pending_windows), batch_size):
        batch = pending_windows[start : start + batch_size]
        prompt = build_understanding_prompt(batch, reference_text)
        requests_log.append({"window_ids": [item["window_id"] for item in batch], "prompt": prompt})
        try:
            response = call_vl_gemini(api_key=api_key, windows=batch, prompt=prompt, model=model)
            response_text = str(response.get("content", "")).strip()
            parsed = parse_gemini_batch_response(response_text)
        except Exception as exc:
            response_text = f"error: {exc}"
            parsed = []
            if has_quota_error(response_text):
                quota_stopped = True
        responses_log.append(
            {
                "window_ids": [item["window_id"] for item in batch],
                "raw_content": response_text,
                "parsed_count": len(parsed),
            }
        )
        for item in normalize_understanding_results(batch, parsed):
            if item.get("visual_summary") and not item.get("needs_review"):
                fresh_windows[str(item["window_id"])] = item
        if quota_stopped:
            break

    understood_windows = merge_understood_windows(windows, existing_windows, fresh_windows)
    failed_batches = [item for item in responses_log if str(item.get("raw_content", "")).startswith("error:")]
    if failed_batches and len(failed_batches) == len(responses_log) and not existing_windows:
        raise RuntimeError(f"All Gemini video-understanding batches failed. First error: {failed_batches[0]['raw_content']}")

    payload = {
        "source_type": "video_understanding",
        "video_path": str(video.resolve()),
        "reference_text_path": str(reference_text_path.resolve()) if reference_text_path else None,
        "model": model,
        "keyframes_path": str(keyframes_path),
        "window_manifest_path": str(window_manifest_path),
        "window_count": len(understood_windows),
        "failed_batch_count": len(failed_batches),
        "cached_window_count": len(existing_windows),
        "fresh_window_count": len(fresh_windows),
        "quota_stopped": quota_stopped,
        "windows": understood_windows,
    }
    understanding_path = output_dir / "video_understanding.json"
    write_json(understanding_path, payload)
    write_json(output_dir / "gemini_understanding_requests.json", {"requests": requests_log})
    write_json(output_dir / "gemini_understanding_responses.json", {"responses": responses_log})
    return {
        "understanding_path": understanding_path,
        "keyframes_path": keyframes_path,
        "window_manifest_path": window_manifest_path,
        "understanding": payload,
    }
