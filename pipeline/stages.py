from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline.shared import project_output_root
from timeline_align.video_understanding import run_video_understanding
from timeline_align.video_script import generate_spoken_from_understanding


def default_understanding_dir(config: dict[str, Any]) -> Path:
    if config.get("understanding_output_dir"):
        return Path(config["understanding_output_dir"])
    return project_output_root(config.get("project_dir"), config.get("video")) / "understanding"


def run_stage_understand(config: dict[str, Any]) -> dict[str, Any]:
    return run_video_understanding(
        video=Path(config["video"]),
        output_dir=default_understanding_dir(config),
        gemini_api_key=config.get("gemini_api_key"),
        reference_text_path=None,
        frame_stride=config.get("frame_stride"),
        min_gap_sec=float(config.get("min_gap_sec") or 1.5),
        global_threshold=float(config.get("global_threshold") or 8.0),
        subtitle_threshold=float(config.get("subtitle_threshold") or 5.5),
        detection_max_width=int(config.get("detection_max_width") or 960),
        fill_gap_sec=float(config.get("fill_gap_sec") or 6.0),
        batch_size=int(config.get("understand_batch_size") or 3),
        model=str(config.get("gemini_model") or "gemini-2.5-flash"),
        use_gemini=bool(config.get("use_gemini_script", True)),
    )


def add_understand_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--understanding-output-dir")
    parser.add_argument("--reference-document")
    parser.add_argument("--gemini-api-key")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--min-gap-sec", type=float)
    parser.add_argument("--global-threshold", type=float)
    parser.add_argument("--subtitle-threshold", type=float)
    parser.add_argument("--detection-max-width", type=int)
    parser.add_argument("--fill-gap-sec", type=float)
    parser.add_argument("--understand-batch-size", type=int, default=3)
    parser.add_argument("--no-gemini-script", action="store_true", help="Draft spoken.json directly from video_understanding.json without another Gemini call.")


def default_script_dir(config: dict[str, Any]) -> Path:
    if config.get("stage1_output_dir"):
        return Path(config["stage1_output_dir"])
    return project_output_root(config.get("project_dir"), config.get("video")) / "script"


def run_stage_video_script(config: dict[str, Any]) -> dict[str, Any]:
    understanding_path = config.get("video_understanding") or config.get("video_understanding_path")
    if not understanding_path:
        project_dir = config.get("project_dir")
        if project_dir:
            candidate = project_output_root(project_dir, config.get("video")) / "understanding" / "video_understanding.json"
            if candidate.exists():
                understanding_path = str(candidate)
    if not understanding_path:
        raise FileNotFoundError("video_understanding.json is required before video-auto script generation")
    return generate_spoken_from_understanding(
        understanding_path=Path(understanding_path),
        output_dir=default_script_dir(config),
        gemini_api_key=config.get("gemini_api_key"),
        reference_text_path=None,
        batch_size=int(config.get("understand_batch_size") or 3),
        model=str(config.get("gemini_model") or "gemini-2.5-flash"),
        use_gemini=bool(config.get("use_gemini_script", True)),
    )
