from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from text_process.run_text_process import prepare_ppt_page, slugify
from voice_process.run_voice_profile import run_voice_profile
from voice_process.run_voice_generate import run_voice_generate
from timeline_align.run_timeline_align import run_timeline_align
from video_compose.run_video_compose import run_video_compose


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


def prompt_text(label: str, default: str | None = None, required: bool = True) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        value = value.strip('"').strip("'")
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""


def prompt_choice(label: str, choices: list[str], default: str | None = None) -> str:
    display = "/".join(choices)
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label} ({display}){suffix}: ").strip().lower()
        if not value and default is not None:
            return default
        if value in choices:
            return value
        print(f"Please choose one of: {', '.join(choices)}")


def parse_title_indices(raw: str) -> set[int]:
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def resolve_profile_path(raw: str) -> Path:
    path = Path(str(raw).strip().strip('"').strip("'"))
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / f"{path.name}.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Invalid voice profile path. Please provide a .pt file or a directory containing <dirname>.pt"
    )


def read_env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip()
    return None


def resolve_initial_args(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}

    config["ppt"] = args.ppt or prompt_text("PPT path")
    config["page"] = args.page or int(prompt_text("Page number"))
    config["video"] = args.video or prompt_text("Target video path")

    title_mode = args.title_mode or prompt_choice(
        "Title mode\n- first: treat the first paragraph as title\n- none: treat all paragraphs as narration\n- manual: choose title paragraph indices manually\nChoice",
        ["first", "none", "manual"],
        default="first",
    )
    config["title_mode"] = title_mode
    if title_mode == "manual":
        raw_indices = args.title_indices or prompt_text(
            "Title paragraph indices (comma separated)", default="1"
        )
    elif title_mode == "first":
        raw_indices = "1"
    else:
        raw_indices = ""
    config["title_indices"] = parse_title_indices(raw_indices)

    profile = args.profile
    voice_name = args.voice_name
    ref_audio = args.ref_audio
    ref_text = args.ref_text

    if not profile:
        has_profile = prompt_choice(
            "Do you already have a voice profile file", ["y", "n"], default="y"
        )
        if has_profile == "y":
            profile = prompt_text("Voice profile path (.pt file or profile directory)")
        else:
            voice_name = voice_name or prompt_text("Voice name")
            ref_audio = ref_audio or prompt_text(
                "Reference audio path (recommended 10-20 seconds)"
            )
            ref_text = ref_text or prompt_text("Reference text")
    config["profile"] = str(resolve_profile_path(profile)) if profile else None
    config["voice_name"] = voice_name
    config["ref_audio"] = ref_audio
    config["ref_text"] = ref_text

    config["stage1_output_dir"] = args.stage1_output_dir
    config["profile_output_dir"] = args.profile_output_dir
    config["voice_output_dir"] = args.voice_output_dir
    config["timeline_output"] = args.timeline_output
    config["timeline_debug_dir"] = args.timeline_debug_dir
    config["compose_output_dir"] = args.compose_output_dir
    config["probe_mode"] = args.probe_mode
    config["probe_times"] = args.probe_times or prompt_text(
        "Initial probe times (comma separated, keyframe times)", default="0,10,20,30"
    )
    config["api_key"] = args.api_key or read_env_key("MAAS_API_KEY")
    if not config["api_key"]:
        config["api_key"] = prompt_text("MAAS_API_KEY", required=True)
    return config


def stage_banner(index: int, total: int, name: str) -> None:
    print()
    print(f"[{index}/{total}] {name}")


def ask_continue_or_stop(stage_name: str) -> str:
    return prompt_choice(
        f"{stage_name} review action\n- c: continue to the next stage\n- s: stop here\nChoice",
        ["c", "s"],
        default="c",
    )


def show_stage1_summary(result: dict[str, Any]) -> None:
    print("Stage 1 completed.")
    print(f"extracted_json: {result['extracted_path']}")
    print(f"spoken_json:    {result['spoken_path']}")
    print("Review suggestions:")
    print("- Check page_XX.spoken.json")
    print("- Edit paragraphs[].spoken_text if wording needs adjustment")
    print(
        "- If title handling is wrong, rerun Stage 1 with title mode or title indices"
    )


def show_profile_summary(profile_path: Path) -> None:
    print("Voice profile created.")
    print(f"profile_path: {profile_path}")
    print("Proceeding to voice generation for actual audio review...")


def show_stage2_summary(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments_dir = manifest_path.parent / "segments"
    print("Stage 2 completed.")
    print(f"manifest: {manifest_path}")
    print(f"segments_dir: {segments_dir}")
    print("Available paragraphs:")
    print(
        ", ".join(str(item["paragraph_index"]) for item in payload.get("segments", []))
    )
    print("Review suggestions:")
    print("- If wording is wrong, edit page_XX.spoken.json -> paragraphs[].spoken_text")
    print("- If wording is correct but audio sounds bad, regenerate by paragraph index")
    print("- Paragraph audio files are stored under segments_dir")
    return payload


def ask_stage2_action() -> str:
    return prompt_choice(
        "Stage 2 review action\n- c: continue to the next stage\n- r: regenerate one or more paragraphs\n- s: stop here\nChoice",
        ["c", "r", "s"],
        default="c",
    )


def ask_regenerate_target() -> str:
    return prompt_text(
        "Enter paragraph indices to regenerate (comma separated or 'all')",
        required=True,
    )


def show_stage3_summary(timeline_path: Path) -> dict[str, Any]:
    payload = json.loads(timeline_path.read_text(encoding="utf-8"))
    print("Stage 3 completed.")
    print(f"timeline: {timeline_path}")
    print(f"status: {payload.get('status')}")
    print(f"missing: {payload.get('missing_paragraph_indices', [])}")
    print("Review suggestions:")
    print("- Edit page_XX.timeline.final.json -> segments[].start")
    print("- For missing paragraphs, set matched=true and provide start")
    print("- end_hint is optional and only used as a reference")
    return payload


def show_stage4_summary(output_video: Path, output_dir: Path) -> None:
    print("Stage 4 completed.")
    print(f"final_video: {output_video}")
    print(f"output_dir:   {output_dir}")
    print("Review suggestions:")
    print("- If audio quality is wrong, go back to Stage 2")
    print("- If timing is wrong, go back to Stage 3 and adjust starts")


def default_stage1_output_dir(config: dict[str, Any]) -> Path | None:
    if config.get("stage1_output_dir"):
        return Path(config["stage1_output_dir"])
    return None


def default_timeline_output(spoken_json: Path, config: dict[str, Any]) -> Path:
    if config.get("timeline_output"):
        return Path(config["timeline_output"])
    page = int(config["page"])
    return spoken_json.parent / f"page_{page:02d}.timeline.final.json"


def default_timeline_debug_dir(config: dict[str, Any]) -> Path | None:
    if config.get("timeline_debug_dir"):
        return Path(config["timeline_debug_dir"])
    page = int(config["page"])
    return OUTPUTS_DIR / "timeline_debug" / f"page_{page:02d}_debug"


def default_compose_output_dir(spoken_json: Path, config: dict[str, Any]) -> Path:
    if config.get("compose_output_dir"):
        return Path(config["compose_output_dir"])
    page_name = f"page_{int(config['page']):02d}_{slugify(Path(config['video']).stem, max_len=36)}"
    return OUTPUTS_DIR / "composed" / page_name


def run_stage1(config: dict[str, Any]) -> dict[str, Any]:
    return prepare_ppt_page(
        ppt_path=Path(config["ppt"]),
        page=int(config["page"]),
        title_indices=config["title_indices"],
        output_dir=default_stage1_output_dir(config),
    )


def run_stage2_profile(config: dict[str, Any]) -> Path:
    return run_voice_profile(
        voice_name=config["voice_name"],
        ref_audio=config["ref_audio"],
        ref_text=config["ref_text"],
        output_dir=Path(config["profile_output_dir"])
        if config.get("profile_output_dir")
        else None,
    )


def run_stage2_voice(
    config: dict[str, Any], spoken_json: Path, profile_path: Path
) -> Path:
    result = run_voice_generate(
        spoken_json=spoken_json,
        profile_path=profile_path,
        voice_name=config.get("voice_name") or profile_path.stem,
        output_dir=Path(config["voice_output_dir"])
        if config.get("voice_output_dir")
        else None,
    )
    return Path(result["manifest_path"])


def rerun_stage2_for_target(
    config: dict[str, Any], spoken_json: Path, profile_path: Path, target: str
) -> Path:
    if target.strip().lower() == "all":
        return run_stage2_voice(config, spoken_json, profile_path)
    indices = [int(item.strip()) for item in target.split(",") if item.strip()]
    manifest_path: Path | None = None
    for paragraph_index in indices:
        result = run_voice_generate(
            spoken_json=spoken_json,
            profile_path=profile_path,
            voice_name=config.get("voice_name") or profile_path.stem,
            paragraph_index=paragraph_index,
            output_dir=Path(config["voice_output_dir"])
            if config.get("voice_output_dir")
            else None,
        )
        manifest_path = Path(result["manifest_path"])
    if manifest_path is None:
        raise ValueError("No valid paragraph indices provided for regeneration.")
    return manifest_path


def run_stage3(config: dict[str, Any], spoken_json: Path) -> Path:
    output = default_timeline_output(spoken_json, config)
    run_timeline_align(
        video=Path(config["video"]),
        spoken_json=spoken_json,
        output=output,
        debug_dir=default_timeline_debug_dir(config),
        api_key=config["api_key"],
        probe_mode=config["probe_mode"],
        probe_times=config["probe_times"],
    )
    return output


def run_stage4(
    config: dict[str, Any], timeline_path: Path, manifest_path: Path
) -> Path:
    return run_video_compose(
        video=Path(config["video"]),
        timeline=timeline_path,
        segments_manifest=manifest_path,
        output_dir=default_compose_output_dir(timeline_path, config),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive NarrateFlow pipeline")
    parser.add_argument("--ppt")
    parser.add_argument("--page", type=int)
    parser.add_argument("--video")
    parser.add_argument("--profile")
    parser.add_argument("--voice-name")
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--title-mode", choices=["first", "none", "manual"])
    parser.add_argument("--title-indices")
    parser.add_argument("--stage1-output-dir")
    parser.add_argument("--profile-output-dir")
    parser.add_argument("--voice-output-dir")
    parser.add_argument("--timeline-output")
    parser.add_argument("--timeline-debug-dir")
    parser.add_argument("--compose-output-dir")
    parser.add_argument(
        "--probe-mode", choices=["keyframes", "times"], default="keyframes"
    )
    parser.add_argument("--probe-times")
    parser.add_argument("--api-key")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = resolve_initial_args(args)

    total = 5

    stage_banner(1, total, "Text Processing")
    stage1_result = run_stage1(config)
    show_stage1_summary(stage1_result)
    if ask_continue_or_stop("Stage 1") == "s":
        return

    profile_path = Path(config["profile"]) if config.get("profile") else None
    if profile_path is None:
        stage_banner(2, total, "Voice Profile Generation")
        profile_path = run_stage2_profile(config)
        config["profile"] = str(profile_path)
        show_profile_summary(profile_path)
    else:
        print()
        print(f"[2/{total}] Voice Profile Generation (skipped, using existing profile)")
        print(f"profile_path: {profile_path}")

    stage_banner(3, total, "Voice Generation")
    manifest_path = run_stage2_voice(
        config, Path(stage1_result["spoken_path"]), profile_path
    )
    while True:
        show_stage2_summary(manifest_path)
        action = ask_stage2_action()
        if action == "c":
            break
        if action == "s":
            return
        target = ask_regenerate_target()
        manifest_path = rerun_stage2_for_target(
            config, Path(stage1_result["spoken_path"]), profile_path, target
        )

    stage_banner(4, total, "Timeline Alignment")
    timeline_path = run_stage3(config, Path(stage1_result["spoken_path"]))
    show_stage3_summary(timeline_path)
    if ask_continue_or_stop("Stage 4") == "s":
        return

    stage_banner(5, total, "Video Composition")
    composed_path = run_stage4(config, timeline_path, manifest_path)
    show_stage4_summary(composed_path, composed_path.parent)


if __name__ == "__main__":
    main()
