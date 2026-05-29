from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_command(name: str) -> bool:
    return shutil.which(name) is not None


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def resolve_device_without_project_import(requested: str) -> tuple[str | None, str | None, str | None]:
    if not has_module("torch"):
        return None, None, "torch is not installed"
    import torch

    normalized = (requested or "auto").strip().lower()
    aliases = {
        "gpu": "cuda:0",
        "cuda": "cuda:0",
        "cuda0": "cuda:0",
        "apple": "mps",
        "metal": "mps",
        "mac": "mps",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"", "auto"}:
        if torch.cuda.is_available():
            resolved = "cuda:0"
        else:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend and mps_backend.is_available() and mps_backend.is_built():
                resolved = "mps"
            else:
                resolved = "cpu"
    else:
        resolved = normalized

    if resolved.startswith("cuda"):
        if not torch.cuda.is_available():
            return resolved, None, "CUDA requested but unavailable"
        capability = torch.cuda.get_device_capability(0)
        dtype = "torch.bfloat16" if capability[0] >= 8 else "torch.float16"
        batch_size = "4"
    elif resolved == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if not (mps_backend and mps_backend.is_available() and mps_backend.is_built()):
            return resolved, None, "MPS requested but unavailable"
        dtype = "torch.float16"
        batch_size = "2"
    elif resolved == "cpu":
        dtype = "torch.float32"
        batch_size = "1"
    else:
        return resolved, None, f"unsupported device: {requested}"
    return resolved, f"{dtype}, batch_size={batch_size}", None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check NarrateFlow local deployment readiness")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-config", default="config/video_mode.toml")
    args = parser.parse_args()

    print("== NarrateFlow deployment check ==")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"python: {sys.version.split()[0]}")
    print(f"python_exe: {sys.executable}")
    print(f"device_requested: {args.device}")
    print(f"ffmpeg: {'yes' if check_command('ffmpeg') else 'no'}")
    print(f"ffprobe: {'yes' if check_command('ffprobe') else 'no'}")

    config_path = Path(args.video_config)
    print(f"video_config_exists: {config_path.exists()}")

    required_modules = [
        "torch",
        "numpy",
        "soundfile",
        "librosa",
        "cv2",
        "google.genai",
    ]
    missing_modules = [module for module in required_modules if not has_module(module)]
    print(
        "python_modules: "
        + ("ok" if not missing_modules else "missing " + ", ".join(missing_modules))
    )

    resolved, details, error = resolve_device_without_project_import(args.device)
    if resolved:
        print(f"resolved_device: {resolved}")
    if details:
        print(f"runtime_defaults: {details}")
    if error:
        print(f"device_check_error: {error}")

    qwen_tts_ok = has_module("qwen_tts")
    print(f"qwen_tts: {'yes' if qwen_tts_ok else 'no'}")

    model_root = ROOT / "models" / "Qwen"
    model_candidates = sorted(model_root.glob("Qwen3-TTS-12Hz-1*Base")) if model_root.exists() else []
    print(
        "qwen_tts_model: "
        + (str(model_candidates[0]) if model_candidates else "missing models/Qwen/Qwen3-TTS-12Hz-1*Base")
    )

    if (
        missing_modules
        or not check_command("ffmpeg")
        or not check_command("ffprobe")
        or error
        or not qwen_tts_ok
    ):
        print("deployment_check: incomplete")
        return 1

    if not model_candidates:
        print("deployment_check: dependencies_ok_model_missing")
        return 0

    print("deployment_check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
