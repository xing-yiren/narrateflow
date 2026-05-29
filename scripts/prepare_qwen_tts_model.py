from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download or prepare local Qwen-TTS model directory for NarrateFlow"
    )
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen3-TTS-12Hz-Base",
        help="Hugging Face repo id for the Qwen-TTS model",
    )
    parser.add_argument(
        "--target-dir",
        default="models/Qwen",
        help="Directory under which the local model snapshot will be stored",
    )
    parser.add_argument(
        "--local-dir-name",
        default=None,
        help="Optional explicit local directory name. Defaults to the repo name.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install dependencies first in the narrateflow environment."
        ) from exc

    repo_name = args.local_dir_name or args.repo_id.split("/")[-1]
    target_root = Path(args.target_dir)
    local_dir = target_root / repo_name
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model snapshot from: {args.repo_id}")
    print(f"Target local directory: {local_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"model_dir: {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
