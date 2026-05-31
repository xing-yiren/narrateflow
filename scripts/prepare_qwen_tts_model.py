from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download or prepare local Qwen-TTS model directory for NarrateFlow"
    )
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
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
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT") or None,
        help="Optional Hugging Face endpoint or mirror, e.g. https://hf-mirror.com",
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
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
        print(f"Using HF endpoint: {args.endpoint}")
    print(f"Target local directory: {local_dir}")
    try:
        snapshot_download(
            repo_id=args.repo_id,
            endpoint=args.endpoint,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Model download failed. If direct access to huggingface.co is blocked, retry with\n"
            "  python scripts/prepare_qwen_tts_model.py --endpoint https://hf-mirror.com\n"
            f"Original error: {exc}"
        ) from exc
    print(f"model_dir: {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
