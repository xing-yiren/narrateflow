from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def project_output_root(project_dir: str | Path | None, video: str | Path | None = None) -> Path:
    if project_dir:
        return Path(project_dir)
    if video:
        return OUTPUTS_DIR / Path(video).stem
    return OUTPUTS_DIR / "project"
