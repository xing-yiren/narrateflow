# NarrateFlow

NarrateFlow is a human-in-the-loop pipeline for turning videos and scripts into dubbed videos.

It supports two modes:

- **script-align**: you already have a narration script. NarrateFlow generates voice audio, understands the target video, and aligns each paragraph to a video time window.
- **video-auto**: you only have a target video. NarrateFlow extracts keyframes, asks Gemini to understand each time window, drafts a continuous narration script, then generates voice and renders the final dubbed video.

Both modes converge on the same downstream stages: voice generation, timeline alignment, and video composition.

## Environment

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.13 recommended | Some helper checks also work on 3.9, but `qwen-tts` needs 3.13 |
| FFmpeg / FFprobe | Required | Used for frame extraction and video composition |
| CUDA | Recommended | Speeds up local TTS inference |
| Local TTS backend | Qwen-TTS | Used in `voice_process` |
| VL backend (video understanding) | Gemini (`gemini-2.5-flash`) via `google-genai` | Used in `timeline_align/video_understanding.py` |
| VL backend (legacy timeline probe) | Qwen2.5-VL via MAAS API | Optional, only used by the legacy probe path |
| Gemini API key | `GEMINI_API_KEY` | Required for `understand` / `script` stages |
| MAAS API key | `MAAS_API_KEY` | Optional, only for legacy timeline probe |

### API key precedence

Keys are resolved in this order:

1. CLI flag: `--gemini-api-key`, `--api-key`
2. Environment variable: `GEMINI_API_KEY`, `MAAS_API_KEY`
3. Project root `.env` file: `GEMINI_API_KEY=...`, `MAAS_API_KEY=...`

`.env` is already in `.gitignore` and will not be committed.

## Installation

```bash
pip install -r requirements.txt
```

Notes:

- For GPU acceleration, install `torch` / `torchaudio` / `torchvision` matching your CUDA version.
- `ffmpeg` / `ffprobe` must be installed separately and available in `PATH`.
- `sox` is optional. If unavailable, speed adjustment falls back to `librosa`.
- `qwen-tts==0.1.1` currently requires Python 3.13. If your default Python is older, run the voice stages with a Python 3.13 environment such as `venv313`.

## Recommended Usage

The main entrypoint is the interactive runner:

```bash
python run_pipeline.py
```

Execution modes:

1. `full`: run the current text-first main path end to end (`text -> profile -> voice -> timeline -> compose`)
2. `only`: run a single stage
3. `from`: start from a stage and continue forward

The CLI asks only for the inputs the selected mode and stage actually need.

Note: `video-auto` is already implemented through the `understand` + `script` stages, but it is not yet wired into a dedicated top-level mode selector in `run_pipeline.py`. For now, use the staged commands shown below for `video-auto`.

To skip every interactive confirmation (e.g. inside scripts), add:

```bash
--yes
```

### Project directory

For one task to live under a single folder, pass:

```bash
--project-dir <path/to/job_dir>
```

When a project directory is provided, NarrateFlow will:

- keep a lightweight `task.json` under that directory remembering the latest inputs and artifacts
- place stage outputs under predictable subdirectories
- reuse recorded paths and artifacts for later `only` / `from` runs

Typical layout for one task:

```text
<project-dir>/
├── task.json                 # latest inputs and artifacts
├── text/                     # spoken.json from existing script (script-align)
├── understanding/            # video understanding outputs (video-auto and script-align)
│   ├── keyframes/
│   ├── keyframes.json
│   ├── window_manifest.json
│   ├── video_understanding.json
│   ├── gemini_understanding_requests.json
│   └── gemini_understanding_responses.json
├── script/                   # spoken.json drafted from video understanding (video-auto)
├── profile/                  # voice profile artifacts
├── voice/                    # paragraph-level audio + segments_manifest.json
├── timeline/                 # page_XX.timeline.final.json (+ debug)
└── compose/                  # final dubbed video
```

## Stages

Supported stage names for `--only-stage` / `--from-stage`:

```text
text, understand, script, profile, voice, timeline, compose
```

| Stage | Used by | Purpose |
|---|---|---|
| `text` | script-align | Parse a `.pptx` or `.txt` into `page_XX.spoken.json` |
| `understand` | both | Extract keyframes and ask Gemini to understand each video window |
| `script` | video-auto | Draft a continuous narration `spoken.json` from `video_understanding.json` |
| `profile` | both | Create a voice profile `.pt` from reference audio + text |
| `voice` | both | Generate paragraph-level audio from `spoken.json` and a voice profile |
| `timeline` | both | Produce `timeline.final.json` aligning narration to the video |
| `compose` | both | Render the final dubbed video, optionally with cover and outro |

Notes:

- `text` is only for the script-align mode. `video-auto` skips it because the script is drafted by Gemini in the `script` stage.
- `understand` is the shared video-understanding base used by both modes.
- For video-auto, `script` reads `<project-dir>/understanding/video_understanding.json` and writes `<project-dir>/script/page_01.spoken.json`.
- For script-align, `timeline` will reuse `<project-dir>/understanding/video_understanding.json` if present and match each paragraph to a video window by text similarity, falling back to monotonic ordering when text overlap is weak.

## Common Recipes

### script-align: existing script + target video

```bash
# 1. parse the existing script into spoken.json
python run_pipeline.py --only-stage text \
  --project-dir outputs/my_task \
  --input path/to/script.txt \
  --title-mode none

# 2. understand the target video
python run_pipeline.py --yes --only-stage understand \
  --project-dir outputs/my_task \
  --video path/to/video.mp4

# 3. align narration to video time windows
python run_pipeline.py --yes --only-stage timeline \
  --project-dir outputs/my_task \
  --video path/to/video.mp4

# 4. generate per-paragraph voice (use a Python 3.13 env if needed)
python run_pipeline.py --yes --only-stage voice \
  --project-dir outputs/my_task \
  --profile outputs/voice_profiles/<voice_name>/<voice_name>.pt

# 5. compose the final video
python run_pipeline.py --yes --only-stage compose \
  --project-dir outputs/my_task \
  --video path/to/video.mp4
```

### video-auto: only the target video

```bash
# 1. understand the target video
python run_pipeline.py --yes --only-stage understand \
  --project-dir outputs/my_task \
  --video path/to/video.mp4

# 2. draft a continuous narration spoken.json with Gemini
python run_pipeline.py --yes --only-stage script \
  --project-dir outputs/my_task

# (optional) draft locally without calling Gemini again
python run_pipeline.py --yes --only-stage script \
  --project-dir outputs/my_task \
  --no-gemini-script

# 3. timeline / voice / compose are the same as script-align
```

### Editing the generated script

Before voice generation, review and edit:

```text
<project-dir>/text/page_01.spoken.json   # script-align
<project-dir>/script/page_01.spoken.json # video-auto
```

Edit `paragraphs[].spoken_text` to fix wording. `is_silent: true` marks paragraphs without narration.

### Cover and outro

The interactive runner can ask for a cover intro and an outro page. In non-interactive mode, pass them explicitly:

```bash
--cover-image path/to/cover.png
--cover-paragraph-index 2
--cover-duration-sec 3.0

--outro-image path/to/outro.png
--outro-audio path/to/outro.wav        # fixed audio, or
--outro-text  "<slogan>"                # synthesized with the voice profile
```

## Output Conventions

- Spoken JSON: `paragraphs[]` with `index`, `spoken_text`, `is_silent`, `start_time`, `end_time`, plus segment-level entries for paragraphs that produce audio.
- Voice manifest: `<project-dir>/voice/segments_manifest.json` listing each paragraph's audio file and duration.
- Timeline: `<project-dir>/timeline/page_XX.timeline.final.json` with `segments[]` containing `paragraph_index`, `start`, `end_hint`, `matched`, `review_status`, and (for video-auto) `source_window_id`.
- Final video: `<project-dir>/compose/page_composed.mp4` plus retimed video and audio mixes alongside it.

## Repository Layout

Active code:

- `run_pipeline.py`: main interactive entrypoint, kept in sync with this README
- `pipeline/`: stage wrappers (`stages.py`, `shared.py`)
- `text_process/`: script parsing for `.pptx` / `.txt`
- `timeline_align/`: keyframe extraction, Gemini client, video understanding, video-auto script drafting, timeline alignment
- `voice_process/`: voice profile creation and paragraph-level TTS
- `video_compose/`: final video composition with retiming, cover, and outro

Lower-priority and local-only directories (not part of the active main pipeline):

- `web_ui/`: incomplete web UI, lower priority
- `backup/`: backups of earlier implementations (ignored by Git)
- `models/`: locally deployed model weights (ignored by Git)
- `wheels/`: cached wheel installers (ignored by Git)
- `sample/`: standalone Huawei Cloud MaaS interface examples (candidate for cleanup)

## Limitations

- Video understanding quality depends on Gemini availability and free-tier quota; large videos may hit rate limits.
- Script-align text matching is currently lightweight (token overlap + monotonic ordering); complex scripts may need manual fixes in `timeline.final.json`.
- The legacy MAAS / Qwen2.5-VL probe path is still in the codebase for the keyframe-based timeline probe but is not required for the main video-understanding flow.
- Human review of the generated `spoken.json` is recommended before voice generation.
- The pipeline is page-oriented; multi-page production runs are not yet first-class.

## Roadmap

See `DEVELOPMENT.md` for the rolling push log and the overall TODO list. Highlights:

- continue tightening the video-auto narration prompt and review workflow
- improve script-align matching beyond token overlap
- document required local model weights without committing large artifacts
- decide retention or removal of `sample/`, `backup/`, and legacy backups
- finish or explicitly defer `web_ui/`
