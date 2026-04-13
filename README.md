# NarrateFlow

NarrateFlow is a human-in-the-loop pipeline for turning PPTs or documents into dubbed videos.

It extracts narration text, generates paragraph-level voice audio, aligns audio to a target video timeline with a vision-language model, and renders the final dubbed video.

## Environment

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.13 | Recommended runtime |
| FFmpeg / FFprobe | Required | Used for frame extraction and video composition |
| CUDA | Recommended | Speeds up local TTS inference |
| Local TTS backend | Qwen-TTS | Used in `voice_process` |
| VL backend | Qwen2.5-VL-72B via MAAS API | Used in `timeline_align` |
| API key | `MAAS_API_KEY` | Required for Stage 4 VL calls |

## Project Structure

```text
text_process/
  run_text_process.py      # build page text artifacts
  config/
    pronunciation_rules.json

voice_process/
  run_voice_profile.py     # build voice profile
  run_voice_generate.py    # build paragraph audio
  common.py                # shared voice utilities

timeline_align/
  run_timeline_align.py    # build timeline
  keyframe_filter.py       # extract keyframes
  vl_client.py             # call VL backend

video_compose/
  run_video_compose.py     # render final video
```

## Workflow

### Stage 1. Text Processing

**What it does**

Extracts text from a PPT page, applies terminology rules, and prepares narration-ready spoken text.

**Arguments**

- `--ppt`: `.pptx` file
- `--page`: page number
- `--title-mode`: `first`, `none`, or `manual`
- `--title-indices`: title paragraph indices when `--title-mode manual`
- `--output-dir`: optional custom output directory

**Output path**

- `outputs/scripts/<page_title>/page_XX.extracted.json`
  Raw extracted paragraphs from the source page
- `outputs/scripts/<page_title>/page_XX.spoken.json`
  Normalized narration text used by later stages

**Command**

```bash
python text_process/run_text_process.py --ppt "inputs/example.pptx" --page 1
```

No-title page example:

```bash
python text_process/run_text_process.py \
  --ppt "inputs/example.pptx" \
  --page 1 \
  --title-mode none
```

### Stage 2. Voice Profile Generation

**What it does**

Builds a reusable voice profile from reference audio.

**Arguments**

- `--voice-name`: output profile name
- `--ref-audio`: reference audio file
- `--ref-text`: reference text for the audio
- `--output-dir`: optional custom output directory

**Output path**

- `outputs/voice_profiles/<voice_name>/<voice_name>.pt`
  Saved voice profile file

If `--output-dir` is provided, the profile is written to that directory instead.

**Command**

```bash
python voice_process/run_voice_profile.py \
  --voice-name reference_voice \
  --ref-audio "path/to/ref.wav" \
  --ref-text "reference text"
```

### Stage 3. Voice Generation

**What it does**

Generates paragraph-level audio from `page_XX.spoken.json` using a saved voice profile.

**Arguments**

- `--spoken-json`: Stage 1 spoken file
- `--profile`: voice profile file
- `--voice-name`: output voice folder name
- `--paragraph-index`: optional, regenerate one paragraph only
- `--output-dir`: optional custom output directory

**Output path**

- `outputs/<voice_name>/<page_title>/segments_manifest.json`
  Main audio index file for the page
- `outputs/<voice_name>/<page_title>/segments/*.wav`
  Paragraph-level audio files

If `--output-dir` is provided, all generated audio artifacts are written to that directory instead.

**Command**

```bash
python voice_process/run_voice_generate.py \
  --spoken-json "outputs/scripts/<page_title>/page_01.spoken.json" \
  --profile "outputs/voice_profiles/<voice_name>/<voice_name>.pt" \
  --voice-name <voice_name>
```

Single-paragraph regeneration:

```bash
python voice_process/run_voice_generate.py \
  --spoken-json "outputs/scripts/<page_title>/page_01.spoken.json" \
  --profile "outputs/voice_profiles/<voice_name>/<voice_name>.pt" \
  --voice-name <voice_name> \
  --paragraph-index 4
```

### Stage 4. Timeline Alignment

**What it does**

Matches narration paragraphs to video moments using keyframes and a vision-language model, then builds a reviewable timeline.

**Arguments**

- `--video`: target video file
- `--spoken-json`: Stage 1 spoken file
- `--output`: final timeline file path
- `--probe-mode`: usually `keyframes`
- `--probe-times`: initial keyframe times for probing

**Output path**

- `outputs/scripts/<page_title>/page_XX.timeline.final.json`
  Main reviewable timeline file
- `outputs/scripts/<page_title>/page_XX.timeline.final.json.debug.json`
  Full debug timeline with internal matching details

**Command**

```bash
python timeline_align/run_timeline_align.py \
  --video "path/to/video.mp4" \
  --spoken-json "outputs/scripts/<page_title>/page_01.spoken.json" \
  --output "outputs/scripts/<page_title>/page_01.timeline.final.json" \
  --probe-mode keyframes \
  --probe-times "0,10,20,30"
```

### Stage 5. Video Composition

**What it does**

Composes the final dubbed video. If a video segment is shorter than the audio duration, the local video segment is retimed instead of truncating audio.

**Arguments**

- `--video`: target video file
- `--timeline`: Stage 4 final timeline file
- `--segments-manifest`: Stage 3 audio manifest file
- `--output-dir`: output folder for composed results

**Output path**

- `outputs/composed/<page_name>/page_audio.wav`
  Final composed audio track
- `outputs/composed/<page_name>/page_retimed_video.mp4`
  Video after local retiming
- `outputs/composed/<page_name>/page_composed.mp4`
  Final dubbed video
- `outputs/composed/<page_name>/page_plan.json`
  Composition and retiming plan

**Command**

```bash
python video_compose/run_video_compose.py \
  --video "path/to/video.mp4" \
  --timeline "outputs/scripts/<page_title>/page_01.timeline.final.json" \
  --segments-manifest "outputs/<voice_name>/<page_title>/segments_manifest.json" \
  --output-dir "outputs/composed/page_01"
```

## Human Review

NarrateFlow assumes human review between stages.

- After Stage 1: check extracted and spoken text
- After Stage 2: check paragraph-level audio quality
- After Stage 4: check starts, missing paragraphs, and ordering
- After Stage 5: check pacing, retiming quality, and final alignment

## Timeline Semantics

- `start` is the primary insertion point
- `end_hint` is a reference window only
- actual playback duration is decided in Stage 5 using audio duration, buffer, and the next segment start

## Limitations

- some source PPT files may contain malformed text encoding
- timeline alignment quality depends on UI visibility, subtitle availability, and visual distinction between adjacent segments
- human review is still recommended for production-quality output
- some sentence endings may require spoken-text rewriting for better TTS delivery

## Roadmap

- support more document input formats
- improve keyframe selection with stable-frame sampling between change points
- add richer keyframe typing such as subtitle-change, scene-change, and stable-fill
- prioritize gap reprobe frames from keyframe candidates instead of uniform time sampling
- explore OCR-assisted timeline alignment for subtitle-bearing frames
- improve adjacent-paragraph conflict resolution in timeline generation
- add paragraph-level ASR review and regeneration workflow
- provide a cleaner end-to-end CLI entrypoint
