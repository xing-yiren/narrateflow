# Parallel Development Tasks

This document splits the current roadmap into low-conflict work packages that can be assigned to separate Claude windows.

## Coordination Rules

- Commit and push after each meaningful stage-module change.
- Keep `README.md`, `DEVELOPMENT.md`, and `run_pipeline.py` aligned whenever CLI behavior changes.
- Treat `--project-dir` as the task boundary.
- Keep API providers as fallback and debugging baselines.
- Do not add runtime LLM evaluation in this phase.
- Avoid changing shared JSON contracts unless the task owns that contract.
- Prefer new focused modules over large edits to shared orchestration files.

## Shared Artifact Contracts

These files are the boundaries between parallel work streams.

```text
<project-dir>/task.json
<project-dir>/understanding/video_understanding.json
<project-dir>/text/page_01.spoken.json
<project-dir>/script/page_01.spoken.json
<project-dir>/voice/segments_manifest.json
<project-dir>/timeline/page_01.timeline.final.json
<project-dir>/compose/page_composed.mp4
```

### `video_understanding.json` minimum stable fields

- `window_id`
- `start_time`
- `end_time`
- `visual_summary`
- `actions`
- `visible_text` or `onscreen_text`
- `objects` or `entities`
- `source_frames`

### `spoken.json` minimum stable fields

- `page`
- `source_type`
- `paragraphs[]`
- `paragraphs[].index`
- `paragraphs[].spoken_text`
- `paragraphs[].is_silent`
- `segments[]` for paragraphs that produce audio

For video-auto, paragraphs may also carry `start_time`, `end_time`, and `source_window_id`.

### `timeline.final.json` minimum stable fields

- `segments[]`
- `segments[].paragraph_index`
- `segments[].spoken_text`
- `segments[].matched`
- `segments[].start`
- `segments[].end_hint`
- `segments[].review_status`

## Priority Overview

### P0

- Stabilize provider abstraction for `understand`.
- Freeze the shared understanding schema.
- Keep Gemini working as the baseline provider.
- Preserve project-dir/task.json behavior.

### P1

- Improve video-auto narration quality.
- Improve script-align timeline matching quality.
- Add clearer debug artifacts for matching and script generation.

### P2

- Add local inference runtime guards.
- Add local VLM provider implementation.
- Add OCR-assisted understanding and matching hooks.
- Improve non-interactive `--yes` coverage.

### P3

- WhisperX/MFA alignment branch.
- GUI or desktop packaging.
- Advanced vLLM/PagedAttention/KV-cache tuning after a local baseline exists.

## Window A: Understanding Provider Abstraction

### Goal

Turn the `understand` stage from Gemini-specific execution into provider-based execution while keeping the current Gemini path fully functional.

### Allowed files

- `timeline_align/video_understanding.py`
- `timeline_align/vl_client.py`
- `pipeline/stages.py`
- `run_pipeline.py`
- Optional new module: `timeline_align/understanding_providers.py`

### Avoid touching

- `timeline_align/video_script.py`
- `timeline_align/run_timeline_align.py`
- `voice_process/`
- `video_compose/`

### Tasks

- Add a provider option such as `--vl-provider gemini|local|mock`.
- Keep `gemini` as the default provider for now.
- Introduce a provider interface that accepts video windows/keyframes and returns normalized understanding entries.
- Add a `mock` or minimal local stub provider that writes compatible `video_understanding.json` without network calls.
- Keep `video_understanding.json` output location unchanged.
- Preserve caching and resume behavior for already understood windows.

### Acceptance

- `python run_pipeline.py --help` works.
- Existing Gemini understand command still works.
- Mock/local stub can produce a valid `video_understanding.json` in project-dir.
- `script` and `timeline` stages do not need schema changes to consume the result.

## Window B: Video-Auto Narration Quality

### Goal

Improve `video_understanding.json -> spoken.json` so generated narration sounds like a continuous product walkthrough rather than isolated frame descriptions.

### Allowed files

- `timeline_align/video_script.py`
- Optional new module: `timeline_align/script_prompt_templates.py`

### Avoid touching

- `run_pipeline.py`
- `pipeline/stages.py`
- `timeline_align/video_understanding.py`
- `timeline_align/run_timeline_align.py`

### Tasks

- Strengthen global summary and previous-window context handling.
- Reduce repeated openings such as "the screen shows" or "this frame shows".
- Add narration-style constraints for SaaS, product, and tutorial walkthroughs.
- Improve fallback generation when Gemini script drafting is disabled.
- Keep human review focused on `<project-dir>/script/page_01.spoken.json`.

### Acceptance

- Given the same `video_understanding.json`, output remains a compatible `spoken.json`.
- Narration is continuous and action-oriented.
- Fallback output is usable as a first draft, even if lower quality than Gemini.
- No runtime LLM evaluation stage is added.

## Window C: Script-Align Timeline Matching

### Goal

Improve matching between an existing script and video understanding windows beyond lightweight token overlap.

### Allowed files

- `timeline_align/run_timeline_align.py`
- Optional new module: `timeline_align/matching.py`

### Avoid touching

- `run_pipeline.py`
- `pipeline/stages.py`
- `timeline_align/video_script.py`
- `timeline_align/video_understanding.py`

### Tasks

- Introduce a scoring layer combining lexical overlap, action terms, visible text, object/entity hints, and monotonic order constraints.
- Add clear debug output for why a paragraph matched a window.
- Preserve `timeline.final.json` compatibility with composition.
- Add hooks for future OCR fields without requiring OCR today.

### Acceptance

- Existing script-align flow still writes `<project-dir>/timeline/page_01.timeline.final.json`.
- Matching debug artifacts explain scores and fallback decisions.
- Compose can consume the timeline without modification.

## Window D: Local Runtime And VRAM Safety

### Goal

Prepare for local VLM and TTS coexistence on RTX 3060 12GB without jumping straight into advanced vLLM work.

### Allowed files

- Optional new module: `pipeline/local_runtime.py`
- Optional new module: `pipeline/process_guard.py`
- Small integration edits in `pipeline/stages.py` or `run_pipeline.py` only if needed

### Avoid touching

- `timeline_align/video_script.py`
- `timeline_align/run_timeline_align.py`
- `video_compose/`

### Tasks

- Design a stage-level subprocess wrapper for local model calls.
- Add a resource cleanup convention after local understand or script stages.
- Add failure handling that preserves existing project artifacts and task state.
- Keep the first implementation simple: process isolation, bounded batch size, explicit cleanup.
- Leave vLLM, PagedAttention, and KV eviction as later optimization, not P0.

### Acceptance

- Local provider failures do not corrupt existing project artifacts.
- Runtime wrapper can be used by a future local VLM provider.
- Gemini/API fallback remains unaffected.

## Window E: Documentation, Validation, And CLI Alignment

### Goal

Keep the repo understandable while other windows modify behavior.

### Allowed files

- `README.md`
- `DEVELOPMENT.md`
- `LOCAL_VIDEO_UNDERSTANDING_PLAN.md`
- `DEVELOPMENT_PARALLEL_TASKS.md`

### Avoid touching

- Core runtime code unless updating help text or CLI docs is impossible otherwise.

### Tasks

- Keep CLI examples aligned with `run_pipeline.py`.
- Append a practical execution plan to `LOCAL_VIDEO_UNDERSTANDING_PLAN.md` without modifying existing earlier content.
- Track commits and major decisions in `DEVELOPMENT.md`.
- Add validation recipes for project-dir, understand, script, timeline, voice, and compose.

### Acceptance

- A new contributor can tell which command to run for each mode.
- Docs clearly distinguish current API baseline from local-first roadmap.
- Docs clearly identify which artifacts belong to project-dir.

## Recommended Parallel Schedule

### Batch 1

Run these in parallel after confirming the tree is stable enough for concurrent work:

- Window A: provider abstraction and schema stabilization
- Window B: video-auto narration improvement
- Window C: script-align matching improvement
- Window E: docs and validation recipes

### Batch 2

Start after Window A has a stable provider entrypoint:

- Window D: local runtime/process guard
- Window A continuation: first real local VLM provider

### Batch 3

Start after a local VLM produces compatible `video_understanding.json`:

- OCR-assisted understanding hooks
- stronger local prompt and schema normalization
- performance measurement on RTX 3060
- optional WhisperX/MFA branch

## Suggested Commit Granularity

- `feat: add understanding provider selection`
- `feat: add mock understanding provider`
- `feat: improve video-auto narration drafting`
- `feat: improve understanding-based timeline matching`
- `feat: add local runtime process guard`
- `doc: add local video understanding execution plan`
- `test: add project-dir validation recipe`

## Validation Fixtures

Use the real sample inputs already established for this repo when a task needs end-to-end checks:

```text
voice profile:
D:\qwen3-tts\outputs\voice_profiles\selina\selina.pt

video:
D:\视频\九问\02_宣传视频剪辑\2026-04-27-JiuwenClaw-TeamSkills\视频素材\视频1-创建团队技能.mp4

script:
D:\视频\九问\02_宣传视频剪辑\2026-04-27-JiuwenClaw-TeamSkills\视频素材\使用团队技能自动生成专家创建团队技能.txt
```

Use a throwaway project directory under `tmp/` or `outputs/` for validation. Do not commit generated media or model weights.
