# Development Book

This document tracks project-level development notes, push history, the overall pipeline architecture, and the TODO list.

## Push Log

### Recent pushes

- `d7643e2` — save current pipeline baseline
- `fa6cf39` — add video understanding pipeline stages
- `34916e5` — add understanding-based script alignment
- `53c9244` — doc: update pipeline usage and development notes

### Current documentation / repo notes

- Current baseline: refactored pipeline centered on `run_pipeline.py`, with `README.md` kept in sync as the main usage documentation.
- Local model migration plan: see `LOCAL_VIDEO_UNDERSTANDING_PLAN.md` for the RTX 3060 12GB local VL strategy and staged rollout recommendation.
- Repository cleanup notes:
  - `sample/` contains Huawei Cloud MaaS interface test scripts and can likely be removed when no longer needed.
  - `pipeline/`, `tools/`, and `voice_clone_tool.py` are preserved backups of older implementations after refactoring.
  - `web_ui/` is a lower-priority, incomplete feature area.
  - `wheels/` contains previous installation packages.
  - `models/` contains local deployment model weights and should not be treated as source code.

## Pipeline Architecture

### Top-level mode selection

NarrateFlow supports two business modes that share the same downstream stages:

```text
                  inputs
              +-------------------+
              | video.mp4   (req) |
              | script.txt  (sa)  |
              | profile.pt  (TTS) |
              +---------+---------+
                        |
              +---------+----------+
              |                    |
              v                    v
      +---------------+    +---------------+
      | script-align  |    |  video-auto   |
      | (existing     |    | (video only)  |
      |  script)      |    |               |
      +-------+-------+    +-------+-------+
              |                    |
              v                    v
         text stage           understand stage
         parse script         Gemini understand
              |                    |
              |                    v
              |              script stage
              |              draft narration
              |                    |
              +---------+----------+
                        v
                   spoken.json     <-- main review point
                        |
              +---------+----------+
              |                    |
              v                    v
          voice stage        timeline stage
          per-paragraph      paragraph -> window
          TTS                                 |
              |                    |
              +---------+----------+
                        v
                   compose stage
                   final dubbed video
                        |
                        v
                   page_composed.mp4
```

### Stage data flow — video-auto

```text
video.mp4
   |
   v  keyframe_filter.sample_keyframes
keyframes.json + window_manifest.json
   |
   v  vl_client.call_vl_gemini (gemini-2.5-flash)
video_understanding.json
   |
   v  video_script.generate_spoken_from_understanding
script/page_01.spoken.json        <-- human review
   |
   v  run_voice_generate
voice/segments_manifest.json + segments/*.wav
   |
   v  run_timeline_align (reuses window times)
timeline/page_01.timeline.final.json
   |
   v  run_video_compose
compose/page_composed.mp4
```

### Stage data flow — script-align

```text
script.txt / .pptx                   video.mp4
       |                                |
       v  prepare_ppt_page              v  understand stage
text/page_01.spoken.json   understanding/video_understanding.json
       |                                |
       +---------------+----------------+
                       v
                 run_timeline_align
                 (text overlap + monotonic fallback)
                       |
                       v
       timeline/page_01.timeline.final.json
                       |
                       v  run_voice_generate
       voice/segments_manifest.json
                       |
                       v  run_video_compose
       compose/page_composed.mp4
```

### Project directory layout

```text
<project-dir>/
├── task.json                          # inputs / artifacts state
├── text/                              # script-align spoken.json
│   ├── page_01.extracted.json
│   └── page_01.spoken.json
├── understanding/                     # shared by both modes
│   ├── keyframes/
│   ├── keyframes.json
│   ├── window_manifest.json
│   ├── video_understanding.json
│   ├── gemini_understanding_requests.json
│   └── gemini_understanding_responses.json
├── script/                            # video-auto spoken.json
│   ├── page_01.spoken.json
│   ├── video_script_drafts.json
│   ├── gemini_script_requests.json
│   └── gemini_script_responses.json
├── profile/
│   └── <voice_name>.pt
├── voice/
│   ├── segments_manifest.json
│   └── segments/*.wav
├── timeline/
│   ├── page_01.timeline.final.json
│   └── debug/
└── compose/
    ├── page_retimed_video.mp4
    ├── page_audio.wav
    └── page_composed.mp4
```

### Code module map

```text
run_pipeline.py                       # interactive entry point
│
├── pipeline/
│   ├── stages.py                     # understand / script stage wrappers
│   └── shared.py                     # project_output_root etc.
│
├── text_process/
│   └── run_text_process.py           # .pptx / .txt -> spoken.json
│
├── timeline_align/
│   ├── keyframe_filter.py            # keyframe extraction
│   ├── vl_client.py                  # Gemini + MAAS VL clients
│   ├── video_understanding.py        # Gemini video understanding (new)
│   ├── video_script.py               # narration drafting (new)
│   └── run_timeline_align.py         # timeline align + script-align match
│
├── voice_process/
│   ├── common.py
│   ├── run_voice_profile.py          # voice profile creation
│   └── run_voice_generate.py         # paragraph-level TTS
│
└── video_compose/
    └── run_video_compose.py          # compose + retime + cover + outro
```

### Stage dependency summary

```text
text -------------+
                  +--> spoken.json --> voice --> segments ----+
understand -------+                                            +--> compose --> final mp4
                  +--> video_understanding --> timeline -------+
                              |
                              +--> script --> spoken.json (video-auto)
```

Key points:

- `understand` is the shared video-understanding base for both modes.
- `spoken.json` is the unified human review checkpoint.
- `voice` and `compose` are fully shared between the two modes.
- For video-auto, `timeline` reuses window times directly from `spoken.json`.
- For script-align, `timeline` matches paragraphs to video windows by text overlap with monotonic ordering as fallback.
- `--project-dir` is the source of truth for cross-stage state through `task.json` plus the per-stage subdirectories.

## Near-term plan

1. Finish the README / CLI alignment and keep `README.md` synchronized with `run_pipeline.py`.
2. Continue tightening `video-auto` narration generation so it sounds like a continuous product walkthrough instead of frame-by-frame description.
3. Improve `script-align` matching beyond token overlap + monotonic fallback.
4. Add better non-interactive defaults for `--yes` across all remaining prompts.
5. Decide whether to expose a top-level `video-auto` mode selector instead of stage-by-stage invocation.

## Overall TODO

- [x] Add a shared `understand` stage that writes `video_understanding.json`.
- [x] Add a `script` stage for `video-auto`.
- [x] Let `script-align` reuse `video_understanding.json` for timeline matching.
- [x] Document the pipeline architecture (mode selection, data flow, project layout, module map).
- [ ] Keep `README.md` and `run_pipeline.py` aligned when pipeline behavior changes.
- [ ] Improve `video-auto` script generation quality under real Gemini output.
- [ ] Improve `script-align` matching quality beyond lightweight token overlap.
- [ ] Add a top-level `video-auto` mode selector to `run_pipeline.py`.
- [ ] Cover all remaining interactive prompts under `--yes`.
- [ ] Decide whether to remove `sample/` after confirming Huawei Cloud MaaS tests are no longer needed.
- [ ] Decide how long to retain legacy backups in `pipeline/`, `tools/`, and `voice_clone_tool.py`.
- [ ] Document required local model weights without committing large model artifacts.
- [ ] Finish or explicitly defer `web_ui/` development.
- [ ] Add validation for the main pipeline golden path after each significant pipeline change.
