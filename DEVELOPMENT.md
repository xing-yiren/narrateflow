# Development Book

This document tracks project-level development notes, push history, and the overall TODO list.

## Push Log

### Recent pushes

- `d7643e2` — save current pipeline baseline
- `fa6cf39` — add video understanding pipeline stages
- `34916e5` — add understanding-based script alignment

### Current documentation / repo notes

- Current baseline: refactored pipeline centered on `run_pipeline.py`, with `README.md` kept in sync as the main usage documentation.
- Repository cleanup notes:
  - `sample/` contains Huawei Cloud MaaS interface test scripts and can likely be removed when no longer needed.
  - `pipeline/`, `tools/`, and `voice_clone_tool.py` are preserved backups of older implementations after refactoring.
  - `web_ui/` is a lower-priority, incomplete feature area.
  - `wheels/` contains previous installation packages.
  - `models/` contains local deployment model weights and should not be treated as source code.

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
- [ ] Keep `README.md` and `run_pipeline.py` aligned when pipeline behavior changes.
- [ ] Improve `video-auto` script generation quality under real Gemini output.
- [ ] Improve `script-align` matching quality beyond lightweight token overlap.
- [ ] Decide whether to remove `sample/` after confirming Huawei Cloud MaaS tests are no longer needed.
- [ ] Decide how long to retain legacy backups in `pipeline/`, `tools/`, and `voice_clone_tool.py`.
- [ ] Document required local model weights without committing large model artifacts.
- [ ] Finish or explicitly defer `web_ui/` development.
- [ ] Add validation for the main pipeline golden path after each significant pipeline change.
