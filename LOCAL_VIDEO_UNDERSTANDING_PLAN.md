# Local Video Understanding Plan

This document records a practical local-model plan for the `understand` stage under an `RTX 3060 12GB` constraint, while keeping the current Qwen3-TTS local voice path unchanged.

## Goal

Current state:

- TTS already runs locally through Qwen3-TTS.
- Video understanding and video-to-script drafting still depend on Gemini.
- The main local replacement target is the shared `understand` stage, because both `script-align` and `video-auto` depend on it.

Target state:

- Keep the current artifact contract unchanged: `keyframes.json`, `window_manifest.json`, `video_understanding.json`, and downstream `spoken.json`.
- Add a local VL backend that can replace Gemini for at least short-window understanding.
- Preserve Gemini as the quality baseline and fallback path until local quality is proven on real samples.

## Models Already Present In This Repo

### `models/OpenBMB/MiniCPM-V-4-int4`

- Explicitly positioned for image, multi-image, and video understanding.
- `int4` quantization makes it the most realistic first local backend on `RTX 3060 12GB`.
- Best fit for the first end-to-end POC because the current pipeline already works on sampled keyframes and short windows rather than full long-video dense decoding.

Recommendation: use this as the first local replacement target for `understand`.

### `models/google/gemma-4-E2B-it`

- Small multimodal Gemma 4 variant with strong general reasoning and multimodal support.
- Better viewed as a compact reasoning-oriented baseline than a purpose-built local video model.
- Useful for comparing frame-bundle summarization quality, but it is less direct than MiniCPM-V for video-window understanding.

Recommendation: keep as the second comparison backend, not the first integration target.

### `models/Qwen/Qwen3-TTS-12Hz-1___7B-Base`

- This is the local TTS base model already aligned with the current voice pipeline.
- It is not a video-understanding model, so it should remain isolated to `voice_process/`.

Recommendation: do not mix TTS model changes into the first local VL migration.

## External Candidates Worth Tracking

These are not the first recommendation for this repo today, but they are worth tracking for a later refresh cycle.

### `Qwen3-VL-4B-Instruct`

- Strong current open multimodal candidate in the `4B` range.
- Good chance of being a better long-term local backend than older `3B`-class models.
- Worth testing if we later want a cleaner local default beyond MiniCPM.

### `Molmo2-4B`

- Interesting open `4B` multimodal candidate with video support.
- Promising as a research comparison option, especially if quantized local inference is stable.

### `7B` video models in general

- Models such as VideoLLaMA3-7B or similar usually exceed a clean `12GB` deployment budget unless quantized aggressively and sometimes partially offloaded.
- They are not the right starting point for a stable local pipeline on this machine class.

Recommendation: keep the local-first shortlist in the `2B` to `4B` range for now.

## RTX 3060 12GB Constraints

This constraint should drive the design more than benchmark marketing.

### What fits

- `2B` to `4B` multimodal models, preferably quantized.
- Single-request inference with small batch size.
- Windowed understanding built from sampled frames.
- Short prompts plus compact structured outputs.

### What does not fit well

- Full-resolution dense frame ingestion over long clips.
- `7B+` multimodal models in comfortable non-quantized mode.
- Large parallel batches across multiple windows.
- Long-context experiments that combine many windows into one giant prompt.

### Practical operating rules

- Keep per-window frame count low, for example `8` to `24` keyframes first.
- Prefer sparse temporal sampling over brute-force dense video feeding.
- Process windows serially.
- Keep the local model focused on structured scene understanding, not polished narration generation.
- Leave `script` generation on Gemini until local `understand` output is reliable.

## Recommended Architecture Change

The current pipeline should change at the backend boundary, not at the artifact boundary.

### Keep unchanged

- `run_pipeline.py` stage flow.
- `timeline_align/video_understanding.py` output schema.
- `script` stage input contract.
- `script-align` reuse of `video_understanding.json`.

### Add

- A backend selector such as `--vl-backend gemini|minicpm|gemma`.
- A local runtime wrapper that accepts sampled frames and returns the same normalized understanding payload.
- Backend-specific prompt adapters, while keeping one shared normalized output schema.

### Best integration point

The cleanest insertion point is around the existing VL client boundary instead of rewriting downstream stages. In this repo that means the local path should be added alongside the current Gemini path used by the `understand` stage, then normalized before writing `video_understanding.json`.

## Model Recommendation By Stage

### Stage 1: local `understand`

- Primary: `MiniCPM-V-4-int4`
- Secondary comparison: `gemma-4-E2B-it`
- Baseline: current Gemini output

This gives the highest probability of shipping a usable local path without destabilizing the rest of the pipeline.

### Stage 2: local `script`

- Do not replace immediately.
- First verify whether local `understand` output is factually complete and temporally stable enough.
- If the understanding payload is still weaker than Gemini, keep script drafting remote.

### Stage 3: optional local full path

- Only attempt after a real-sample evaluation shows acceptable quality on `video-auto`.
- At that point compare whether a newer `4B` model such as `Qwen3-VL-4B-Instruct` is worth importing.

## Evaluation Plan

Do not judge this by one demo clip. Build a small fixed benchmark from real project material.

### Evaluation set

- `5` to `10` representative videos.
- Mix product walkthrough, slide narration, UI operation, and subtitle-heavy footage.
- Freeze the same sampled windows for every backend.

### Metrics

- Factual coverage: are key on-screen actions and entities captured?
- Temporal ordering: does the summary reflect the right sequence?
- Narration usefulness: does the output provide enough material for downstream script drafting?
- Stability: do repeated runs drift materially?
- Cost and speed: wall-clock latency per window.
- Resource fit: peak VRAM and OOM frequency on `RTX 3060 12GB`.

### Acceptance bar for first merge

- No repeated OOM on the benchmark set.
- Output schema fully compatible with current downstream code.
- Average latency acceptable for offline processing.
- Quality not obviously worse than Gemini on simple product-demo windows.

The first merge does not need to beat Gemini. It needs to be usable, stable, and reversible.

## Implementation Order

### Phase 1

- Add backend selection for the `understand` stage.
- Integrate `MiniCPM-V-4-int4` using existing sampled keyframes rather than raw full-video ingestion.
- Write normalized `video_understanding.json` with the same schema as Gemini.

### Phase 2

- Run side-by-side comparison on fixed samples.
- Tune frame count, prompt shape, and output normalization.
- Add `gemma-4-E2B-it` as a comparison backend if it reveals better reasoning on ambiguous windows.

### Phase 3

- Promote local backend to supported workflow for `script-align` first.
- Keep Gemini as fallback for `video-auto` until narration quality is validated.

### Phase 4

- Re-evaluate whether to import a newer `4B` class model such as `Qwen3-VL-4B-Instruct`.
- Only then consider replacing the `script` stage or offering a fully local mode.

## Bottom-Line Recommendation

If the target machine is `RTX 3060 12GB`, the pragmatic route is:

1. Keep Qwen3-TTS as-is for local voice.
2. Replace only the shared `understand` stage first.
3. Start with `MiniCPM-V-4-int4`.
4. Use `gemma-4-E2B-it` only as a compact comparison backend.
5. Keep Gemini as baseline and fallback until local quality is verified on real samples.

This gives the smallest blast radius, matches the models already present in the repo, and aligns with the actual bottleneck in the current pipeline.
