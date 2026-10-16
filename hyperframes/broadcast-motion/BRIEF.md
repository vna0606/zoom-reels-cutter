---
workflow: general-video
flow: automation
storyboard: generated-by-codex
destination: instagram-reels
aspect: 1080x1920
language: ru
visual_engine: hyperframes
motion_planner: codex
style_preset: runtime-selected
---

# Broadcast Reel automation brief

## HyperFrames Catalog policy

- The complete current Catalog is installed locally under `compositions/` and indexed by `catalog.json`.
- Every scene selects one semantic Catalog block, one motion primitive, and one transition; the Reel selects one caption style.
- Catalog choices must be grounded in the transcript. Branded demos, geography, code, social platforms, currencies, logos, and product UI are forbidden unless the spoken content explicitly supports them.
- Adjacent scenes vary their Catalog grammar. Shader, WebGPU, and 3D treatments are reserved for genuine hero beats.
- The renderer records every choice and its reason in `catalog_selection.json` for review and coverage tracking.

## Intent

Turn a selected segment of a horizontal Zoom broadcast into a portrait Reel whose meaning is explained by a full-frame motion presentation. The original speaker audio is retained. One fixed speaker slot at lower left and an optional second fixed slot at lower right are composited after HyperFrames; identity and side never swap when the active voice changes. Subtitles are added in the same final pass.

## Required pipeline

1. Codex reads the complete clip-local transcript and creates a semantic storyboard.
2. The selected preset's `frame.md` is the normative visual system for every scene.
3. The plan is materialized as a local, offline HyperFrames composition.
4. HyperFrames checks runtime, layout, motion and contrast before rendering.
5. HyperFrames renders a silent, opaque 1080x1920 motion layer.
6. FFmpeg adds the fixed speaker slot(s), ASS subtitles and only the Zoom source audio.

Codex is preferred for the motion plan. If its network stream is unavailable, the renderer first reuses a valid clip-local semantic plan and otherwise builds a deterministic transcript-grounded Catalog plan, recording the fallback provider and reason. A network failure must not prevent video delivery.

## Story direction

- Scene boundaries follow completed thoughts and rhetorical turns, not an equal time grid.
- The first frame states the hook; later frames develop, prove, contrast, explain or close it.
- A headline is an editorial thesis of at most eight words. Body copy adds one new supporting idea.
- All visible facts, values, labels and quotations must be supported by the transcript.
- Graphics show relationships between entities instead of decorating every spoken word.

## Composition requirements

- Alternate layout, visual variant and motion recipe; adjacent scenes must not repeat all three.
- Use grounded numbers, processes, comparisons and source-screen evidence only when present.
- Keep both lower corner safe areas clear so either one or two fixed speaker slots can be enabled.
- Do not add stock media, external URLs, music, logos or unsupported labels.
- Apply the selected visual preset's exact colors, typography, elevation and component rules.

## Quality bar

The expected result is a sequence of compositionally varied, manually directed HyperFrames scenes—not one universal card template with replaced text.
