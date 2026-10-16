---
name: hyperframes-reel-factory
description: Develop, diagnose, validate, and safely restart this repository's Reel Factory and HyperFrames broadcast-video pipeline. Use when Codex works on mode «Эфир», motion visual plans, active-speaker selection or visibility, source-presentation screenshots, subtitles, FFmpeg composition, highlight rendering, HyperFrames checks, cached render artifacts, or the local Reel Factory server.
---

# HyperFrames Reel Factory

Maintain the repository's two video modes without rediscovering its runtime path or breaking completed projects.

## Start with project state

1. Read [references/project-map.md](references/project-map.md) completely.
2. For any video, animation, or motion-composition edit, load the installed `hyperframes` skill before acting. Load its narrower domain skills only when required.
3. Inspect the relevant source and persisted project artifacts before changing code. Never infer the live render path from filenames alone.
4. Query `/api/health` before server lifecycle actions. Treat `busy: true` as a hard restart barrier.

## Preserve the product contract

- Keep `podcast` as the backward-compatible default and isolate `broadcast` behavior.
- Keep broadcast output at 1080×1920 with Zoom audio, ASS subtitles, a full-frame motion background, and the tracked active speaker in the lower-left safe area.
- Keep the last tracked speaker visible through pauses and brief ambiguous detections. Replace the inset only after another participant is confirmed.
- Keep the current active-speaker scale and placement constants centralized in `reel_factory/broadcast.py`; make tests state the intended geometry.
- Use source-presentation frames only when they provide grounded visual evidence. Crop the detected presentation region, not the whole Zoom frame.
- Keep HyperFrames compositions offline, deterministic, seek-safe, and driven by one paused registered main timeline.
- Bump `VISUAL_PROMPT_VERSION` when prompt/schema/normalization changes must invalidate existing visual-plan caches.
- Preserve old output files until a replacement render completes successfully. Do not silently mutate historical MP4 files.

## Implement through the live pipeline

1. Trace UI request → `app.py` → `reel_factory/highlights.py` → the mode-specific renderer.
2. Put broadcast layout, speaker tracking, and FFmpeg composition in `reel_factory/broadcast.py`.
3. Put visual-plan generation, source-frame attachment, HyperFrames materialization, CLI resolution, and fallback behavior in `reel_factory/hyperframes_visual.py`.
4. Edit `hyperframes/broadcast-motion/index.html` only for the motion composition itself.

## Verify in increasing-cost order

1. Run focused unit tests for changed helpers.
2. Run `python -m unittest discover -v` and `py_compile` for changed Python modules.
3. For composition edits, materialize a short project and run the pinned local HyperFrames `check` at representative early/middle/late timestamps. Do not use `npx`, `pnpx`, `dlx`, or network-fetched packages.
4. For FFmpeg graph edits, run a 1–3 second synthetic smoke that exercises speaker switches, persistence through pauses, dimensions, and Zoom audio mapping.
5. Inspect real-source snapshots when visual correctness depends on Zoom layout or presentation crops.
6. Do not launch a long or final render unless the user asks for it.

## Restart safely

1. Confirm `/api/health` belongs to this workspace and reports `busy: false`.
2. Run `launch_reel_factory.py --no-open`; let the launcher replace only a stale build from the same workspace.
3. Query `/api/health` again and verify a new `build_id`, a live PID, and `busy: false`.
4. Tell the user to refresh the page and rerender. Existing MP4 files retain their previous visual until rebuilt.

## Report evidence

Lead with the user-visible outcome. State which checks actually ran, whether the server was restarted, and whether a real clip was rendered or only synthetic/materialized evidence was used.
