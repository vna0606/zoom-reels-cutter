# Reel Factory project map

## Runtime flow

`web/app.js` → HTTP routes in `app.py` → `HighlightsManager` in `reel_factory/highlights.py` → transcription in `reel_factory/pipeline.py` → mode renderer.

- Podcast render: `reel_factory/pipeline.py`.
- Broadcast render: `reel_factory/broadcast.py`.
- Broadcast motion planning and HyperFrames execution: `reel_factory/hyperframes_visual.py`.
- HyperFrames template: `hyperframes/broadcast-motion/index.html`.

## Persisted data

- Projects: `workspace/highlights/<project_id>/project.json`.
- Per-candidate work: `workspace/highlights/<project_id>/work/<candidate_id>/`.
- Broadcast evidence commonly includes `broadcast_layout.json`, `visual_plan.json`, `source_frames/`, `hyperframes_project/`, `hyperframes.log`, `motion.mp4`, `broadcast.filter.txt`, and `ffmpeg.log`.
- Completed outputs and ZIPs are historical artifacts. A source-code change does not alter them until rerender.

## Compatibility

- Persist `project_mode` as `podcast|broadcast`; treat missing mode as `podcast`.
- Keep top-level project status lifecycle compatible with the existing UI.
- Failed rendering must not block rendering another candidate when reusable candidates remain.
- Retry analysis should reuse a saved transcript rather than repeat ASR.

## Broadcast composition invariants

- Output: 1080×1920, 30 fps, H.264/yuv420p.
- Audio: map from Zoom input `0:a:0?`, never from the silent motion input.
- Background: HyperFrames motion video fills the portrait canvas.
- Speaker: persistent lower-left overlay with centrally configured geometry; keep the last tracked participant through pauses and replace it only after a confirmed speaker change.
- Subtitles: final ASS pass after speaker compositing.
- Presentation screenshots: extract only from the detected presentation crop; reject unusable dark/flat samples; copy selected assets into the materialized project.

## HyperFrames contract

- Use the pinned local CLI resolved by `resolve_hyperframes_command()`.
- Keep all media local; do not depend on CDN assets.
- Root and timed clip duration must be literal after materialization.
- Register exactly one paused GSAP main timeline before framework readiness.
- Run `hyperframes check` with snapshots and representative timestamps after render-affecting template changes.

## Validation and server lifecycle

- Unit suite: `python -m unittest discover -v`.
- Optional real HyperFrames smoke is gated by `RUN_HYPERFRAMES_SMOKE=1` and must use the existing local CLI.
- Health: `GET http://127.0.0.1:8765/api/health` returns `build_id`, `workspace_id`, `pid`, and `busy`.
- Restart only through `launch_reel_factory.py --no-open` after `busy: false`.
- A launcher seeing a stale same-workspace build performs a controlled restart. It must not stop a foreign workspace or an active render.
