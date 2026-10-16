from __future__ import annotations

import copy
import functools
import hashlib
import html as html_lib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import llm, runtime_paths
from .pipeline import FFMPEG, ROOT, PipelineError, ProgressCallback

import cv2
import numpy as np


HYPERFRAMES_VERSION = "0.8.4"
HYPERFRAMES_TEMPLATE = ROOT / "hyperframes" / "broadcast-motion"
ISOLATED_HYPERFRAMES_RUNTIME = ROOT / ".tools" / "broadcast-motion-runtime-full"
VISUAL_PLAN_VERSION = 3
VISUAL_PROMPT_VERSION = 10
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
HOOK_TITLE_MAX_LENGTH = 120
HOOK_TITLE_DURATION = 2.35
DEFAULT_VISUAL_PRESET = "editorial_lime"
CATALOG_MANIFEST = HYPERFRAMES_TEMPLATE / "catalog.json"
ICON_CATALOG = HYPERFRAMES_TEMPLATE / "icons" / "iconify-lucide.json"
CATALOG_CAPTION_DEFAULT = "caption-editorial-emphasis"
SUPPORTED_CATALOG_CAPTIONS = {
    "caption-pill-karaoke",
    "caption-highlight",
    "caption-editorial-emphasis",
    "caption-kinetic-slam",
    "caption-weight-shift",
    "caption-neon-accent",
    "caption-neon-glow",
    "caption-glitch-rgb",
    "caption-clip-wipe",
}
VISUAL_PRESETS = {
    "editorial_lime": {
        "label": "Editorial Lime",
        "accent": "#C7F54A",
    },
    "itbooster": {
        "label": "ITBooster",
        "accent": "#AEFBC3",
    },
}
ALLOWED_SCENE_KINDS = {
    "statement",
    "quote",
    "stat",
    "steps",
    "comparison",
    "timeline",
    "diagram",
    "process",
    "network",
}
ALLOWED_VISUAL_VARIANTS = {
    "poster", "editorial", "data", "split", "flow", "radial", "ledger", "signal", "source",
}
ALLOWED_COMPOSITIONS = {
    "left-led", "right-led", "full-width", "split-stage", "stacked", "diagonal",
}
ALLOWED_MOTION_RECIPES = {
    "impact", "cascade", "draw", "counter", "track", "reveal", "editorial",
}
KIND_VISUAL_VARIANTS = {
    "statement": "poster",
    "quote": "editorial",
    "stat": "data",
    "comparison": "split",
    "steps": "flow",
    "process": "flow",
    "timeline": "flow",
    "diagram": "radial",
    "network": "radial",
}
KIND_CATALOG_BLOCK_POOLS = {
    "statement": ("split-flap-board", "cosmic-orb", "halftone-field", "weight-wave"),
    "quote": ("split-flap-board", "rack-focus", "vfx-anamorphic-flare", "weight-wave"),
    "stat": ("split-flap-board", "oscilloscope-trace", "weight-wave", "halftone-field"),
    "steps": ("split-flap-board", "weight-wave", "cosmic-orb", "halftone-field"),
    "comparison": ("split-flap-board", "rack-focus", "halftone-field", "weight-wave"),
    "timeline": ("split-flap-board", "oscilloscope-trace", "halftone-field"),
    "diagram": ("cosmic-orb", "halftone-field", "oscilloscope-trace"),
    "process": ("rack-focus", "halftone-field", "cosmic-orb", "split-flap-board"),
    "network": ("cosmic-orb", "halftone-field", "oscilloscope-trace", "vfx-anamorphic-flare"),
}
KIND_CATALOG_BLOCKS = {
    kind: pool[0] for kind, pool in KIND_CATALOG_BLOCK_POOLS.items()
}
KIND_CATALOG_MOTIONS = {
    "statement": "headline-slam",
    "quote": "per-word-rise",
    "stat": "count-up",
    "steps": "stagger-cascade",
    "comparison": "split-tilt-cards",
    "timeline": "state-chip-rail",
    "diagram": "constellation-hub",
    "process": "tracing-beam",
    "network": "radial-surround",
}
KIND_CATALOG_MOTION_POOLS = {
    "statement": ("headline-slam", "per-word-rise", "focus-rack", "line-swap"),
    "quote": ("per-word-rise", "line-swap", "headline-slam"),
    "stat": ("count-up", "focus-rack", "offset-path-traveler"),
    "steps": ("stagger-cascade", "tracing-beam", "state-chip-rail"),
    "comparison": ("split-tilt-cards", "line-swap", "focus-rack"),
    "timeline": ("state-chip-rail", "offset-path-traveler", "tracing-beam"),
    "diagram": ("constellation-hub", "tracing-beam", "offset-path-traveler"),
    "process": ("tracing-beam", "stagger-cascade", "offset-path-traveler"),
    "network": ("radial-surround", "constellation-hub", "stagger-cascade"),
}
# Curated runtime-safe subset of the installed motion components.  These items
# expose all meaningful copy as variables and do not leave product-demo claims
# inside a user's video.  Registry blocks are filtered separately below.
CATALOG_RUNTIME_COMPONENTS = frozenset({
    *KIND_CATALOG_MOTIONS.values(),
    "focus-rack",
    "line-swap",
    "offset-path-traveler",
})
# These installed blocks are either fully content-variable or deliberately
# abstract. Partially bound UI demos remain available for explicit manual
# adaptation, but never enter an automatic transcript-grounded plan.
CATALOG_RUNTIME_BLOCKS = frozenset(
    name for pool in KIND_CATALOG_BLOCK_POOLS.values() for name in pool
) - {"weight-wave"}
CATALOG_TRANSITION_FALLBACKS = (
    "directional-wipe", "fade-through", "type-match-cut", "whip-pan-cut",
    "halftone-dissolve", "zoom-through-transition", "iris-reveal", "match-cut",
    "grid-pixelate-wipe", "parallax-unzoom", "parallax-zoom", "morph-swap",
)
KIND_ICON_POOLS = {
    "statement": ("sparkles", "lightbulb", "target", "zap"),
    "quote": ("quote", "message-circle", "lightbulb", "users"),
    "stat": ("trending-up", "chart-column", "gauge", "percent"),
    "steps": ("list-checks", "route", "layers", "check-circle"),
    "comparison": ("scale", "arrow-left-right", "columns", "search"),
    "timeline": ("clock", "calendar-days", "route", "history"),
    "diagram": ("workflow", "network", "shapes", "git-branch"),
    "process": ("workflow", "settings", "route", "refresh-cw"),
    "network": ("network", "users", "share-2", "message-circle"),
}
CATALOG_FALLBACK_ITEMS = tuple(
    {"name": name, "type": "block", "title": name, "description": "", "tags": []}
    for name in sorted(set(KIND_CATALOG_BLOCKS.values()))
) + tuple(
    {"name": name, "type": "component", "title": name, "description": "", "tags": tags}
    for name, tags in [
        *[(name, ["motion-primitive"]) for name in sorted(set(KIND_CATALOG_MOTIONS.values()))],
        *[(name, ["transition"]) for name in CATALOG_TRANSITION_FALLBACKS],
        (CATALOG_CAPTION_DEFAULT, ["captions", "caption-style"]),
        ("caption-pill-karaoke", ["captions", "caption-style"]),
        ("caption-highlight", ["captions", "caption-style"]),
        ("caption-kinetic-slam", ["captions", "caption-style"]),
        ("caption-weight-shift", ["captions", "caption-style"]),
    ]
)
SOURCE_FRAME_MARKERS = (
    "скрин", "переписк", "чат", "график", "диаграм", "таблиц", "дашборд",
    "экран", "слайд", "показыва", "посмотрите", "сообщени", "интерфейс",
)


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_is_readable_file(path: Path) -> bool:
    """Return false for pnpm hardlinks that Windows can stat but cannot open."""
    try:
        with path.open("rb") as stream:
            stream.read(1)
        return True
    except OSError:
        return False


def _catalog_items() -> tuple[dict, ...]:
    """Load the installed registry snapshot without reaching the network."""
    try:
        payload = json.loads(CATALOG_MANIFEST.read_text(encoding="utf-8"))
        raw_items = payload.get("items") if isinstance(payload, dict) else payload
    except (OSError, json.JSONDecodeError):
        raw_items = None
    if not isinstance(raw_items, list):
        return CATALOG_FALLBACK_ITEMS
    items = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        name = _clean_text(raw.get("name"), 90)
        item_type = str(raw.get("type") or "").removeprefix("hyperframes:")
        if not name or item_type not in {"block", "component"}:
            continue
        tags = [
            _clean_text(tag, 50) for tag in (raw.get("tags") or [])
            if _clean_text(tag, 50)
        ]
        items.append({
            "name": name,
            "type": item_type,
            "title": _clean_text(raw.get("title") or name, 120),
            "description": _clean_text(raw.get("description"), 220),
            "tags": tags,
            "dimensions": raw.get("dimensions") if isinstance(raw.get("dimensions"), dict) else {},
            "duration": raw.get("duration"),
        })
    return tuple(items) or CATALOG_FALLBACK_ITEMS


def _catalog_index() -> dict[str, dict]:
    return {item["name"]: item for item in _catalog_items()}


@functools.lru_cache(maxsize=1)
def _icon_catalog() -> dict[str, str]:
    """Load the bundled Iconify-format subset used by offline renders."""
    try:
        payload = json.loads(ICON_CATALOG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw_icons = payload.get("icons") if isinstance(payload, dict) else None
    if not isinstance(raw_icons, dict):
        return {}
    icons: dict[str, str] = {}
    for raw_name, raw_icon in raw_icons.items():
        name = re.sub(r"[^a-z0-9-]", "", str(raw_name or "").lower())
        body = raw_icon.get("body") if isinstance(raw_icon, dict) else None
        if not name or not isinstance(body, str):
            continue
        # The file is trusted and checked in, but keep its render surface to
        # SVG drawing primitives only before embedding it into motion-plan JSON.
        if re.search(r"<(?:script|style|image|foreignObject)\b|\bon\w+\s*=", body, re.I):
            continue
        icons[name] = body
    return icons


def _icon_names() -> list[str]:
    return sorted(_icon_catalog())


def _stable_choice(options, seed: str, previous: str = "") -> str:
    choices = [str(value) for value in options if value]
    if not choices:
        return ""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % len(choices)
    selected = choices[index]
    if len(choices) > 1 and selected == previous:
        selected = choices[(index + 1) % len(choices)]
    return selected


def _scene_catalog_choice(kind: str, scene_text: str, index: int, previous: str = "") -> str:
    available = [
        name for name in KIND_CATALOG_BLOCK_POOLS[kind]
        if name in CATALOG_RUNTIME_BLOCKS and _catalog_source_info(name, "block") is not None
    ]
    return _stable_choice(available, f"block|{kind}|{index}|{scene_text}", previous) \
        or KIND_CATALOG_BLOCKS[kind]


def _scene_motion_choice(kind: str, scene_text: str, index: int, previous: str = "") -> str:
    available = [
        name for name in KIND_CATALOG_MOTION_POOLS[kind]
        if _catalog_source_info(name, "component") is not None
    ]
    return _stable_choice(available, f"motion|{kind}|{index}|{scene_text}", previous) \
        or KIND_CATALOG_MOTIONS[kind]


def _scene_icon_choice(kind: str, scene_text: str, index: int, previous: str = "") -> str:
    available = [name for name in KIND_ICON_POOLS[kind] if name in _icon_catalog()]
    return _stable_choice(available, f"icon|{kind}|{index}|{scene_text}", previous)


def _catalog_names(item_type: str, *, tag: str | None = None) -> list[str]:
    return sorted({
        item["name"] for item in _catalog_items()
        if item["type"] == item_type and (tag is None or tag in item["tags"])
    })


def _catalog_source_path(name: str, item_type: str) -> Path | None:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", str(name or ""))
    if not safe_name or safe_name != name:
        return None
    if item_type == "block":
        path = HYPERFRAMES_TEMPLATE / "compositions" / f"{safe_name}.html"
    elif item_type == "component":
        path = HYPERFRAMES_TEMPLATE / "compositions" / "components" / f"{safe_name}.html"
    else:
        return None
    return path if _safe_is_file(path) else None


@functools.lru_cache(maxsize=512)
def _catalog_source_info(name: str, item_type: str) -> dict | None:
    """Describe whether an installed item can render locally and deterministically."""
    if item_type == "component" and name not in CATALOG_RUNTIME_COMPONENTS:
        return None
    path = _catalog_source_path(name, item_type)
    if path is None:
        return None
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'data-composition-id="([^"]+)"', source)
    if match is None or "window.__timelines" not in source:
        return None
    remote_scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)["\'][^>]*>\s*</script>', source, re.I)
    unsupported_scripts = [
        src for src in remote_scripts
        if not re.search(r"(?:^|/)gsap(?:\.min)?\.js(?:\?|$)", src, re.I)
    ]
    if unsupported_scripts:
        return None
    item = _catalog_index().get(name, {})
    dimensions = item.get("dimensions") if isinstance(item.get("dimensions"), dict) else {}
    width_match = re.search(r'data-width="([0-9.]+)"', source)
    height_match = re.search(r'data-height="([0-9.]+)"', source)
    duration_match = re.search(r'data-duration="([0-9.]+)"', source)
    width = int(float(dimensions.get("width") or (width_match.group(1) if width_match else 1920)))
    height = int(float(dimensions.get("height") or (height_match.group(1) if height_match else 1080)))
    duration = float(item.get("duration") or (duration_match.group(1) if duration_match else 6))
    return {
        "path": path,
        "source": source,
        "composition_id": match.group(1),
        "width": width,
        "height": height,
        "duration": max(0.5, duration),
        "template": bool(re.search(r"<template(?:\s|>)", source, re.I)),
    }


def _renderable_catalog_names(item_type: str) -> list[str]:
    return [
        name for name in _catalog_names(item_type)
        if _catalog_source_info(name, item_type) is not None
    ]


def _catalog_inventory_for_prompt() -> str:
    renderable_blocks = set(_renderable_catalog_names("block")).intersection(
        CATALOG_RUNTIME_BLOCKS
    )
    renderable_components = set(_renderable_catalog_names("component"))
    lines = []
    for item in _catalog_items():
        name = item["name"]
        is_transition_or_caption = bool(
            {"transition", "transition-primitive", "captions", "caption-style"}
            & set(item["tags"])
        )
        if item["type"] == "block" and name not in renderable_blocks:
            continue
        if item["type"] == "component" and name not in renderable_components and not is_transition_or_caption:
            continue
        tags = ",".join(item["tags"][:5]) or "untagged"
        lines.append(f"{item['type']}:{name} | {tags}")
    return "\n".join(lines)


def _clean_text(value, limit: int) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip()


def normalize_visual_preset(value) -> str:
    preset = str(value or DEFAULT_VISUAL_PRESET).strip().lower()
    if preset not in VISUAL_PRESETS:
        raise PipelineError("Неизвестный визуальный пресет.")
    return preset


def visual_preset_accent(value) -> str:
    return str(VISUAL_PRESETS[normalize_visual_preset(value)]["accent"])


def _preset_dir(value) -> Path:
    preset = normalize_visual_preset(value)
    path = HYPERFRAMES_TEMPLATE / "presets" / preset
    if not path.is_dir():
        raise PipelineError(f"В HyperFrames-шаблоне отсутствует пресет {preset}.")
    return path


def _visual_plan_schema() -> dict:
    block_names = sorted(set(_renderable_catalog_names("block")).intersection(
        CATALOG_RUNTIME_BLOCKS
    ))
    component_names = _renderable_catalog_names("component")
    transition_names = sorted({
        item["name"] for item in _catalog_items()
        if "transition" in item["tags"] or "transition-primitive" in item["tags"]
    })
    caption_names = sorted({
        item["name"] for item in _catalog_items()
        if item["name"] in SUPPORTED_CATALOG_CAPTIONS
    })
    scene_properties = {
        "id": {"type": "string", "maxLength": 24},
        "start": {"type": "number", "minimum": 0},
        "end": {"type": "number", "minimum": 0},
        "kind": {"type": "string", "enum": sorted(ALLOWED_SCENE_KINDS)},
        "eyebrow": {"type": "string", "maxLength": 40},
        "headline": {"type": "string", "maxLength": 76},
        "body": {"type": "string", "maxLength": 170},
        "primary": {"type": "string", "maxLength": 64},
        "secondary": {"type": "string", "maxLength": 90},
        "items": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 54},
        },
        "visual_variant": {
            "type": "string", "enum": sorted(ALLOWED_VISUAL_VARIANTS),
        },
        "composition": {
            "type": "string", "enum": sorted(ALLOWED_COMPOSITIONS),
        },
        "motion_recipe": {
            "type": "string", "enum": sorted(ALLOWED_MOTION_RECIPES),
        },
        "accent_word": {"type": "string", "maxLength": 40},
        "visual_concept": {"type": "string", "maxLength": 180},
        "use_source_frame": {"type": "boolean"},
        "catalog_block": {"type": "string", "enum": block_names},
        "catalog_motion": {"type": "string", "enum": component_names},
        "catalog_transition": {"type": "string", "enum": transition_names},
        "catalog_icon": {"type": "string", "enum": _icon_names()},
        "catalog_reason": {"type": "string", "maxLength": 180},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["version", "caption_style", "scenes"],
        "properties": {
            "version": {"type": "integer", "enum": [VISUAL_PLAN_VERSION]},
            "caption_style": {"type": "string", "enum": caption_names},
            "scenes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    # Codex strict structured output requires every property here
                    # to be listed in required. Optional visible text is represented
                    # by an empty string/array rather than an omitted property.
                    "required": list(scene_properties),
                    "properties": scene_properties,
                },
            },
        },
    }


def _transcript_for_visual_plan(transcript: dict) -> str:
    lines = []
    for segment in transcript.get("segments") or []:
        try:
            start = max(0.0, float(segment["start"]))
            end = max(start, float(segment["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        text = _clean_text(segment.get("text"), 500)
        if text:
            lines.append(f"[{start:.2f}-{end:.2f}] {text}")
    return "\n".join(lines)


def _design_instruction(name: str, visual_preset: str = DEFAULT_VISUAL_PRESET) -> str:
    path = (
        _preset_dir(visual_preset) / name
        if name == "frame.md"
        else HYPERFRAMES_TEMPLATE / name
    )
    if not _safe_is_file(path):
        raise PipelineError(f"В HyperFrames-шаблоне отсутствует обязательный {name}.")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PipelineError(f"Не удалось прочитать HyperFrames-инструкцию {name}.") from exc


def _visual_prompt(transcript: dict, duration: float, candidate: dict | None,
                   visual_preset: str = DEFAULT_VISUAL_PRESET) -> str:
    candidate = candidate or {}
    visual_preset = normalize_visual_preset(visual_preset)
    title = _clean_text(candidate.get("title"), 120)
    hook = _clean_text(candidate.get("hook"), 260)
    brief = _design_instruction("BRIEF.md")
    frame = _design_instruction("frame.md", visual_preset)
    return f"""You are the motion director for Reel Factory. Plan a bespoke, seek-safe
HyperFrames presentation for a vertical Reel made from a Zoom broadcast. Treat the supplied
BRIEF.md as the production contract and frame.md as normative design truth. Your output is the
semantic storyboard consumed by an offline HyperFrames renderer; do not output HTML or prose.
The selected visual preset is {visual_preset}; follow its frame.md exactly for palette, typography,
component treatment and visual density while keeping the semantic scene choices content-led.

Editorial method:
- read the whole transcript first and identify the hook, development, evidence, contrast and payoff;
- place scene boundaries on completed thoughts and rhetorical turns, never on an arbitrary time grid;
- cover 0.00-{duration:.2f} seconds with ordered adjacent scenes, usually 4-9 seconds each;
- headline is a sharp thesis of at most eight words, never a raw transcript fragment or subtitle;
- body adds one new idea in at most two short lines; it must not restate the headline;
- every visible claim, number, label and quotation must be explicitly supported by the transcript;
- use stat only for a complete grounded value. Keep currency/unit/context in primary or secondary;
- items contains up to four short semantic labels used by diagrams, not sentence fragments;
- accent_word is an exact meaningful substring of headline, or an empty string;
- visual_concept briefly explains the relationship the graphic must show, without adding facts;
- use_source_frame is true only for useful on-screen evidence (chat, graph, table, dashboard,
  interface, document). Use source for at most 40% of scenes and never in adjacent scenes.

Visual choreography:
- combine kind, visual_variant, composition and motion_recipe intentionally; they are independent;
- visual_variant: poster, editorial, data, split, flow, radial, ledger, signal or source;
- composition: left-led, right-led, full-width, split-stage, stacked or diagonal;
- motion_recipe: impact, cascade, draw, counter, track, reveal or editorial;
- adjacent scenes must differ in all three direction fields; do not cycle mechanically;
- alternate visual mass across the frame and reserve both lower speaker safe areas; one or two
  fixed speaker slots may be enabled by the renderer;
- prefer a semantic diagram, path, scale, ledger, sequence or typographic poster over generic cards;
- choose one catalog_block, one catalog_motion, one catalog_transition and one catalog_icon
  for every scene;
- catalog_reason must state the transcript-grounded semantic fit in one short sentence;
- catalog_block supplies the scene grammar, catalog_motion the main proof gesture, and
  catalog_transition the handoff. Never select a branded demo, geography, platform UI, code,
  currency, logo or product-specific block unless that exact subject is grounded in the transcript;
- vary catalog choices across adjacent scenes. Prefer portrait blocks for portrait scenes and use
  shader/WebGPU/3D items only for a genuine hero beat, never as decoration;
- catalog_icon comes from the bundled offline Iconify/Lucide subset. Use it as a semantic signpost,
  not decoration, and do not repeat an icon in adjacent scenes;
- choose one caption_style for the whole Reel. It must support the speaking energy and remain
  readable above the speaker safe area; captions still come from exact word timestamps;
- do not invent media, logos, URLs, music, code or unsupported conclusions;
- return only JSON matching the supplied strict schema.

Selected clip title: {title or "(not supplied)"}
Selected clip hook: {hook or "(not supplied)"}

--- BRIEF.md ---
{brief}

--- frame.md ---
{frame}

TRANSCRIPT WITH CLIP-LOCAL TIMECODES:
{_transcript_for_visual_plan(transcript)}

INSTALLED HYPERFRAMES CATALOG (compact type:name | tags index):
{_catalog_inventory_for_prompt()}

LOCAL ICONIFY/LUCIDE ICONS:
{", ".join(_icon_names())}
"""


def validate_visual_plan(plan: dict, duration: float) -> dict:
    """Validate and normalize a model or cached visual plan deterministically."""
    try:
        duration = float(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("Visual-plan duration must be numeric.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Visual-plan duration must be positive and finite.")
    if not isinstance(plan, dict) or int(plan.get("version", 0)) != VISUAL_PLAN_VERSION:
        raise ValueError("Unsupported visual-plan version.")
    raw_scenes = plan.get("scenes")
    if not isinstance(raw_scenes, list) or not 1 <= len(raw_scenes) <= 12:
        raise ValueError("Visual plan must contain between one and twelve scenes.")

    catalog = _catalog_index()
    caption_candidates = set(SUPPORTED_CATALOG_CAPTIONS).intersection(catalog)
    caption_style = str(plan.get("caption_style") or CATALOG_CAPTION_DEFAULT)
    if caption_style not in caption_candidates:
        caption_style = CATALOG_CAPTION_DEFAULT

    scenes = []
    previous_start = -1.0
    for raw in raw_scenes:
        if not isinstance(raw, dict):
            raise ValueError("Every visual-plan scene must be an object.")
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Every visual-plan scene needs numeric start/end values.") from exc
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("Visual-plan scene timestamps are invalid.")
        if start < -0.05 or end > duration + 0.75 or start < previous_start:
            raise ValueError("Visual-plan scenes are outside the clip or unordered.")
        previous_start = start
        kind = str(raw.get("kind") or "")
        if kind not in ALLOWED_SCENE_KINDS:
            raise ValueError(f"Unsupported visual-plan scene kind: {kind}")
        headline = _clean_text(raw.get("headline"), 76)
        if not headline:
            raise ValueError("Every visual-plan scene needs a headline.")
        raw_items = raw.get("items") or []
        if not isinstance(raw_items, list):
            raise ValueError("Visual-plan scene items must be an array.")
        items = [_clean_text(item, 54) for item in raw_items[:4]]
        items = [item for item in items if item]
        layout_safe = bool(raw.get("_layout_safe", False))
        catalog_disabled = bool(raw.get("_catalog_disabled", False))
        requested_variant = str(raw.get("visual_variant") or "")
        variant = requested_variant if requested_variant in ALLOWED_VISUAL_VARIANTS else KIND_VISUAL_VARIANTS[kind]
        requested_composition = str(raw.get("composition") or "")
        composition = requested_composition if requested_composition in ALLOWED_COMPOSITIONS else "left-led"
        requested_motion = str(raw.get("motion_recipe") or "")
        motion_recipe = requested_motion if requested_motion in ALLOWED_MOTION_RECIPES else "reveal"
        use_source_frame = bool(raw.get("use_source_frame", False))
        if layout_safe:
            variant = "flow"
            composition = "stacked"
            motion_recipe = "cascade"
            use_source_frame = False
        elif use_source_frame:
            variant = "source"
        scene_text = " | ".join(filter(None, [
            headline,
            _clean_text(raw.get("body"), 170),
            _clean_text(raw.get("primary"), 64),
            _clean_text(raw.get("secondary"), 90),
            " | ".join(items),
        ]))
        previous_block = scenes[-1]["catalog_block"] if scenes else ""
        previous_motion = scenes[-1]["catalog_motion"] if scenes else ""
        previous_icon = scenes[-1]["catalog_icon"] if scenes else ""
        catalog_block = str(raw.get("catalog_block") or KIND_CATALOG_BLOCKS[kind])
        if (
            catalog_block not in catalog
            or catalog[catalog_block]["type"] != "block"
            or _catalog_source_info(catalog_block, "block") is None
            or catalog_block not in CATALOG_RUNTIME_BLOCKS
        ):
            catalog_block = _scene_catalog_choice(
                kind, scene_text, len(scenes), previous_block,
            )
        catalog_motion = str(raw.get("catalog_motion") or KIND_CATALOG_MOTIONS[kind])
        if (
            catalog_motion not in catalog
            or catalog[catalog_motion]["type"] != "component"
            or _catalog_source_info(catalog_motion, "component") is None
        ):
            catalog_motion = _scene_motion_choice(
                kind, scene_text, len(scenes), previous_motion,
            )
        catalog_transition = str(
            raw.get("catalog_transition")
            or CATALOG_TRANSITION_FALLBACKS[len(scenes) % len(CATALOG_TRANSITION_FALLBACKS)]
        )
        transition_item = catalog.get(catalog_transition)
        if transition_item is None or not (
            "transition" in transition_item["tags"]
            or "transition-primitive" in transition_item["tags"]
        ):
            catalog_transition = CATALOG_TRANSITION_FALLBACKS[
                len(scenes) % len(CATALOG_TRANSITION_FALLBACKS)
            ]
        if scenes and variant == scenes[-1]["visual_variant"] and variant != "source" and not layout_safe:
            alternates = [
                KIND_VISUAL_VARIANTS[kind], "poster", "editorial", "data", "split", "flow",
                "radial", "ledger", "signal",
            ]
            variant = next((item for item in alternates if item != scenes[-1]["visual_variant"]), variant)
        if scenes and composition == scenes[-1]["composition"] and not layout_safe:
            composition = next(
                item for item in (
                    "right-led", "full-width", "split-stage", "stacked", "diagonal", "left-led",
                ) if item != scenes[-1]["composition"]
            )
        if scenes and motion_recipe == scenes[-1]["motion_recipe"] and not layout_safe:
            motion_recipe = next(
                item for item in (
                    "impact", "cascade", "draw", "counter", "track", "reveal", "editorial",
                ) if item != scenes[-1]["motion_recipe"]
            )
        if scenes and catalog_block == previous_block:
            catalog_block = _scene_catalog_choice(
                kind, scene_text + "|adjacent", len(scenes), previous_block,
            )
        if scenes and catalog_motion == previous_motion:
            catalog_motion = _scene_motion_choice(
                kind, scene_text + "|adjacent", len(scenes), previous_motion,
            )
        if scenes and catalog_transition == scenes[-1]["catalog_transition"]:
            catalog_transition = CATALOG_TRANSITION_FALLBACKS[
                len(scenes) % len(CATALOG_TRANSITION_FALLBACKS)
            ]
        accent_word = _clean_text(raw.get("accent_word"), 40)
        if accent_word and accent_word.casefold() not in headline.casefold():
            accent_word = ""
        catalog_icon = str(raw.get("catalog_icon") or "")
        if catalog_icon not in _icon_catalog() or catalog_icon == previous_icon:
            catalog_icon = _scene_icon_choice(
                kind, scene_text, len(scenes), previous_icon,
            )
        scenes.append({
            "start": max(0.0, start),
            "end": min(duration, end),
            "kind": kind,
            "eyebrow": _clean_text(raw.get("eyebrow"), 40),
            "headline": headline,
            "body": _clean_text(raw.get("body"), 170),
            "primary": _clean_text(raw.get("primary"), 64),
            "secondary": _clean_text(raw.get("secondary"), 90),
            "items": items,
            "visual_variant": variant,
            "composition": composition,
            "motion_recipe": motion_recipe,
            "accent_word": accent_word,
            "visual_concept": _clean_text(raw.get("visual_concept"), 180),
            "use_source_frame": use_source_frame,
            "source_frame": _clean_text(raw.get("source_frame"), 520),
            "catalog_block": catalog_block,
            "catalog_motion": catalog_motion,
            "catalog_transition": catalog_transition,
            "catalog_icon": catalog_icon,
            "catalog_reason": _clean_text(raw.get("catalog_reason"), 180),
            "catalog_tags": sorted(set(catalog.get(catalog_block, {}).get("tags") or [])),
            "_layout_safe": layout_safe,
            "_catalog_disabled": catalog_disabled,
        })

    edge_tolerance = min(1.0, duration * 0.08)
    if scenes[0]["start"] > edge_tolerance or scenes[-1]["end"] < duration - edge_tolerance:
        raise ValueError("Visual-plan scenes do not cover the clip edges.")

    boundaries = [0.0]
    minimum_scene = min(1.25, duration / max(1, len(scenes)))
    for previous, current in zip(scenes, scenes[1:]):
        gap = current["start"] - previous["end"]
        if abs(gap) > 1.25:
            raise ValueError("Visual-plan scenes contain a large gap or overlap.")
        boundary = (previous["end"] + current["start"]) / 2.0
        if boundary - boundaries[-1] < minimum_scene:
            raise ValueError("Visual-plan scenes are too short after normalization.")
        boundaries.append(boundary)
    boundaries.append(duration)
    if boundaries[-1] - boundaries[-2] < minimum_scene:
        raise ValueError("The last visual-plan scene is too short.")

    normalized = []
    for index, (scene, start, end) in enumerate(zip(scenes, boundaries, boundaries[1:]), 1):
        normalized.append({
            "id": f"s{index:03d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "kind": scene["kind"],
            "eyebrow": scene["eyebrow"],
            "headline": scene["headline"],
            "body": scene["body"],
            "primary": scene["primary"],
            "secondary": scene["secondary"],
            "items": scene["items"],
            "visual_variant": scene["visual_variant"],
            "composition": scene["composition"],
            "motion_recipe": scene["motion_recipe"],
            "accent_word": scene["accent_word"],
            "visual_concept": scene["visual_concept"],
            "use_source_frame": scene["use_source_frame"],
            "source_frame": scene["source_frame"],
            "catalog_block": scene["catalog_block"],
            "catalog_motion": scene["catalog_motion"],
            "catalog_transition": scene["catalog_transition"],
            "catalog_icon": scene["catalog_icon"],
            "catalog_reason": scene["catalog_reason"],
            "catalog_tags": scene["catalog_tags"],
            "_layout_safe": scene["_layout_safe"],
            "_catalog_disabled": scene["_catalog_disabled"],
        })
    result = {
        "version": VISUAL_PLAN_VERSION,
        "duration": round(duration, 3),
        "hook_title": _clean_text(plan.get("hook_title"), HOOK_TITLE_MAX_LENGTH)
        or normalized[0]["headline"],
        "hook_duration": round(min(HOOK_TITLE_DURATION, max(0.9, duration * 0.45)), 3),
        "caption_style": caption_style,
        "scenes": normalized,
    }
    for key in ("provider_requested", "provider_used", "fallback_reason", "fingerprint"):
        if plan.get(key) is not None:
            result[key] = _clean_text(plan[key], 600)
    if plan.get("visual_preset") is not None:
        result["visual_preset"] = normalize_visual_preset(plan.get("visual_preset"))
    return result


def _hyperframes_check_payload(output: str) -> dict:
    for match in re.finditer(r"(?m)^\{", output or ""):
        try:
            payload = json.loads((output or "")[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("layout"), dict):
            return payload
    return {}


def _layout_repair_plan(plan: dict, check_payload: dict, duration: float) -> tuple[dict | None, list[str]]:
    layout = check_payload.get("layout") if isinstance(check_payload, dict) else None
    if not isinstance(layout, dict):
        return None, []
    findings = [
        item for item in (layout.get("findings") or [])
        if isinstance(item, dict) and item.get("severity") == "error"
    ]
    if not findings:
        return None, []
    for section_name in ("lint", "runtime", "motion", "contrast"):
        section = check_payload.get(section_name) or {}
        if int(section.get("errorCount") or 0) > 0:
            return None, []

    repaired = copy.deepcopy(plan)
    affected_ids: list[str] = []
    times = [float(item["time"]) for item in findings if isinstance(item.get("time"), (int, float))]
    for scene in repaired.get("scenes") or []:
        start = float(scene.get("start") or 0.0)
        end = float(scene.get("end") or duration)
        if not any(start - 0.01 <= timestamp <= end + 0.01 for timestamp in times):
            continue
        scene["_layout_safe"] = True
        scene["_catalog_disabled"] = True
        scene["visual_variant"] = "flow"
        scene["composition"] = "stacked"
        scene["motion_recipe"] = "cascade"
        scene["use_source_frame"] = False
        scene["source_frame"] = ""
        if not scene.get("items"):
            scene["items"] = [
                value for value in (scene.get("primary"), scene.get("secondary"), scene.get("headline"))
                if _clean_text(value, 54)
            ][:4]
        affected_ids.append(str(scene.get("id") or ""))
    if not affected_ids:
        return None, []
    return validate_visual_plan(repaired, duration), affected_ids


def _fallback_kind(text: str) -> str:
    lowered = text.lower()
    if re.search(r"(?<!\w)\d+(?:[.,]\d+)?\s*%?", lowered):
        return "stat"
    if any(marker in lowered for marker in ("во-первых", "во вторых", "шаг", "этап", "пункт")):
        return "steps"
    if any(marker in lowered for marker in ("с одной стороны", "с другой стороны", "в отличие", "но при этом")):
        return "comparison"
    if "?" in text:
        return "quote"
    return "statement"


def _fallback_eyebrow(kind: str) -> str:
    return {
        "stat": "Факт",
        "steps": "Шаги",
        "process": "Процесс",
        "comparison": "Сравнение",
        "timeline": "Последовательность",
        "diagram": "Схема",
        "network": "Связи",
        "quote": "Цитата",
    }.get(kind, "Ключевая мысль")


def _fallback_primary(kind: str, text: str, headline: str) -> str:
    if kind == "stat":
        match = re.search(r"(?<!\w)\d+(?:[.,]\d+)?\s*%?", text)
        if match:
            return _clean_text(match.group(0), 64)
    return _clean_text(headline, 64)


def _source_frame_needed(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in SOURCE_FRAME_MARKERS)


def _headline_from_text(text: str) -> str:
    first = re.split(r"(?<=[.!?])\s+|[;:]\s+", text.strip(), maxsplit=1)[0]
    words = first.split()
    if len(words) > 10:
        first = " ".join(words[:10]).rstrip(",.-") + "…"
    return _clean_text(first, 76) or "Ключевая мысль эфира"


def fallback_visual_plan(transcript: dict, duration: float,
                         candidate: dict | None = None) -> dict:
    """Build a stable transcript-grounded plan without any model call."""
    duration = max(0.1, float(duration))
    segments = []
    for segment in transcript.get("segments") or []:
        try:
            start = max(0.0, float(segment["start"]))
            end = min(duration, float(segment["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        text = _clean_text(segment.get("text"), 500)
        if text and end > start:
            segments.append({"start": start, "end": end, "text": text})
    segments.sort(key=lambda item: item["start"])

    target = 6.5
    scene_count = max(1, min(12, int(math.ceil(duration / target))))
    boundaries = [duration * index / scene_count for index in range(scene_count + 1)]
    candidate = candidate or {}
    scenes = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        selected = [
            item["text"] for item in segments
            if item["end"] > start and item["start"] < end
        ]
        body = _clean_text(" ".join(selected), 170)
        preferred = ""
        if index == 0:
            preferred = _clean_text(candidate.get("hook") or candidate.get("title"), 76)
        headline = preferred or _headline_from_text(body)
        kind = _fallback_kind(body or headline)
        use_source_frame = _source_frame_needed(body or headline)
        item_candidates = [
            _clean_text(item, 54)
            for item in re.split(r"\s*[;•]\s*|(?<=[.!?])\s+", body)
        ]
        items = [item for item in item_candidates if item and item != headline][:4]
        if kind not in {"steps", "comparison", "timeline", "diagram"}:
            items = []
        scene_text = " | ".join(filter(None, [headline, body, " | ".join(items)]))
        previous_block = scenes[-1]["catalog_block"] if scenes else ""
        previous_motion = scenes[-1]["catalog_motion"] if scenes else ""
        previous_icon = scenes[-1]["catalog_icon"] if scenes else ""
        scenes.append({
            "id": f"s{index + 1:03d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "kind": kind,
            "eyebrow": _fallback_eyebrow(kind),
            "headline": headline,
            "body": body if body != headline else "",
            "primary": _fallback_primary(kind, body, headline),
            "secondary": _clean_text(body, 90),
            "items": items,
            "visual_variant": "source" if use_source_frame else KIND_VISUAL_VARIANTS[kind],
            "composition": (
                "left-led", "right-led", "full-width", "split-stage", "stacked", "diagonal",
            )[index % 6],
            "motion_recipe": (
                "impact", "cascade", "draw", "counter", "track", "reveal", "editorial",
            )[index % 7],
            "accent_word": "",
            "visual_concept": "Transcript-grounded emergency plan",
            "use_source_frame": use_source_frame,
            "catalog_block": _scene_catalog_choice(kind, scene_text, index, previous_block),
            "catalog_motion": _scene_motion_choice(kind, scene_text, index, previous_motion),
            "catalog_transition": _stable_choice(
                CATALOG_TRANSITION_FALLBACKS,
                f"transition|{kind}|{index}|{scene_text}",
                scenes[-1]["catalog_transition"] if scenes else "",
            ),
            "catalog_icon": _scene_icon_choice(kind, scene_text, index, previous_icon),
            "catalog_reason": "Deterministic semantic match for the scene kind",
        })
    return validate_visual_plan({
        "version": VISUAL_PLAN_VERSION,
        "hook_title": _clean_text(
            candidate.get("title") or candidate.get("hook"), HOOK_TITLE_MAX_LENGTH,
        ),
        "caption_style": CATALOG_CAPTION_DEFAULT,
        "scenes": scenes,
    }, duration)


def without_photographic_visuals(plan: dict, duration: float) -> dict:
    """Plan variant for broadcasts with no real screen share.

    Drops source-frame stills and the catalog runtime blocks (photo/video-like
    bokeh, flares, device mock-ups) so every scene is built from the template's
    own text, icons and shapes."""
    normalized = validate_visual_plan(plan, duration)
    for scene in normalized["scenes"]:
        scene["use_source_frame"] = False
        scene["source_frame"] = ""
        scene["_catalog_disabled"] = True
        if scene.get("visual_variant") == "source":
            scene["visual_variant"] = KIND_VISUAL_VARIANTS.get(scene.get("kind"), "editorial")
    return normalized


def attach_presentation_frames(video_path: Path, plan: dict, presentation_crop: dict,
                               work_dir: Path, callback: ProgressCallback) -> dict:
    """Attach a small number of useful, sharp screen-share stills to the visual plan."""
    normalized = validate_visual_plan(plan, float(plan.get("duration") or 0))
    requested = []
    for index, scene in enumerate(normalized["scenes"]):
        grounded_text = " ".join([
            scene.get("headline", ""), scene.get("body", ""),
            scene.get("primary", ""), scene.get("secondary", ""),
            " ".join(scene.get("items") or []),
        ])
        if scene.get("use_source_frame") or _source_frame_needed(grounded_text):
            priority = 2 if _source_frame_needed(grounded_text) else 1
            requested.append((priority, index))
    if not requested:
        return normalized
    maximum = max(1, min(3, int(math.ceil(len(normalized["scenes"]) * 0.4))))
    requested = sorted(requested, key=lambda item: (-item[0], item[1]))[:maximum]
    selected_indexes = {index for _, index in requested}
    # Avoid visually repetitive back-to-back source screenshots. Keep the
    # higher-priority/earlier scene and let the neighbour remain motion-led.
    for index in sorted(tuple(selected_indexes)):
        if index - 1 in selected_indexes:
            selected_indexes.discard(index)
    if not selected_indexes:
        return normalized

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return normalized
    frame_dir = work_dir / "source_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    callback(62, "Выбираю уместные кадры исходной презентации")
    crop = {key: int(presentation_crop.get(key, 0)) for key in ("x", "y", "width", "height")}
    for index in sorted(selected_indexes):
        scene = normalized["scenes"][index]
        span = max(0.01, scene["end"] - scene["start"])
        candidates = []
        for ratio in (0.28, 0.52, 0.76):
            timestamp = scene["start"] + span * ratio
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            x = max(0, min(width - 2, crop["x"]))
            y = max(0, min(height - 2, crop["y"]))
            right = max(x + 2, min(width, x + crop["width"]))
            bottom = max(y + 2, min(height, y + crop["height"]))
            still = frame[y:bottom, x:right]
            if still.size == 0:
                continue
            gray = cv2.cvtColor(still, cv2.COLOR_BGR2GRAY)
            black_ratio = float(np.count_nonzero(gray < 12)) / float(gray.size)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            contrast = float(gray.std())
            score = sharpness + contrast * 2.0 - black_ratio * 240.0
            candidates.append((score, black_ratio, still))
        if not candidates:
            continue
        score, black_ratio, still = max(candidates, key=lambda item: item[0])
        if black_ratio > 0.72 or score < 18.0:
            continue
        path = frame_dir / f"{scene['id']}.jpg"
        if not cv2.imwrite(str(path), still, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            continue
        scene["source_frame"] = str(path.resolve())
        scene["use_source_frame"] = True
        scene["visual_variant"] = "source"
    cap.release()
    (work_dir / "visual_plan.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return normalized


def _run_llm(prompt: str, schema: dict, work_dir: Path, provider: str) -> dict:
    """Ask the selected AI provider (Claude or Codex) for the visual plan JSON."""
    return llm.run_llm_json(
        prompt, schema, work_dir, task="visual_plan", provider=provider,
        effort="high", timeout=900,
    )


def build_visual_plan(transcript: dict, duration: float, work_dir: Path,
                      options: dict, candidate: dict | None,
                      callback: ProgressCallback) -> dict:
    """Create or load a strict, cached, clip-local AI visual plan (Claude or Codex)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    requested = str(options.get("provider") or "auto")
    # Motion plans need an AI director; the built-in heuristic is not an option here.
    provider = llm.resolve_provider("auto" if requested == "heuristic" else requested)
    candidate = candidate or {}
    hook_title = _clean_text(
        candidate.get("title") or candidate.get("hook"), HOOK_TITLE_MAX_LENGTH,
    )
    visual_preset = normalize_visual_preset(options.get("visual_preset", DEFAULT_VISUAL_PRESET))
    prompt = _visual_prompt(transcript, duration, candidate, visual_preset)
    request_payload = {
        "visual_prompt_version": VISUAL_PROMPT_VERSION,
        "visual_plan_version": VISUAL_PLAN_VERSION,
        "provider": provider,
        "visual_preset": visual_preset,
        "instruction_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "duration": round(float(duration), 3),
        "candidate": {
            "title": _clean_text(candidate.get("title"), HOOK_TITLE_MAX_LENGTH),
            "hook": _clean_text(candidate.get("hook"), 260),
        },
        "segments": transcript.get("segments") or [],
    }
    fingerprint = hashlib.sha256(
        json.dumps(request_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    plan_path = work_dir / "visual_plan.json"
    fingerprint_path = work_dir / "visual_plan.request.sha256"
    previous_plan = None
    if _safe_is_file(plan_path):
        try:
            previous_plan = validate_visual_plan(
                {
                    **json.loads(plan_path.read_text(encoding="utf-8")),
                    **({"hook_title": hook_title} if hook_title else {}),
                },
                duration,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            previous_plan = None
    callback(55, "Создаю визуальный план по транскрипту")
    if previous_plan is not None and _safe_is_file(fingerprint_path):
        try:
            if fingerprint_path.read_text(encoding="utf-8").strip() == fingerprint:
                callback(61, "Визуальный план готов")
                return previous_plan
        except OSError:
            pass

    raw_output = work_dir / "visual_plan.ai.raw.json"
    try:
        raw_output.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        payload = _run_llm(prompt, _visual_plan_schema(), work_dir, provider)
        raw_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        normalized = validate_visual_plan({
            **payload,
            **({"hook_title": hook_title} if hook_title else {}),
        }, duration)
    except Exception as exc:
        label = provider.capitalize()
        failure = f"{provider}: {type(exc).__name__}: {exc}"
        (work_dir / f"visual_plan.{provider}.error.log").write_text(failure, encoding="utf-8")
        fallback_reason = _clean_text(exc, 600)
        if previous_plan is not None:
            normalized = previous_plan
            provider_used = f"{provider}-cache-resilient"
            callback(59, f"{label} недоступен — использую сохранённый motion-план")
        else:
            normalized = fallback_visual_plan(transcript, duration, candidate)
            provider_used = "local-catalog-resilient"
            callback(59, f"{label} недоступен — создаю локальный каталоговый motion-план")
        (work_dir / "visual_plan.fallback.log").write_text(
            json.dumps({
                "provider_used": provider_used,
                "reason": fallback_reason,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        normalized.update({
            "provider_requested": requested,
            "provider_used": provider_used,
            "fallback_reason": fallback_reason,
            "fingerprint": fingerprint,
            "visual_preset": visual_preset,
        })
        normalized = validate_visual_plan(normalized, duration)
        plan_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fingerprint_path.write_text(fingerprint, encoding="utf-8")
        callback(61, "Визуальный план готов")
        return normalized
    normalized.update({
        "provider_requested": requested,
        "provider_used": provider,
        "fingerprint": fingerprint,
        "visual_preset": visual_preset,
    })
    normalized = validate_visual_plan(normalized, duration)
    plan_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fingerprint_path.write_text(fingerprint, encoding="utf-8")
    callback(61, "Визуальный план готов")
    return normalized


def resolve_hyperframes_command() -> list[str]:
    """Resolve only the pinned local CLI; never invoke npx/dlx or the network."""
    package_path = HYPERFRAMES_TEMPLATE / "package.json"
    node_executable = runtime_paths.node_path()
    if not node_executable:
        raise PipelineError("Не найден Node.js для HyperFrames. Запустите установщик («Установить (Mac).command» или «Установить (Windows).cmd»).")
    if not _safe_is_file(package_path):
        raise PipelineError("Не найден локальный HyperFrames-проект broadcast-motion.")
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
        installed_version = str((package.get("devDependencies") or {}).get("hyperframes") or "")
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError("Не удалось проверить локальную версию HyperFrames.") from exc
    if installed_version != HYPERFRAMES_VERSION:
        raise PipelineError(
            f"Ожидался локальный HyperFrames {HYPERFRAMES_VERSION}, найден {installed_version or 'unknown'}."
        )
    isolated_cli = (
        ISOLATED_HYPERFRAMES_RUNTIME
        / "node_modules" / "hyperframes" / "bin" / "hyperframes.mjs"
    )
    linked_cli = HYPERFRAMES_TEMPLATE / "node_modules" / "hyperframes" / "bin" / "hyperframes.mjs"
    cli_entry = next(
        (candidate for candidate in (isolated_cli, linked_cli) if _safe_is_readable_file(candidate)),
        None,
    )
    if cli_entry is None:
        pnpm_root = HYPERFRAMES_TEMPLATE / "node_modules" / ".pnpm"
        try:
            candidates = sorted(
                pnpm_root.glob(f"hyperframes@{HYPERFRAMES_VERSION}_*/node_modules/hyperframes/bin/hyperframes.mjs")
            )
        except OSError:
            candidates = []
        cli_entry = next((candidate for candidate in candidates if _safe_is_readable_file(candidate)), None)
    if cli_entry is None:
        raise PipelineError("Не найден локальный CLI HyperFrames. Переустановите зависимости шаблона.")
    return [node_executable, str(cli_entry)]


def resolve_hyperframes_media_tools() -> tuple[Path, Path]:
    """Resolve local media tools without relying on the process PATH."""
    configured_ffmpeg = os.environ.get("HYPERFRAMES_FFMPEG_PATH", "").strip()
    configured_ffprobe = os.environ.get("HYPERFRAMES_FFPROBE_PATH", "").strip()
    ffmpeg_path = Path(configured_ffmpeg) if configured_ffmpeg else Path(FFMPEG)
    ffprobe_candidates = [
        Path(configured_ffprobe) if configured_ffprobe else None,
        Path(runtime_paths.ffprobe_path() or ""),
        ffmpeg_path.with_name(f"ffprobe{runtime_paths.EXE}"),
    ]
    ffprobe_path = next(
        (candidate for candidate in ffprobe_candidates if candidate and _safe_is_file(candidate)),
        None,
    )
    if not _safe_is_file(ffmpeg_path):
        raise PipelineError("Не найден локальный FFmpeg для рендера HyperFrames.")
    if ffprobe_path is None:
        raise PipelineError(
            "Не найден локальный FFprobe для рендера HyperFrames. "
            "Установите комплект медиабинарников проекта и повторите рендер."
        )
    return ffmpeg_path.resolve(), ffprobe_path.resolve()


_TRANSIENT_HYPERFRAMES_ERRORS = (
    "eperm",
    "ebusy",
    "resource busy",
    "targetcloseerror",
    "target closed",
    "browser has disconnected",
    "failed to launch the browser",
    "econnreset",
    "socket hang up",
)


def _run_hyperframes_process(command: list[str], *, cwd: Path, env: dict,
                             timeout: int, attempts: int = 3) -> subprocess.CompletedProcess:
    """Retry only transient OS/browser failures; semantic check failures stay immediate."""
    last_result: subprocess.CompletedProcess | None = None
    last_timeout: subprocess.TimeoutExpired | None = None
    for attempt in range(max(1, attempts)):
        try:
            result = subprocess.run(
                command, cwd=cwd, env=env,
                text=True, encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
            )
            last_result = result
            output = (result.stdout or "").lower()
            if result.returncode == 0:
                return result
            if not any(marker in output for marker in _TRANSIENT_HYPERFRAMES_ERRORS):
                return result
        except subprocess.TimeoutExpired as exc:
            last_timeout = exc
        if attempt + 1 < max(1, attempts):
            time.sleep(0.4 * (attempt + 1))
    if last_result is not None:
        return last_result
    assert last_timeout is not None
    raise last_timeout


_CATALOG_GENERIC_VARIABLES = (
    {"id": "hfTitle", "type": "string", "label": "Scene title", "default": "Main idea"},
    {"id": "hfBody", "type": "string", "label": "Scene body", "default": "Supporting context"},
    {"id": "hfPrimary", "type": "string", "label": "Primary value", "default": "Result"},
    {"id": "hfSecondary", "type": "string", "label": "Secondary value", "default": "Context"},
    {"id": "hfItems", "type": "string", "label": "Scene items JSON", "default": "[]"},
    {"id": "hfAccent", "type": "color", "label": "Scene accent", "default": "#C7F54A"},
)


def _catalog_variable_declarations(source: str, *, include_generic: bool = False) -> list[dict]:
    match = re.search(
        r"data-composition-variables\s*=\s*(['\"])(.*?)\1",
        source,
        re.I | re.DOTALL,
    )
    declarations: list[dict] = []
    if match:
        try:
            raw = json.loads(html_lib.unescape(match.group(2)))
            if isinstance(raw, list):
                declarations = [item for item in raw if isinstance(item, dict) and item.get("id")]
        except json.JSONDecodeError:
            declarations = []
    if include_generic:
        known = {str(item.get("id")) for item in declarations}
        declarations.extend(copy.deepcopy(item) for item in _CATALOG_GENERIC_VARIABLES if item["id"] not in known)
    return declarations


def _set_catalog_variable_declarations(source: str, declarations: list[dict]) -> str:
    encoded = html_lib.escape(json.dumps(declarations, ensure_ascii=False, separators=(",", ":")), quote=True)
    attribute = f"data-composition-variables='{encoded}'"
    pattern = re.compile(r"data-composition-variables\s*=\s*(['\"])(.*?)\1", re.I | re.DOTALL)
    if pattern.search(source):
        return pattern.sub(attribute, source, count=1)
    return re.sub(r"<html\b", f"<html {attribute}", source, count=1, flags=re.I)


def _strip_catalog_remote_resources(source: str) -> str:
    source = re.sub(
        r"\s*<script[^>]+src=['\"]https?://[^'\"]+['\"][^>]*>\s*</script>\s*",
        "\n",
        source,
        flags=re.I,
    )
    source = re.sub(
        r"\s*<link[^>]+href=['\"]https?://[^'\"]+['\"][^>]*>\s*",
        "\n",
        source,
        flags=re.I,
    )
    source = re.sub(r"@import\s+(?:url\()?['\"]?https?://[^;]+;", "", source, flags=re.I)
    return re.sub(r"url\(\s*['\"]?https?://[^)]+\)", "none", source, flags=re.I)


def _catalog_adapter_script(root_id: str) -> str:
    safe_root = json.dumps(root_id, ensure_ascii=False)
    return f"""
<script data-reel-factory-catalog-adapter>
  (function () {{
    "use strict";
    var vars = window.__hyperframes && window.__hyperframes.getVariables
      ? window.__hyperframes.getVariables() : {{}};
    var root = document.getElementById({safe_root});
    if (!root) return;
    var items = [];
    try {{ items = JSON.parse(String(vars.hfItems || "[]")); }} catch (_) {{ items = []; }}
    if (!Array.isArray(items)) items = [];
    var primitive = root.querySelector("[data-hf-primitive-data]");
    if (primitive) {{
      try {{
        var payload = JSON.parse(primitive.textContent.replace(/\\s*[\\r\\n]+\\s*/g, " "));
        if (Array.isArray(payload.messages)) {{
          var messages = items.length ? items : [vars.hfTitle, vars.hfBody].filter(Boolean);
          payload.messages = messages.slice(0, 6).map(function (value, index) {{
            return {{ side: index % 2 ? "outgoing" : "incoming", text: String(value) }};
          }});
          primitive.textContent = JSON.stringify(payload);
        }}
      }} catch (_) {{}}
    }}
    var values = [vars.hfTitle, vars.hfBody, vars.hfPrimary, vars.hfSecondary]
      .concat(items).map(function (value) {{ return String(value || "").trim(); }}).filter(Boolean);
    if (!values.length) return;
    var nodes = root.querySelectorAll(
      "h1, h2, h3, p, .headline, .title, .subtitle, .caption, .label, .value, [data-hf-text]"
    );
    var cursor = 0;
    Array.prototype.forEach.call(nodes, function (node) {{
      if (node.children.length || node.closest("[data-hf-primitive-data]") || !node.textContent.trim()) return;
      node.textContent = values[Math.min(cursor, values.length - 1)];
      cursor += 1;
    }});
  }})();
</script>
"""


def _find_js_object_end(source: str, open_brace: int) -> int | None:
    depth = 0
    quote = ""
    escaped = False
    for index in range(open_brace, len(source)):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in {'"', "'", "`"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _inject_catalog_config_overrides(source: str) -> str:
    marker = re.search(r"\bvar\s+CONFIG\s*=\s*\{", source)
    if marker is None:
        return source
    open_brace = source.find("{", marker.start())
    close_brace = _find_js_object_end(source, open_brace)
    if close_brace is None:
        return source
    insertion = r'''
        var __hfCatalogVars = window.__hyperframes && window.__hyperframes.getVariables
          ? window.__hyperframes.getVariables() : {};
        var __hfCatalogItems = [];
        try { __hfCatalogItems = JSON.parse(String(__hfCatalogVars.hfItems || "[]")); } catch (_) {}
        if (!Array.isArray(__hfCatalogItems)) __hfCatalogItems = [];
        var __hfTitle = String(__hfCatalogVars.hfTitle || __hfCatalogVars.hfPrimary || "");
        var __hfBody = String(__hfCatalogVars.hfBody || __hfCatalogVars.hfSecondary || "");
        if (Object.prototype.hasOwnProperty.call(CONFIG, "title") && __hfTitle) CONFIG.title = __hfTitle;
        if (Object.prototype.hasOwnProperty.call(CONFIG, "text") && __hfTitle) CONFIG.text = __hfTitle;
        if (Object.prototype.hasOwnProperty.call(CONFIG, "headline") && __hfTitle) CONFIG.headline = __hfTitle;
        if (Object.prototype.hasOwnProperty.call(CONFIG, "label") && __hfTitle) CONFIG.label = __hfTitle;
        if (Object.prototype.hasOwnProperty.call(CONFIG, "caption") && __hfBody) CONFIG.caption = __hfBody;
        if (Object.prototype.hasOwnProperty.call(CONFIG, "subtitle") && __hfBody) CONFIG.subtitle = __hfBody;
        if (Object.prototype.hasOwnProperty.call(CONFIG, "nodes") && __hfCatalogItems.length) {
          CONFIG.nodes = __hfCatalogItems.slice(0, 5).map(function (label) { return { label: String(label) }; });
        }
        if (Object.prototype.hasOwnProperty.call(CONFIG, "rows") && __hfCatalogItems.length) {
          CONFIG.rows = __hfCatalogItems.slice(0, 5).map(function (label, index) {
            return { label: String(label), value: index === 0 ? __hfBody : "" };
          });
        }
        if (Object.prototype.hasOwnProperty.call(CONFIG, "items") && __hfCatalogItems.length) {
          CONFIG.items = __hfCatalogItems.slice(0, 6);
        }
'''
    return source[:close_brace + 1] + insertion + source[close_brace + 1:]


def _ensure_catalog_root_contract(source: str, info: dict) -> str:
    composition_id = re.escape(str(info["composition_id"]))
    pattern = re.compile(
        rf'(<(?!html\b)[a-zA-Z][^>]*\bdata-composition-id="{composition_id}"[^>]*)(>)',
        re.I,
    )
    match = pattern.search(source)
    if match is None:
        return source
    tag = match.group(1)
    additions = []
    for attribute, value in (
        ("data-start", "0"),
        ("data-duration", f"{info['duration']:.3f}"),
        ("data-width", str(info["width"])),
        ("data-height", str(info["height"])),
    ):
        if re.search(rf"\b{attribute}=(['\"])[^'\"]*\1", tag, re.I):
            tag = re.sub(
                rf"\b{attribute}=(['\"])[^'\"]*\1",
                f'{attribute}="{value}"',
                tag,
                count=1,
                flags=re.I,
            )
        else:
            additions.append(f'{attribute}="{value}"')
    replacement = tag + (" " + " ".join(additions) if additions else "") + match.group(2)
    return source[:match.start()] + replacement + source[match.end():]


def _retime_catalog_clip(source: str, duration: float) -> str:
    """Keep the catalog's primary clip alive for the complete scene."""
    pattern = re.compile(
        r'(<[a-zA-Z][^>]*\bclass=(?P<quote>["\'])[^"\']*\bclip\b[^"\']*(?P=quote)[^>]*)(>)',
        re.I,
    )
    match = pattern.search(source)
    if match is None:
        return source
    tag = match.group(1)
    value = f"{duration:.3f}"
    if re.search(r'\bdata-duration=(["\'])[^"\']*\1', tag, re.I):
        tag = re.sub(
            r'\bdata-duration=(["\'])[^"\']*\1',
            f'data-duration="{value}"',
            tag,
            count=1,
            flags=re.I,
        )
    else:
        tag += f' data-duration="{value}"'
    replacement = tag + match.group(3)
    return source[:match.start()] + replacement + source[match.end():]


def _normalise_catalog_block(source: str, info: dict) -> str:
    source = _strip_catalog_remote_resources(source)
    declarations = _catalog_variable_declarations(source, include_generic=True)
    source = _set_catalog_variable_declarations(source, declarations)
    root_id_match = re.search(r'<[^>]+\bid="([^"]+)"[^>]+data-composition-id=', source, re.I)
    root_id = root_id_match.group(1) if root_id_match else "root"
    adapter = _catalog_adapter_script(root_id)
    if info["template"]:
        source = re.sub(
            r"(<script(?![^>]*\bsrc=)[^>]*>)",
            lambda match: adapter + match.group(1),
            source,
            count=1,
            flags=re.I,
        )
        return _ensure_catalog_root_contract(_inject_catalog_config_overrides(source), info)

    styles = "\n".join(re.findall(r"<style[^>]*>.*?</style>", source, re.I | re.DOTALL))
    body_match = re.search(r"<body[^>]*>(.*?)</body>", source, re.I | re.DOTALL)
    if body_match is None:
        raise PipelineError(f"Catalog block {info['composition_id']} has no body.")
    body = re.sub(r"<style[^>]*>.*?</style>", "", body_match.group(1), flags=re.I | re.DOTALL)
    for attribute in ("data-composition-id", "data-start", "data-duration", "data-width", "data-height", "data-fps"):
        body = re.sub(rf'\s+{attribute}="[^"]*"', "", body, count=1, flags=re.I)
    body = re.sub(
        r"(<script(?![^>]*\bsrc=)[^>]*>)",
        lambda match: adapter + match.group(1),
        body,
        count=1,
        flags=re.I,
    )
    wrapped = f'''<!doctype html>
<html lang="en">
  <head><meta charset="UTF-8" /></head>
  <body>
    <template>
      {styles}
      <div id="root" data-composition-id="{html_lib.escape(info['composition_id'])}" data-start="0" data-duration="{info['duration']:.3f}" data-width="{info['width']}" data-height="{info['height']}" data-fps="30" style="position:absolute;inset:0;width:{info['width']}px;height:{info['height']}px;overflow:hidden">
        {body}
      </div>
    </template>
  </body>
</html>'''
    wrapped = _set_catalog_variable_declarations(wrapped, declarations)
    return _ensure_catalog_root_contract(_inject_catalog_config_overrides(wrapped), info)


def _scene_number_parts(scene: dict) -> tuple[float, str, str]:
    raw = " ".join(str(scene.get(key) or "") for key in ("primary", "headline", "body"))
    match = re.search(r"[-+]?\d[\d\s.,]*", raw)
    if match is None:
        return 100.0, "", "%"
    token = match.group(0).strip()
    try:
        value = float(token.replace(" ", "").replace(",", "."))
    except ValueError:
        value = 100.0
    primary = str(scene.get("primary") or "")
    primary_match = re.search(r"[-+]?\d[\d\s.,]*", primary)
    if primary_match:
        return value, primary[:primary_match.start()].strip(), primary[primary_match.end():].strip()
    return value, "", ""


def _catalog_variable_values(scene: dict, source: str, accent: str) -> dict:
    declarations = _catalog_variable_declarations(source, include_generic=True)
    title = _clean_text(scene.get("headline") or scene.get("primary"), 120)
    body = _clean_text(scene.get("body") or scene.get("secondary"), 180)
    primary = _clean_text(scene.get("primary") or title, 90)
    secondary = _clean_text(scene.get("secondary") or body, 120)
    items = [_clean_text(item, 70) for item in (scene.get("items") or []) if _clean_text(item, 70)]
    if not items:
        items = [value for value in (primary, secondary) if value]
    number, prefix, suffix = _scene_number_parts(scene)
    def short_label(value: str, limit: int = 18) -> str:
        cleaned = _clean_text(value, 80)
        if len(cleaned) <= limit:
            return cleaned
        clipped = cleaned[:limit + 1].rsplit(" ", 1)[0].strip()
        return clipped or cleaned[:limit].rstrip()
    generic = {
        "hfTitle": title,
        "hfBody": body,
        "hfPrimary": primary,
        "hfSecondary": secondary,
        "hfItems": json.dumps(items, ensure_ascii=False),
        "hfAccent": accent,
    }
    values: dict = {}
    for declaration in declarations:
        variable_id = str(declaration.get("id") or "")
        lowered = re.sub(r"[^a-z0-9]", "", variable_id.lower())
        variable_type = str(declaration.get("type") or "string")
        role = str(declaration.get("role") or "")
        max_length = int(declaration.get("maxLength") or 0)
        def fitted(value: str, fallback_limit: int = 70) -> str:
            return _clean_text(value, max_length or fallback_limit)
        indexed_match = re.search(r"(?:message|check)([1-4])(?:label|value)?$", lowered)
        indexed_value = ""
        if indexed_match:
            item_index = int(indexed_match.group(1)) - 1
            indexed_value = items[item_index] if item_index < len(items) else (body or secondary)
        if variable_id in generic:
            values[variable_id] = generic[variable_id]
        elif lowered in {"text", "boardtext", "title", "titlel1", "notiftitle", "headlinetop", "cardtitle"}:
            values[variable_id] = fitted(title or primary, 80)
        elif lowered in {"titlel2", "subtitle", "headlineaccent"}:
            values[variable_id] = fitted(secondary or body or primary, 80)
        elif lowered in {"contactname", "appname", "cardtop"}:
            values[variable_id] = fitted(_clean_text(scene.get("eyebrow"), 32) or primary, 32)
        elif lowered in {"carddomain", "footertext", "eccta"}:
            values[variable_id] = fitted(secondary or body, 70)
        elif lowered == "ecproof":
            values[variable_id] = fitted(primary or title, 70)
        elif indexed_match and variable_type == "string":
            values[variable_id] = fitted(indexed_value, 70)
        elif variable_type == "color" and any(token in lowered for token in ("accent", "brand", "color", "colour")):
            values[variable_id] = accent
        elif variable_type == "number" and lowered in {"end", "value", "count", "number", "percent", "progress"}:
            values[variable_id] = number
        elif variable_type == "number" and lowered == "start":
            values[variable_id] = 0
        elif variable_type == "string" and lowered in {"prefix", "numberprefix"}:
            values[variable_id] = prefix
        elif variable_type == "string" and lowered in {"suffix", "unit", "numberunit"}:
            values[variable_id] = suffix
        elif variable_type == "string" and any(token in lowered for token in ("labels", "items", "steps", "options", "nodes")):
            values[variable_id] = ", ".join(short_label(item, 16) for item in items)
        elif variable_type == "string" and lowered in {"travelerlabel", "hublabel"}:
            values[variable_id] = fitted(
                short_label((items[0] if items else "") or _clean_text(scene.get("accent_word"), 30) or primary, 20),
                20,
            )
        elif variable_type == "string" and lowered in {"detailtext", "stattext", "metrictext", "primarytext"}:
            values[variable_id] = fitted(_clean_text(scene.get("accent_word"), 24) or (items[0] if items else primary), 24)
        elif variable_type == "string" and lowered in {"underlineword", "accentword"}:
            values[variable_id] = fitted(_clean_text(scene.get("accent_word"), 30), 30)
        elif variable_type == "string" and any(token in lowered for token in ("body", "subtitle", "caption", "description", "support")):
            values[variable_id] = fitted(body or secondary, 100)
        elif variable_type == "string" and any(token in lowered for token in ("title", "headline", "heading", "quote", "statement", "prompt", "centerlabel", "subject")):
            values[variable_id] = fitted(title or primary, 80)
        elif variable_type == "string" and role == "content" and lowered not in {"path", "svgpath", "src", "url", "code", "json"}:
            if lowered.endswith(("a", "1", "left", "before")):
                chosen = items[0] if items else primary
            elif lowered.endswith(("b", "2", "right", "after")):
                chosen = items[1] if len(items) > 1 else secondary
            elif "label" in lowered:
                chosen = primary or title
            else:
                chosen = title or primary
            values[variable_id] = fitted(
                short_label(chosen, 18) if "label" in lowered else chosen,
                18 if "label" in lowered else 80,
            )
    return values


def _catalog_runtime_style(name: str) -> str:
    overrides = {
        "focus-rack": (
            ".fr-kicker{color:#d6d9df!important}"
            ".fr-label{color:#f4f4ef!important;font-size:3.6cqw!important;line-height:.95!important}"
        ),
        "offset-path-traveler": ".opt-label{font-size:1.3cqw!important;line-height:1.15!important}",
        "constellation-hub": ".ch-node{font-size:1.75cqw!important}",
    }
    css = overrides.get(name, "")
    return f"<style data-reel-factory-catalog-fit>{css}</style>" if css else ""


def _copy_catalog_dependencies(source_path: Path, project_dir: Path, source: str) -> None:
    references = set(re.findall(r'(?:src|href)=["\']([^"\']+)["\']', source, re.I))
    references.update(re.findall(r'url\(\s*["\']?([^\)"\']+)', source, re.I))
    for reference in references:
        if not reference or reference.startswith(("http://", "https://", "data:", "#")):
            continue
        clean_reference = reference.split("?", 1)[0].split("#", 1)[0]
        candidates = (
            (source_path.parent / clean_reference).resolve(),
            (HYPERFRAMES_TEMPLATE / clean_reference).resolve(),
        )
        source_asset = next((path for path in candidates if _safe_is_file(path)), None)
        if source_asset is None:
            continue
        try:
            relative_asset = source_asset.relative_to(HYPERFRAMES_TEMPLATE.resolve())
        except ValueError:
            continue
        destination = project_dir / relative_asset
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_asset, destination)


def _materialize_catalog_items(plan: dict, project_dir: Path, visual_preset: str) -> None:
    generated_compositions = project_dir / "compositions"
    if generated_compositions.exists():
        resolved = generated_compositions.resolve()
        if resolved.parent != project_dir.resolve() or resolved.name != "compositions":
            raise PipelineError("Unsafe generated catalog directory.")
        shutil.rmtree(resolved)
    accent = visual_preset_accent(visual_preset)
    for scene in plan.get("scenes", []):
        if scene.get("source_frame") or scene.get("_catalog_disabled"):
            scene["catalog_runtime"] = {}
            continue
        block_name = str(scene.get("catalog_block") or "")
        motion_name = str(scene.get("catalog_motion") or "")
        block_info = _catalog_source_info(block_name, "block")
        motion_info = _catalog_source_info(motion_name, "component")
        block_is_safe = bool(block_info and block_name in CATALOG_RUNTIME_BLOCKS)
        if block_is_safe or motion_info is None:
            name, item_type, info = block_name, "block", block_info
        else:
            name, item_type, info = motion_name, "component", motion_info
        if info is None:
            scene["catalog_runtime"] = {}
            continue
        source_path: Path = info["path"]
        source_relative = source_path.relative_to(HYPERFRAMES_TEMPLATE)
        scene_id = re.sub(r"[^a-zA-Z0-9_-]", "-", str(scene.get("id") or "scene"))
        relative_path = source_relative.with_name(f"{scene_id}-{source_relative.name}")
        target_path = project_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_id = re.sub(
            r"[^a-zA-Z0-9_-]", "-", f"{info['composition_id']}-{scene_id}",
        )
        runtime_duration = max(0.8, float(scene.get("end") or 0) - float(scene.get("start") or 0))
        runtime_info = {
            **info,
            "composition_id": runtime_id,
            "duration": runtime_duration,
        }
        original_id = re.escape(str(info["composition_id"]))
        runtime_source = re.sub(
            rf"(['\"]){original_id}\1",
            lambda match: f"{match.group(1)}{runtime_id}{match.group(1)}",
            info["source"],
        )
        normalized = (
            _normalise_catalog_block(runtime_source, runtime_info)
            if item_type == "block"
            else _ensure_catalog_root_contract(
                _strip_catalog_remote_resources(runtime_source), runtime_info,
            )
        )
        normalized = _retime_catalog_clip(normalized, runtime_duration)
        compatibility_style = _catalog_runtime_style(name)
        if compatibility_style:
            if "</template>" in normalized:
                normalized = normalized.replace("</template>", compatibility_style + "</template>", 1)
            else:
                normalized = normalized.replace("</head>", compatibility_style + "</head>", 1)
        target_path.write_text(normalized, encoding="utf-8")
        _copy_catalog_dependencies(source_path, project_dir, normalized)
        scene["catalog_runtime"] = {
            "name": name,
            "src": relative_path.as_posix(),
            "composition_id": runtime_id,
            "width": info["width"],
            "height": info["height"],
            "variables": _catalog_variable_values(scene, normalized, accent),
        }


def materialize_hyperframes_project(plan: dict, duration: float, safe_areas: list[dict],
                                    work_dir: Path,
                                    visual_preset: str = DEFAULT_VISUAL_PRESET) -> dict:
    """Create an isolated render project from the checked immutable template."""
    visual_preset = normalize_visual_preset(
        visual_preset or plan.get("visual_preset", DEFAULT_VISUAL_PRESET)
    )
    callback_plan = validate_visual_plan({**plan, "visual_preset": visual_preset}, duration)
    template_path = HYPERFRAMES_TEMPLATE / "index.html"
    if not _safe_is_file(template_path):
        raise PipelineError("Не найден шаблон motion-композиции HyperFrames.")
    try:
        html = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PipelineError("Не удалось прочитать шаблон motion-композиции HyperFrames.") from exc
    duration_token = 'data-duration="12"'
    if html.count(duration_token) < 2:
        raise PipelineError("Шаблон HyperFrames не содержит обязательные materialization-токены.")
    fingerprint = str(callback_plan.get("fingerprint") or hashlib.sha256(
        json.dumps(callback_plan, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest())
    project_id = f"broadcast-motion-{re.sub(r'[^a-f0-9]', '', fingerprint.lower())[:12] or 'local'}"
    project_dir = work_dir / "hyperframes_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    project_assets = project_dir / "assets"
    project_assets.mkdir(parents=True, exist_ok=True)
    source_assets = project_assets / "source_frames"
    for scene in callback_plan["scenes"]:
        raw_source = scene.get("source_frame")
        source_path = Path(raw_source) if raw_source else None
        if source_path is None or not _safe_is_file(source_path):
            scene["source_frame"] = ""
            scene["use_source_frame"] = False
            if scene.get("visual_variant") == "source":
                scene["visual_variant"] = KIND_VISUAL_VARIANTS[scene["kind"]]
            continue
        source_assets.mkdir(parents=True, exist_ok=True)
        suffix = source_path.suffix.lower() if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
        target_name = f"{scene['id']}{suffix}"
        shutil.copyfile(source_path, source_assets / target_name)
        scene["source_frame"] = f"assets/source_frames/{target_name}"
    _materialize_catalog_items(callback_plan, project_dir, visual_preset)
    for scene in callback_plan["scenes"]:
        scene["catalog_icon_svg"] = _icon_catalog().get(scene.get("catalog_icon", ""), "")
    html = html.replace(duration_token, f'data-duration="{float(duration):.3f}"')
    preset_stylesheet = 'href="presets/editorial_lime/style.css"'
    if html.count(preset_stylesheet) != 1:
        raise PipelineError("Шаблон HyperFrames не содержит stylesheet-токен визуального пресета.")
    html = html.replace(preset_stylesheet, 'href="preset/style.css"')
    serialized_plan = json.dumps(callback_plan, ensure_ascii=False, separators=(",", ":"))
    serialized_plan = serialized_plan.replace("<", "\\u003c")
    plan_pattern = re.compile(
        r'(<script\s+id="motion-plan"\s+type="application/json">)(.*?)(</script>)',
        re.DOTALL,
    )
    html, substitutions = plan_pattern.subn(
        lambda match: match.group(1) + serialized_plan + match.group(3), html, count=1,
    )
    if substitutions != 1:
        raise PipelineError("Шаблон HyperFrames не содержит script#motion-plan.")

    (project_dir / "index.html").write_text(html, encoding="utf-8")
    template_config = HYPERFRAMES_TEMPLATE / "hyperframes.json"
    if _safe_is_file(template_config):
        shutil.copyfile(template_config, project_dir / "hyperframes.json")
    for instruction_name in ("BRIEF.md", "frame.md"):
        instruction_path = (
            _preset_dir(visual_preset) / instruction_name
            if instruction_name == "frame.md"
            else HYPERFRAMES_TEMPLATE / instruction_name
        )
        if not _safe_is_file(instruction_path):
            raise PipelineError(f"В шаблоне HyperFrames отсутствует {instruction_name}.")
        shutil.copyfile(instruction_path, project_dir / instruction_name)
    preset_target = project_dir / "preset"
    shutil.copytree(_preset_dir(visual_preset), preset_target, dirs_exist_ok=True)
    template_gsap = HYPERFRAMES_TEMPLATE / "assets" / "gsap.min.js"
    if not _safe_is_file(template_gsap):
        raise PipelineError("В шаблоне HyperFrames не найден локальный assets/gsap.min.js.")
    shutil.copyfile(template_gsap, project_assets / "gsap.min.js")
    (project_dir / "meta.json").write_text(json.dumps({
        "id": project_id,
        "name": project_id,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (project_dir / "safe_areas.json").write_text(
        json.dumps(safe_areas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    catalog_selection = {
        "catalog_source": "https://hyperframes.heygen.com/catalog",
        "caption_style": callback_plan["caption_style"],
        "hook": {
            "title": callback_plan["hook_title"],
            "duration": callback_plan["hook_duration"],
            "motion": ["waterfall-entry", "gradient-text-sweep", "ambient-glow-bloom"],
        },
        "scenes": [{
            "id": scene["id"],
            "block": scene["catalog_block"],
            "motion": scene["catalog_motion"],
            "transition": scene["catalog_transition"],
            "icon": scene["catalog_icon"],
            "reason": scene["catalog_reason"],
            "tags": scene["catalog_tags"],
            "runtime": scene.get("catalog_runtime") or {},
        } for scene in callback_plan["scenes"]],
    }
    (project_dir / "catalog_selection.json").write_text(
        json.dumps(catalog_selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if _safe_is_file(CATALOG_MANIFEST):
        shutil.copyfile(CATALOG_MANIFEST, project_dir / "catalog.json")
    if _safe_is_file(ICON_CATALOG):
        shutil.copyfile(ICON_CATALOG, project_dir / "icon-catalog.json")
    (project_dir / "index.motion.json").write_text(json.dumps({
        "duration": round(float(duration), 3),
        "maxStaticSec": 2.4,
        "assertions": [
            {"kind": "appearsBy", "selector": "#hook-card", "bySec": min(0.12, duration / 3)},
            {
                "kind": "appearsBy",
                "selector": "#scene-s001",
                "bySec": min(float(duration), callback_plan["hook_duration"] + 0.15),
            },
            {"kind": "staysInFrame", "selector": "#scene-s001 .scene-inner"},
            {"kind": "keepsMoving", "withinSelector": "#motion-stage"},
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "project_dir": project_dir,
        "composition_id": "main",
        "visual_preset": visual_preset,
    }


def _verify_motion_video(path: Path, duration: float) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise PipelineError("HyperFrames не создал читаемый motion-видеофайл.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    actual_duration = frames / fps if fps > 0 else 0.0
    if (width, height) != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
        raise PipelineError(
            f"HyperFrames создал кадр {width}x{height}, ожидался {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}."
        )
    if not 29.0 <= fps <= 31.0:
        raise PipelineError(f"HyperFrames создал motion-видео с неожиданной частотой {fps:.3f} fps.")
    tolerance = max(0.15, 3.0 / OUTPUT_FPS)
    if abs(actual_duration - float(duration)) > tolerance:
        raise PipelineError(
            f"Длительность motion-видео {actual_duration:.3f} с не совпала с клипом {duration:.3f} с."
        )
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration": actual_duration,
    }


def render_hyperframes_visual(plan: dict, duration: float, safe_areas: list[dict],
                              work_dir: Path, callback: ProgressCallback,
                              visual_preset: str = DEFAULT_VISUAL_PRESET) -> Path:
    """Render an opaque, silent 1080x1920 motion background with the pinned CLI."""
    callback(64, "Собираю motion-композицию HyperFrames")
    materialized = materialize_hyperframes_project(
        plan, duration, safe_areas, work_dir, visual_preset=visual_preset,
    )
    raw_output = work_dir / "motion.raw.mp4"
    try:
        raw_output.unlink(missing_ok=True)
    except OSError as exc:
        raise PipelineError("Не удалось подготовить путь для motion-рендера HyperFrames.") from exc
    cli = resolve_hyperframes_command()
    ffmpeg_path, ffprobe_path = resolve_hyperframes_media_tools()
    env = runtime_paths.subprocess_env()
    env["HYPERFRAMES_FFMPEG_PATH"] = str(ffmpeg_path)
    env["HYPERFRAMES_FFPROBE_PATH"] = str(ffprobe_path)
    runtime_temp = work_dir / "hyperframes-tmp"
    runtime_temp.mkdir(parents=True, exist_ok=True)
    for name in ("TEMP", "TMP", "TMPDIR"):
        env[name] = str(runtime_temp.resolve())
    sample_times = sorted({
        round(min(max(0.05, float(duration) * ratio), max(0.05, float(duration) - 0.05)), 3)
        for ratio in (0.12, 0.5, 0.88)
    })
    check_command = cli + [
        "check", str(materialized["project_dir"]), "--json",
        "--at", ",".join(str(item) for item in sample_times),
        "--timeout", "30000",
    ]
    callback(66, "Проверяю композицию HyperFrames")
    def run_check() -> subprocess.CompletedProcess:
        try:
            return _run_hyperframes_process(
                check_command, cwd=HYPERFRAMES_TEMPLATE, env=env,
                timeout=150, attempts=3,
            )
        except subprocess.TimeoutExpired as exc:
            raise PipelineError("HyperFrames не успел проверить motion-композицию.") from exc

    checked = run_check()
    check_log = checked.stdout or ""
    if checked.returncode != 0:
        repair_plan, repaired_scene_ids = _layout_repair_plan(
            plan, _hyperframes_check_payload(check_log), duration,
        )
        if repair_plan is not None:
            callback(67, "Исправляю плотную компоновку HyperFrames")
            (work_dir / "visual_plan.layout-repair.json").write_text(
                json.dumps({
                    "reason": "automatic_layout_repair",
                    "scene_ids": repaired_scene_ids,
                    "plan": repair_plan,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            materialized = materialize_hyperframes_project(
                repair_plan, duration, safe_areas, work_dir, visual_preset=visual_preset,
            )
            checked = run_check()
            check_log += "\n\n--- automatic layout repair ---\n" + (checked.stdout or "")
    (work_dir / "hyperframes.check.log").write_text(check_log, encoding="utf-8")
    if checked.returncode != 0:
        raise PipelineError(
            "Проверка HyperFrames отклонила motion-композицию. " + (checked.stdout or "")[-1800:]
        )

    command = cli + [
        "render", str(materialized["project_dir"]),
        "--output", str(raw_output.resolve()),
        "--fps", str(OUTPUT_FPS),
        "--quality", "high",
        "--format", "mp4",
        "--strict",
    ]
    callback(69, "Рендерю motion-графику HyperFrames")
    timeout = max(240, int(math.ceil(float(duration) * 14)))
    try:
        result = _run_hyperframes_process(
            command, cwd=HYPERFRAMES_TEMPLATE, env=env,
            timeout=timeout, attempts=3,
        )
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("HyperFrames не успел отрендерить motion-графику за отведённое время.") from exc
    (work_dir / "hyperframes.log").write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0 or not _safe_is_file(raw_output):
        raise PipelineError(
            "HyperFrames не смог отрендерить motion-графику. " + (result.stdout or "")[-1800:]
        )

    # Even if a future template accidentally gains audio, the renderer contract
    # remains silent: retain only the rendered video stream without re-encoding.
    motion_path = work_dir / "motion.mp4"
    remux = subprocess.run([
        str(ffmpeg_path), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(raw_output), "-map", "0:v:0", "-an", "-c:v", "copy",
        "-movflags", "+faststart", str(motion_path),
    ], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if remux.returncode != 0:
        raise PipelineError(
            "Не удалось подготовить немой motion-слой HyperFrames. "
            + remux.stderr.decode("utf-8", errors="replace")[-1000:]
        )
    _verify_motion_video(motion_path, duration)
    callback(81, "Motion-графика HyperFrames готова")
    return motion_path
