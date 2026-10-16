---
colors:
  background: "#191919"
  surface: "#202022"
  surface_raised: "#2F2F2F"
  foreground: "#F6F6F6"
  muted: "#A3A3A8"
  accent: "#AEFBC3"
  border: "#F6F6F61A"
  utility_purple: "#7A52FF"
typography:
  display:
    family: "Onest Variable"
    weight: 700
    support_weight: 500
  editorial:
    family: "Onest Variable"
    weight: 500
  data:
    family: "ui-monospace"
    weight: 500
spacing:
  frame_x: 72
  frame_top: 72
  content_bottom: 648
  speaker_safe_left: 54
  speaker_safe_right: 54
  speaker_safe_bottom: 144
components:
  radius: 16
  small_radius: 10
  border: "2px solid #F6F6F61A"
  accent_border: "2px solid #AEFBC3"
  depth: "subtle inset glow; quiet external shadow only for source evidence"
---

# ITBooster-inspired broadcast motion system

## Overview

This preset adapts the current ITBooster website identity to portrait video without copying its page layout or branding. It uses the captured `#191919` charcoal canvas, `#F6F6F6` soft white, `#AEFBC3` mint accent and Onest typography. The video should feel like a confident developer ecosystem: functional, modular and technically precise, with enough scale and motion to remain cinematic.

## The frame

- Canvas is always 1080x1920.
- Keep primary meaning above 1280px; speaker slots occupy the lower corners when enabled.
- Subtitles are added after HyperFrames and must not be duplicated.
- Every scene has a dominant thesis plus one supporting visual relationship.
- Use asymmetrical card matrices, terminal labels and technical glyphs, not a centered web-page stack.

## Typography

- Headlines are Onest 700 at 72–118px with compact line height.
- Supporting copy is Onest 400–500 at 28–36px and no more than two short lines.
- Numbers are 140–210px, tabular where possible, with a grounded unit and context.
- Monospace is limited to navigation, indices and short code-like status labels.
- Mint may emphasize one exact word or one decisive value.

## Composition language

- Alternate strong typography, asymmetric feature-card structures, pathways, comparisons, technical networks and evidence frames.
- Use 14–16px rounded corners, visible low-opacity borders and subtle inset highlights.
- Keep mint scarce: one focal state per scene. Purple is allowed only as a tiny announcement/status accent.
- Source screenshots sit inside a dark rounded device-like frame and remain readable.
- Use technical orbit/ring forms as supporting geometry only when they clarify a network or process.

## Motion language

- Entrances take 0.35–0.75s and combine direction, scale and opacity intentionally.
- Card matrices assemble in a stagger; paths draw; values count; terminal labels reveal with a brief cursor beat.
- Ambient motion is restrained: a slow orbit, panel drift or light inset pulse after the entrance.
- All motion remains seek-safe in the single paused GSAP timeline.

## What not to do

- Do not reproduce the ITBooster logo, navigation, copy or promotional claims.
- Do not use generic cyan/purple neon, full-screen gradients or glossy glassmorphism.
- Do not fill every panel with mint; dark surfaces must dominate.
- Do not repeat the same card grid or motion recipe in adjacent scenes.
- Do not put important content inside either enabled speaker safe area.
