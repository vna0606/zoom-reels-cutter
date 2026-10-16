from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

from . import runtime_paths
from .pipeline import (
    FFMPEG,
    ROOT,
    PipelineError,
    ProgressCallback,
    detect_faces,
    media_info,
    verify_output,
    write_ass,
)
from .hyperframes_visual import (
    attach_presentation_frames,
    build_visual_plan,
    normalize_visual_preset,
    render_hyperframes_visual,
    visual_preset_accent,
    without_photographic_visuals,
)

import cv2
import numpy as np

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
PRESENTATION_HEIGHT = 608
SPEAKER_HEIGHT = 1312
SPEAKER_ASPECT = OUTPUT_WIDTH / SPEAKER_HEIGHT
PRESENTATION_FACE_PENALTY = 1.25
SPEAKER_OVERLAY_X = 54
SPEAKER_OVERLAY_BOTTOM = 144
SPEAKER_OVERLAY_SCALE = 1.69
SPEAKER_OVERLAY_MAX_WIDTH = 608
SPEAKER_PAIR_MAX_WIDTH = 432
SPEAKER_BORDER_COLOR = "0xC7F54A"
SPEAKER_BORDER_WIDTH = 6
ALLOWED_PLAYBACK_SPEEDS = {1.0, 1.05, 1.1, 1.15, 1.2, 1.25}
ALLOWED_SPEAKER_COUNTS = {1, 2}

# Zoom recordings are commonly either screen-share + a participant strip or a
# side-by-side camera/monitor view.  Named presets make the selected crop easy
# to understand in broadcast_layout.json and keep the renderer deterministic.
PRESENTATION_PRESETS = (
    ("full", (0.00, 0.00, 1.00, 1.00)),
    ("left_78", (0.00, 0.00, 0.78, 1.00)),
    ("right_78", (0.22, 0.00, 0.78, 1.00)),
    ("left_66", (0.00, 0.00, 2.0 / 3.0, 1.00)),
    ("right_66", (1.0 / 3.0, 0.00, 2.0 / 3.0, 1.00)),
    ("top_78", (0.00, 0.00, 1.00, 0.78)),
    ("bottom_78", (0.00, 0.22, 1.00, 0.78)),
)


def even(value: float, minimum: int = 2) -> int:
    """Return a positive even integer accepted by H.264 crop filters."""
    return max(minimum, int(round(value)) // 2 * 2)


def normalize_playback_speed(value) -> float:
    try:
        speed = round(float(value), 2)
    except (TypeError, ValueError):
        speed = 1.0
    if speed not in ALLOWED_PLAYBACK_SPEEDS:
        raise PipelineError("Скорость ролика должна быть от 1.0× до 1.25× с шагом 0.05.")
    return speed


def normalize_speaker_count(value) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = 2
    if count not in ALLOWED_SPEAKER_COUNTS:
        raise PipelineError("Количество отображаемых спикеров должно быть 1 или 2.")
    return count


def clamp_crop(x: float, y: float, width: float, height: float,
               source_width: int, source_height: int) -> dict:
    crop_width = min(source_width - source_width % 2, even(width))
    crop_height = min(source_height - source_height % 2, even(height))
    crop_width = max(2, crop_width)
    crop_height = max(2, crop_height)
    crop_x = even(max(0, min(source_width - crop_width, x)), minimum=0)
    crop_y = even(max(0, min(source_height - crop_height, y)), minimum=0)
    return {"x": crop_x, "y": crop_y, "width": crop_width, "height": crop_height}


def preset_crop(name: str, source_width: int, source_height: int) -> dict:
    normalized = dict(PRESENTATION_PRESETS).get(name, dict(PRESENTATION_PRESETS)["full"])
    x, y, width, height = normalized
    return clamp_crop(
        source_width * x,
        source_height * y,
        source_width * width,
        source_height * height,
        source_width,
        source_height,
    )


def speaker_crop_from_face(face: tuple[int, int, int, int] | list[int],
                           source_width: int, source_height: int) -> dict:
    x, y, width, height = [float(value) for value in face]
    face_based_height = height * 2.4
    tile_floor_height = source_height * 0.17
    crop_height = min(float(source_height), max(face_based_height, tile_floor_height))
    crop_width = crop_height * SPEAKER_ASPECT
    if crop_width > source_width:
        crop_width = float(source_width)
        crop_height = crop_width / SPEAKER_ASPECT
    center_x = x + width / 2
    # A tiny detected face uses the tile-size floor; anchor it slightly lower
    # in that crop so the crop remains within its Zoom row. Larger faces keep
    # the center at one third, leaving natural shoulder room.
    center_y = y + height / 2
    face_center_ratio = 0.40 if tile_floor_height > face_based_height else 0.33
    return clamp_crop(
        center_x - crop_width / 2,
        center_y - crop_height * face_center_ratio,
        crop_width,
        crop_height,
        source_width,
        source_height,
    )


def fallback_speaker_crop(source_width: int, source_height: int) -> dict:
    crop_height = float(source_height)
    crop_width = crop_height * SPEAKER_ASPECT
    if crop_width > source_width:
        crop_width = float(source_width)
        crop_height = crop_width / SPEAKER_ASPECT
    return clamp_crop(
        (source_width - crop_width) / 2,
        (source_height - crop_height) / 2,
        crop_width,
        crop_height,
        source_width,
        source_height,
    )


def _intersection_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return float(max(0, right - left) * max(0, bottom - top))


def score_presentation_preset(frame: np.ndarray, faces: list[tuple[int, int, int, int]],
                              preset_name: str) -> float:
    """Score a stable screen-share crop: detailed content, few detected faces."""
    source_height, source_width = frame.shape[:2]
    crop = preset_crop(preset_name, source_width, source_height)
    x, y, width, height = crop["x"], crop["y"], crop["width"], crop["height"]
    roi = frame[y:y + height, x:x + width]
    if roi.size == 0:
        return -1000.0
    scale = min(1.0, 360.0 / max(roi.shape[:2]))
    if scale < 1:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edge_density = float(np.count_nonzero(cv2.Canny(gray, 70, 150))) / float(gray.size)
    sharpness = math.log1p(float(cv2.Laplacian(gray, cv2.CV_32F).var())) / 7.0
    contrast = min(1.0, float(gray.std()) / 64.0)
    crop_tuple = (x, y, width, height)
    face_penalty = 0.0
    for face in faces:
        face_area = max(1.0, float(face[2] * face[3]))
        face_penalty += _intersection_area(crop_tuple, face) / face_area
    area_ratio = width * height / float(source_width * source_height)
    return (
        sharpness + edge_density * 4.5 + contrast * 0.25 + area_ratio * 0.45
        - face_penalty * PRESENTATION_FACE_PENALTY
    )


def trim_presentation_letterbox(frame: np.ndarray, crop: dict) -> dict:
    """Remove stable dark horizontal Zoom bars while retaining the selected preset."""
    source_height, source_width = frame.shape[:2]
    safe = clamp_crop(
        crop["x"], crop["y"], crop["width"], crop["height"],
        source_width, source_height,
    )
    x, y, width, height = safe["x"], safe["y"], safe["width"], safe["height"]
    roi = frame[y:y + height, x:x + width]
    if roi.size == 0 or height < 24:
        return safe
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # The 90th percentile distinguishes a full-width black Zoom bar from a
    # naturally dark slide, which still tends to contain text or UI details.
    row_signal = np.percentile(gray, 90, axis=1)
    active = row_signal > 24
    stable = np.convolve(active.astype(np.uint8), np.ones(7, dtype=np.uint8), mode="same") >= 4
    indices = np.flatnonzero(stable)
    if not len(indices):
        return safe
    top = max(0, int(indices[0]))
    bottom = min(height, int(indices[-1]) + 1)
    retained = bottom - top
    if retained < height * 0.35 or height - retained < height * 0.04:
        return safe
    return clamp_crop(x, y + top, width, retained, source_width, source_height)


def top_anchored_presentation_crop(crop: dict, source_width: int,
                                   source_height: int) -> dict:
    """Fit a crop to the output panel, dropping vertical excess from the bottom."""
    safe = clamp_crop(
        crop["x"], crop["y"], crop["width"], crop["height"],
        source_width, source_height,
    )
    target_aspect = OUTPUT_WIDTH / PRESENTATION_HEIGHT
    current_aspect = safe["width"] / safe["height"]
    if current_aspect < target_aspect:
        # The source is too tall: keep the detected top edge and discard the
        # lower Zoom letterbox/content excess.
        return clamp_crop(
            safe["x"], safe["y"], safe["width"], safe["width"] / target_aspect,
            source_width, source_height,
        )
    fitted_width = safe["height"] * target_aspect
    return clamp_crop(
        safe["x"] + (safe["width"] - fitted_width) / 2,
        safe["y"], fitted_width, safe["height"], source_width, source_height,
    )


def select_presentation_preset(scores: dict[str, float]) -> str:
    """Prefer a nearly tied side share over a mixed vertical Zoom region."""
    best = max(scores, key=scores.get)
    side_names = [
        name for name in ("left_66", "right_66", "left_78", "right_78")
        if name in scores
    ]
    if best in {"full", "top_78", "bottom_78"} and side_names:
        best_side = max(side_names, key=scores.get)
        if scores[best] - scores[best_side] <= 0.18:
            return best_side
    return best


def _scaled_for_detection(frame: np.ndarray, maximum_width: int = 960) -> tuple[np.ndarray, float]:
    width = frame.shape[1]
    if width <= maximum_width:
        return frame, 1.0
    scale = maximum_width / float(width)
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def _detect_faces(frame: np.ndarray, frontal, profile) -> list[tuple[int, int, int, int]]:
    smaller, scale = _scaled_for_detection(frame)
    min_face = max(26, int(smaller.shape[1] * 0.026))
    found = detect_faces(smaller, frontal, profile, min_face)
    if scale == 1:
        return found
    return [
        (round(x / scale), round(y / scale), round(width / scale), round(height / scale))
        for x, y, width, height in found
    ]


def _mouth_motion(first: np.ndarray, second: np.ndarray,
                  face: tuple[int, int, int, int]) -> float:
    x, y, width, height = face
    x = max(0, x)
    y = max(0, y)
    width = min(width, first.shape[1] - x, second.shape[1] - x)
    height = min(height, first.shape[0] - y, second.shape[0] - y)
    if width < 8 or height < 8:
        return 0.0
    first_face = cv2.cvtColor(first[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    second_face = cv2.cvtColor(second[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(first_face, second_face)
    mouth = float(diff[int(height * 0.52):int(height * 0.93), :].mean())
    upper = float(diff[int(height * 0.08):int(height * 0.42), :].mean())
    return max(0.0, mouth - upper * 0.45)


def _zoom_border_score(frame: np.ndarray, face: tuple[int, int, int, int]) -> float:
    """Estimate Zoom's green active-speaker outline around a participant tile."""
    x, y, width, height = face
    margin_x, margin_y = int(width * 0.8), int(height * 0.8)
    left, top = max(0, x - margin_x), max(0, y - margin_y)
    right = min(frame.shape[1], x + width + margin_x)
    bottom = min(frame.shape[0], y + height + margin_y)
    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 70, 55), (95, 255, 255))
    thickness = max(2, int(min(green.shape[:2]) * 0.045))
    side_ratios = [
        float(np.count_nonzero(green[:thickness, :])) / max(1, green[:thickness, :].size),
        float(np.count_nonzero(green[-thickness:, :])) / max(1, green[-thickness:, :].size),
        float(np.count_nonzero(green[:, :thickness])) / max(1, green[:, :thickness].size),
        float(np.count_nonzero(green[:, -thickness:])) / max(1, green[:, -thickness:].size),
    ]
    # A Zoom outline is rectangular and therefore appears on more than one
    # side. Requiring the second strongest side avoids treating a green shirt
    # or a green patch inside a slide as an active-speaker border.
    ordered = sorted(side_ratios, reverse=True)
    return min(1.0, ordered[1] * 5.0 + float(np.mean(side_ratios)) * 2.0)


def _rank_speaker_candidates(first: np.ndarray, second: np.ndarray,
                             faces: list[tuple[int, int, int, int]]) -> list[dict]:
    if not faces:
        return []
    largest = max(1.0, float(max(width * height for _, _, width, height in faces)))
    candidates = []
    for face in faces:
        area_score = face[2] * face[3] / largest
        motion = min(1.0, _mouth_motion(first, second, face) / 18.0)
        border = _zoom_border_score(first, face)
        score = motion * 4.0 + border * 3.0 + area_score * 2.0
        candidates.append({
            "face": list(face),
            "score": round(score, 4),
            "mouth_motion": round(motion, 4),
            "zoom_border": round(border, 4),
            "relative_area": round(area_score, 4),
        })
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _face_outside_crop(face: list[int], crop: dict) -> bool:
    overlap = _intersection_area(
        (face[0], face[1], face[2], face[3]),
        (crop["x"], crop["y"], crop["width"], crop["height"]),
    )
    return overlap / max(1.0, float(face[2] * face[3])) < 0.35


def _face_distance(first: list[int] | tuple[int, int, int, int],
                   second: list[int] | tuple[int, int, int, int],
                   source_width: int, source_height: int) -> float:
    first_x, first_y = first[0] + first[2] / 2, first[1] + first[3] / 2
    second_x, second_y = second[0] + second[2] / 2, second[1] + second[3] / 2
    return math.hypot((first_x - second_x) / source_width, (first_y - second_y) / source_height)


def _speech_intervals(transcript: dict | None) -> list[tuple[float, float]]:
    words = (transcript or {}).get("words") or []
    intervals: list[list[float]] = []
    for word in words:
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if intervals and start - intervals[-1][1] <= 0.75:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    return [(max(0.0, start - 0.18), end + 0.28) for start, end in intervals]


def _speech_active(timestamp: float, intervals: list[tuple[float, float]]) -> bool:
    return not intervals or any(start <= timestamp <= end for start, end in intervals)


def _eligible_speaker_candidates(candidates: list[dict], presentation_crop: dict,
                                 presentation_name: str, source_width: int,
                                 source_height: int) -> tuple[list[dict], bool]:
    if presentation_name == "full":
        outside = []
        for item in candidates:
            face = item["face"]
            center_x = (face[0] + face[2] / 2) / source_width
            center_y = (face[1] + face[3] / 2) / source_height
            frame_area = face[2] * face[3] / float(source_width * source_height)
            near_edge = center_x < 0.23 or center_x > 0.77 or center_y < 0.20 or center_y > 0.80
            if item.get("zoom_border", 0.0) >= 0.08 or (near_edge and frame_area <= 0.08):
                outside.append(item)
    else:
        outside = [item for item in candidates if _face_outside_crop(item["face"], presentation_crop)]
    return (outside or candidates), not bool(outside)


def _select_speaker(candidates: list[dict], presentation_crop: dict,
                    presentation_name: str, previous_face: list[int] | None,
                    speech_active: bool, source_width: int,
                    source_height: int) -> tuple[list[int] | None, float, bool]:
    if not candidates:
        return (None if speech_active else previous_face), 0.0, False
    ranked, region_fallback = _eligible_speaker_candidates(
        candidates, presentation_crop, presentation_name, source_width, source_height,
    )
    best = ranked[0]
    runner_up = ranked[1]["score"] if len(ranked) > 1 else 0.0
    confidence = max(0.0, min(1.0, (best["score"] - runner_up) / max(best["score"], 0.25)))
    strong_change_signal = (
        best.get("zoom_border", 0.0) >= 0.06
        or (best.get("mouth_motion", 0.0) >= 0.10 and confidence >= 0.05)
    )
    credible_active_speaker = strong_change_signal or confidence >= 0.18
    if speech_active and not credible_active_speaker:
        # Showing no face is preferable to keeping a listener on screen when
        # neither Zoom nor mouth motion confirms who owns the current speech.
        return None, confidence, region_fallback
    if previous_face is not None and not speech_active:
        nearest = min(
            ranked,
            key=lambda item: _face_distance(
                item["face"], previous_face, source_width, source_height,
            ),
        )
        if _face_distance(nearest["face"], previous_face, source_width, source_height) <= 0.18:
            return nearest["face"], confidence, region_fallback
        return previous_face, 0.0, region_fallback
    return best["face"], confidence, region_fallback


def _cluster_observations(observations: list[dict], source_width: int, source_height: int) -> tuple[list[dict], list[int | None]]:
    clusters: list[dict] = []
    raw_ids: list[int | None] = []
    for observation in observations:
        active_face = observation.get("face")
        faces = []
        if active_face:
            faces.append(active_face)
        for face in observation.get("participant_faces") or []:
            if face and face not in faces:
                faces.append(face)
        active_id = None
        for face in faces:
            center_x = face[0] + face[2] / 2
            center_y = face[1] + face[3] / 2
            best_id, best_distance = None, float("inf")
            for index, cluster in enumerate(clusters):
                distance = math.hypot(
                    (center_x - cluster["center_x"]) / source_width,
                    (center_y - cluster["center_y"]) / source_height,
                )
                if distance < best_distance:
                    best_id, best_distance = index, distance
            if best_id is None or best_distance > 0.16:
                clusters.append({"center_x": center_x, "center_y": center_y, "faces": [face]})
                face_id = len(clusters) - 1
            else:
                cluster = clusters[best_id]
                cluster["faces"].append(face)
                count = len(cluster["faces"])
                cluster["center_x"] += (center_x - cluster["center_x"]) / count
                cluster["center_y"] += (center_y - cluster["center_y"]) / count
                face_id = best_id
            if active_face and face == active_face:
                active_id = face_id
        raw_ids.append(active_id)
    return clusters, raw_ids


def smooth_speaker_ids(raw_ids: list[int | None], minimum_run: int = 2) -> list[int | None]:
    """Remove one-sample false switches and debounce changes between tiles."""
    if not raw_ids:
        return []
    filled: list[int | None] = []
    previous = next((value for value in raw_ids if value is not None), None)
    for value in raw_ids:
        if value is None:
            value = previous
        else:
            previous = value
        filled.append(value)
    for index in range(len(filled) - 2, -1, -1):
        if filled[index] is None:
            filled[index] = filled[index + 1]
    windowed = []
    for index, value in enumerate(filled):
        window = [item for item in filled[max(0, index - 1):index + 2] if item is not None]
        if not window:
            windowed.append(value)
            continue
        counts = Counter(window)
        windowed.append(max(counts, key=lambda item: (counts[item], item == value)))
    active = windowed[0]
    pending = None
    pending_count = 0
    result = []
    for value in windowed:
        if value is None or value == active:
            pending, pending_count = None, 0
        elif value == pending:
            pending_count += 1
            if pending_count >= minimum_run:
                active = value
                # The switch became trustworthy on this sample, but it began
                # on the previous one. Move the cut back to the first sample
                # in the stable run instead of leaving the old speaker on
                # screen for an extra sampling interval.
                for offset in range(1, minimum_run):
                    if len(result) >= offset:
                        result[-offset] = active
                pending, pending_count = None, 0
        else:
            pending, pending_count = value, 1
        result.append(active)
    return result


def smooth_speaker_timeline(raw_ids: list[int | None], timestamps: list[float],
                            minimum_seconds: float = 0.65,
                            strong_switches: list[bool] | None = None) -> list[int | None]:
    """Debounce speaker changes by elapsed time and keep the original cut point."""
    if not raw_ids:
        return []
    filled: list[int | None] = []
    previous = next((value for value in raw_ids if value is not None), None)
    for value in raw_ids:
        if value is None:
            value = previous
        else:
            previous = value
        filled.append(value)
    for index in range(len(filled) - 2, -1, -1):
        if filled[index] is None:
            filled[index] = filled[index + 1]
    active = filled[0]
    pending = None
    pending_start = 0.0
    pending_index = 0
    result: list[int | None] = []
    for index, value in enumerate(filled):
        timestamp = float(timestamps[index]) if index < len(timestamps) else float(index)
        strong_switch = bool(strong_switches[index]) if strong_switches and index < len(strong_switches) else False
        if value is None or value == active:
            pending = None
        elif strong_switch:
            active = value
            pending = None
        elif value != pending:
            pending = value
            pending_start = timestamp
            pending_index = index
        elif timestamp - pending_start >= minimum_seconds:
            active = value
            for previous_index in range(pending_index, len(result)):
                result[previous_index] = active
            pending = None
        result.append(active)
    return result


def _median_face(cluster: dict) -> tuple[int, int, int, int]:
    values = np.asarray(cluster["faces"], dtype=np.float32)
    medians = np.median(values, axis=0)
    return tuple(int(round(value)) for value in medians)


def build_speaker_segments(observations: list[dict], source_width: int,
                           source_height: int, duration: float,
                           grid: dict | None = None) -> tuple[list[dict], int]:
    clusters, raw_ids = _cluster_observations(observations, source_width, source_height)
    active_ids = smooth_speaker_timeline(
        raw_ids, [float(item.get("time", index)) for index, item in enumerate(observations)],
        strong_switches=[bool(item.get("strong_switch", False)) for item in observations],
    )
    if not clusters or not any(value is not None for value in active_ids):
        return [{
            "start": 0.0,
            "end": round(duration, 3),
            "speaker_id": None,
            "crop": fallback_speaker_crop(source_width, source_height),
            "confidence": 0.0,
            "fallback": "no_face_detected",
        }], 0
    cluster_crops = {}
    for index, cluster in enumerate(clusters):
        tile = grid_tile_for_face(grid, _median_face(cluster))
        cluster_crops[index] = (
            speaker_crop_in_tile(cluster["faces"], tile, source_width, source_height)
            if tile else speaker_crop_from_face(_median_face(cluster), source_width, source_height)
        )
    segments = []
    current_id = active_ids[0]
    segment_start = 0.0
    confidences = []
    for index, (observation, speaker_id) in enumerate(zip(observations, active_ids)):
        if speaker_id == current_id:
            confidences.append(float(observation.get("confidence", 0.0)))
            continue
        boundary = float(observation["time"])
        if index:
            boundary = (float(observations[index - 1]["time"]) + boundary) / 2
        segments.append({
            "start": round(segment_start, 3),
            "end": round(boundary, 3),
            "speaker_id": current_id,
            "crop": cluster_crops.get(current_id, fallback_speaker_crop(source_width, source_height)),
            "confidence": round(float(np.mean(confidences)) if confidences else 0.0, 3),
            "fallback": None if current_id is not None else "previous_or_center",
        })
        segment_start = boundary
        current_id = speaker_id
        confidences = [float(observation.get("confidence", 0.0))]
    segments.append({
        "start": round(segment_start, 3),
        "end": round(duration, 3),
        "speaker_id": current_id,
        "crop": cluster_crops.get(current_id, fallback_speaker_crop(source_width, source_height)),
        "confidence": round(float(np.mean(confidences)) if confidences else 0.0, 3),
        "fallback": None if current_id is not None else "previous_or_center",
    })
    segments = [item for item in segments if item["end"] - item["start"] >= 0.02]
    reliable_cluster_ids = sorted(
        (index for index, cluster in enumerate(clusters) if len(cluster["faces"]) >= 2),
        key=lambda index: len(clusters[index]["faces"]),
        reverse=True,
    )
    active_counts = Counter(value for value in active_ids if value is not None)
    stable_candidates = reliable_cluster_ids or list(active_counts)
    primary_id = max(
        stable_candidates,
        key=lambda speaker_id: (
            active_counts.get(speaker_id, 0),
            len(clusters[speaker_id]["faces"]),
            -speaker_id,
        ),
        default=None,
    )
    secondary_id = next(
        (speaker_id for speaker_id in reliable_cluster_ids if speaker_id != primary_id),
        None,
    )
    for segment in segments:
        if primary_id is not None:
            segment["left_speaker_id"] = primary_id
            segment["left_crop"] = cluster_crops[primary_id]
        if secondary_id is not None:
            segment["right_speaker_id"] = secondary_id
            segment["right_crop"] = cluster_crops[secondary_id]
    return segments, len(clusters)


# ---------------------------------------------------------------------------
# Zoom gallery tiles: keep speaker crops inside the participant's own video.
# ---------------------------------------------------------------------------
ZOOM_TILE_ASPECT = 16.0 / 9.0
TILE_INSET_SIDE = 0.016     # skip Zoom's active-speaker outline / neighbour bleed
TILE_INSET_TOP = 0.02
TILE_INSET_BOTTOM = 0.075   # skip the name bar that Zoom draws at the tile bottom
SUBTITLE_GAP = 40           # px between subtitle block bottom and speaker tiles
SUBTITLE_BLOCK_HEIGHT = 130


def content_bbox_from_projections(row_active: np.ndarray, col_active: np.ndarray,
                                  scale: float = 1.0) -> dict | None:
    """Bounding box of the non-black gallery area from boolean row/column masks."""
    rows = np.flatnonzero(row_active)
    cols = np.flatnonzero(col_active)
    if not len(rows) or not len(cols):
        return None
    x, y = int(cols[0] / scale), int(rows[0] / scale)
    right, bottom = int((cols[-1] + 1) / scale), int((rows[-1] + 1) / scale)
    return {"x": x, "y": y, "width": right - x, "height": bottom - y}


def detect_zoom_grid(bbox: dict | None,
                     faces: list[tuple[int, int, int, int] | list[int]]) -> dict | None:
    """Infer the 16:9 tile grid of a Zoom gallery from its content box and faces.

    Returns the grid with the fewest tiles in which every face sits fully inside
    one tile and no tile holds two faces, or None if the layout is not a plain
    gallery grid (speaker view, share with thumbnails, centred partial rows...).
    """
    if not bbox or bbox["width"] < 64 or bbox["height"] < 64 or not faces:
        return None
    best = None
    for cols in range(1, 7):
        for rows in range(1, 7):
            tile_w = bbox["width"] / cols
            tile_h = bbox["height"] / rows
            if abs(tile_w / tile_h / ZOOM_TILE_ASPECT - 1.0) > 0.12:
                continue
            used: set[tuple[int, int]] = set()
            ok = True
            for x, y, w, h in faces:
                cx, cy = x + w / 2 - bbox["x"], y + h / 2 - bbox["y"]
                col, row = int(cx // tile_w), int(cy // tile_h)
                if not (0 <= col < cols and 0 <= row < rows) or (col, row) in used:
                    ok = False
                    break
                slack = 0.08 * max(w, h)
                if (x - bbox["x"] < col * tile_w - slack or y - bbox["y"] < row * tile_h - slack
                        or x + w - bbox["x"] > (col + 1) * tile_w + slack
                        or y + h - bbox["y"] > (row + 1) * tile_h + slack):
                    ok = False
                    break
                used.add((col, row))
            if ok and (best is None or cols * rows < best["cols"] * best["rows"]):
                best = {"cols": cols, "rows": rows, "bbox": dict(bbox),
                        "tile_width": tile_w, "tile_height": tile_h}
    return best


def grid_tile_for_face(grid: dict | None, face: tuple[int, int, int, int] | list[int]) -> dict | None:
    """Raw (outer) grid tile rectangle that contains the face centre."""
    if not grid:
        return None
    bbox = grid["bbox"]
    cx = face[0] + face[2] / 2 - bbox["x"]
    cy = face[1] + face[3] / 2 - bbox["y"]
    col, row = int(cx // grid["tile_width"]), int(cy // grid["tile_height"])
    if not (0 <= col < grid["cols"] and 0 <= row < grid["rows"]):
        return None
    return {
        "x": bbox["x"] + col * grid["tile_width"], "y": bbox["y"] + row * grid["tile_height"],
        "width": grid["tile_width"], "height": grid["tile_height"],
    }


def inset_tile(tile: dict) -> dict:
    """Shrink a tile so the crop excludes its outline, edge bleed and name bar."""
    inset_x = tile["width"] * TILE_INSET_SIDE
    top = tile["height"] * TILE_INSET_TOP
    bottom = tile["height"] * TILE_INSET_BOTTOM
    return {
        "x": tile["x"] + inset_x, "y": tile["y"] + top,
        "width": tile["width"] - 2 * inset_x, "height": tile["height"] - top - bottom,
    }


def speaker_crop_in_tile(faces: list, tile: dict, source_width: int, source_height: int) -> dict:
    """Speaker crop that stays inside the (inset) tile and keeps every observed
    face position (5-95 percentile of the clip) fully inside with margin."""
    values = np.asarray(faces, dtype=np.float32)
    median = tuple(float(v) for v in np.median(values, axis=0))
    base = speaker_crop_from_face(median, source_width, source_height)
    inner = inset_tile(tile)
    face_w, face_h = median[2], median[3]
    left = float(np.percentile(values[:, 0], 5)) - face_w * 0.45
    right = float(np.percentile(values[:, 0] + values[:, 2], 95)) + face_w * 0.45
    top = float(np.percentile(values[:, 1], 5)) - face_h * 0.6
    bottom = float(np.percentile(values[:, 1] + values[:, 3], 95)) + face_h * 0.5
    x1 = max(inner["x"], min(base["x"], left))
    x2 = min(inner["x"] + inner["width"], max(base["x"] + base["width"], right))
    y1 = max(inner["y"], min(base["y"], top))
    y2 = min(inner["y"] + inner["height"], max(base["y"] + base["height"], bottom))
    return clamp_crop(x1, y1, x2 - x1, y2 - y1, source_width, source_height)


def assess_presentation(crop: dict, camera_rects: list[dict], edge_density: float,
                        mean_luma: float, min_edge_density: float = 0.012,
                        min_luma: float = 28.0, max_camera_cover: float = 0.15) -> dict:
    """Decide whether the chosen presentation crop is a real screen share.

    It is not when it overlaps camera tiles (speaker tiles / faces) noticeably
    or is nearly featureless (dark, no text/edges)."""
    area = max(1.0, float(crop["width"] * crop["height"]))
    crop_box = (crop["x"], crop["y"], crop["width"], crop["height"])
    covered = 0.0
    for rect in camera_rects:
        rect_box = (rect["x"], rect["y"], rect["width"], rect["height"])
        inter = _intersection_area(crop_box, rect_box)
        # Only tiles that are mostly inside the crop count: a thumbnail strip
        # merely touching a real share must not invalidate it.
        if inter >= 0.5 * rect["width"] * rect["height"]:
            covered += inter
    cover = covered / area
    reasons = []
    if cover >= max_camera_cover:
        reasons.append("camera_tiles_in_crop")
    if edge_density < min_edge_density:
        reasons.append("low_detail")
    if mean_luma < min_luma:
        reasons.append("dark")
    return {"valid": not reasons, "reasons": reasons, "camera_cover": round(cover, 3),
            "edge_density": round(edge_density, 4), "mean_luma": round(mean_luma, 1)}


def subtitle_safe_margin(safe_areas: list[dict], output_height: int = OUTPUT_HEIGHT) -> int:
    """ASS MarginV (from the bottom) that keeps captions above all speaker tiles."""
    tops = [area["bottom"] + area["height"] for area in safe_areas
            if area.get("kind", "").startswith("speaker") and area.get("height")]
    tile_top_offset = max(tops, default=SPEAKER_OVERLAY_BOTTOM)
    return int(min(output_height - SUBTITLE_BLOCK_HEIGHT, tile_top_offset + SUBTITLE_GAP))


def apply_subtitle_margin(ass_path: Path, margin_v: int) -> None:
    if not ass_path.is_file():
        return
    text = ass_path.read_text(encoding="utf-8")
    text = re.sub(r"(,2,\d+,\d+,)\d+(,\d+\s*\n)", lambda m: f"{m.group(1)}{margin_v}{m.group(2)}",
                  text, count=1)
    ass_path.write_text(text, encoding="utf-8")


def presentation_roi_stats(frame: np.ndarray, crop: dict) -> tuple[float, float]:
    x, y, w, h = crop["x"], crop["y"], crop["width"], crop["height"]
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = float(np.count_nonzero(cv2.Canny(gray, 70, 150))) / float(gray.size)
    return edges, float(gray.mean())


def analyze_broadcast_layout(video_path: Path, info: dict,
                             callback: ProgressCallback, transcript: dict | None = None) -> dict:
    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError("Не удалось открыть эфир для анализа раскладки.")
    duration = float(info["duration"])
    # Sub-second sampling keeps a newly active Zoom tile from remaining stale
    # on screen. Debouncing still rejects isolated face-detector jumps.
    interval = max(0.30, min(0.48, duration / 100.0))
    sample_times = []
    cursor = min(0.25, max(0.0, duration / 2))
    while cursor < duration:
        sample_times.append(cursor)
        cursor += interval
    if not sample_times:
        sample_times = [0.0]
    observations = []
    presentation_scores = {name: [] for name, _ in PRESENTATION_PRESETS}
    presentation_trim_ratios = {name: [] for name, _ in PRESENTATION_PRESETS}
    presentation_detail = {name: [] for name, _ in PRESENTATION_PRESETS}
    row_active = col_active = None
    detection_scale = 1.0
    callback(18, "Определяю экран и участников эфира")
    for index, timestamp in enumerate(sample_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, first = cap.read()
        if not ok:
            continue
        second_time = min(max(0.0, duration - 0.001), timestamp + 0.12)
        cap.set(cv2.CAP_PROP_POS_MSEC, second_time * 1000)
        second_ok, second = cap.read()
        if not second_ok or second.shape != first.shape:
            second = first
        faces = _detect_faces(first, frontal, profile)
        observations.append({
            "time": round(timestamp, 3),
            "face": None,
            "confidence": 0.0,
            "faces_detected": len(faces),
            "speech_active": False,
            "speaker_region_fallback": False,
            "_candidates": _rank_speaker_candidates(first, second, faces),
        })
        scoring_frame, scale = _scaled_for_detection(first, maximum_width=640)
        scoring_faces = [
            (round(x * scale), round(y * scale), round(width * scale), round(height * scale))
            for x, y, width, height in faces
        ]
        detection_scale = scale
        luma = cv2.cvtColor(scoring_frame, cv2.COLOR_BGR2GRAY)
        rows_now = (luma > 12).mean(axis=1) > 0.02
        cols_now = (luma > 12).mean(axis=0) > 0.02
        row_active = rows_now if row_active is None else (row_active | rows_now)
        col_active = cols_now if col_active is None else (col_active | cols_now)
        for name in presentation_scores:
            presentation_scores[name].append(score_presentation_preset(scoring_frame, scoring_faces, name))
            base_crop = preset_crop(name, scoring_frame.shape[1], scoring_frame.shape[0])
            presentation_detail[name].append(presentation_roi_stats(scoring_frame, base_crop))
            content_crop = trim_presentation_letterbox(scoring_frame, base_crop)
            presentation_trim_ratios[name].append((
                (content_crop["y"] - base_crop["y"]) / max(1, base_crop["height"]),
                (content_crop["y"] + content_crop["height"] - base_crop["y"])
                / max(1, base_crop["height"]),
            ))
        progress = 18 + int((index + 1) / len(sample_times) * 36)
        callback(progress, "Слежу за активным спикером и демонстрацией")
    cap.release()
    average_scores = {
        name: round(float(np.mean(values)), 4) if values else -1000.0
        for name, values in presentation_scores.items()
    }
    presentation_name = select_presentation_preset(average_scores)
    preset_source_crop = preset_crop(presentation_name, info["width"], info["height"])
    trim_samples = presentation_trim_ratios.get(presentation_name) or [(0.0, 1.0)]
    trim_top = max(0.0, min(1.0, float(np.median([item[0] for item in trim_samples]))))
    trim_bottom = max(trim_top, min(1.0, float(np.median([item[1] for item in trim_samples]))))
    presentation_crop = clamp_crop(
        preset_source_crop["x"],
        preset_source_crop["y"] + preset_source_crop["height"] * trim_top,
        preset_source_crop["width"],
        preset_source_crop["height"] * (trim_bottom - trim_top),
        info["width"],
        info["height"],
    )
    presentation_crop = top_anchored_presentation_crop(
        presentation_crop, info["width"], info["height"],
    )
    speech_intervals = _speech_intervals(transcript)
    previous_face = None
    for observation in observations:
        is_speech = _speech_active(float(observation["time"]), speech_intervals)
        candidates = observation.pop("_candidates", [])
        speaker, confidence, region_fallback = _select_speaker(
            candidates, presentation_crop, presentation_name,
            previous_face, is_speech, info["width"], info["height"],
        )
        eligible, eligible_fallback = _eligible_speaker_candidates(
            candidates, presentation_crop, presentation_name, info["width"], info["height"],
        )
        selected = next((item for item in candidates if item.get("face") == speaker), None)
        zoom_border = float((selected or {}).get("zoom_border", 0.0))
        mouth_motion = float((selected or {}).get("mouth_motion", 0.0))
        observation["face"] = speaker
        observation["confidence"] = round(confidence, 3)
        observation["zoom_border"] = round(zoom_border, 3)
        observation["mouth_motion"] = round(mouth_motion, 3)
        observation["strong_switch"] = bool(
            is_speech and (
                zoom_border >= 0.06
                or (mouth_motion >= 0.10 and confidence >= 0.05)
            )
        )
        observation["speech_active"] = is_speech
        observation["speaker_confirmed"] = bool(is_speech and speaker is not None)
        observation["speaker_region_fallback"] = region_fallback
        observation["participant_faces"] = (
            [item["face"] for item in eligible]
            if not eligible_fallback
            else ([speaker] if speaker is not None else [])
        )
        if speaker is not None:
            previous_face = speaker
    clusters, _ = _cluster_observations(observations, info["width"], info["height"])
    reliable_faces = [_median_face(c) for c in clusters if len(c["faces"]) >= 2]
    content_box = (
        content_bbox_from_projections(row_active, col_active, detection_scale)
        if row_active is not None else None
    )
    grid = detect_zoom_grid(content_box, reliable_faces)
    segments, speakers_detected = build_speaker_segments(
        observations, info["width"], info["height"], duration, grid=grid,
    )
    camera_rects = [grid_tile_for_face(grid, face) for face in reliable_faces] if grid else []
    camera_rects = [rect for rect in camera_rects if rect] or [
        {"x": f[0] - f[2], "y": f[1] - f[3], "width": f[2] * 3, "height": f[3] * 3}
        for f in reliable_faces
    ]
    detail = presentation_detail.get(presentation_name) or [(0.0, 0.0)]
    assessment = assess_presentation(
        presentation_crop, camera_rects,
        float(np.mean([d[0] for d in detail])), float(np.mean([d[1] for d in detail])),
    )
    visible_speaker_ids: list[int] = []
    for segment in segments:
        speaker_id = segment.get("speaker_id")
        if segment.get("visible", speaker_id is not None) and speaker_id is not None:
            if not visible_speaker_ids or visible_speaker_ids[-1] != speaker_id:
                visible_speaker_ids.append(speaker_id)
    return {
        "version": 1,
        "output": {
            "width": OUTPUT_WIDTH,
            "height": PRESENTATION_HEIGHT + SPEAKER_HEIGHT,
            "presentation_height": PRESENTATION_HEIGHT,
            "speaker_height": SPEAKER_HEIGHT,
        },
        "presentation": {
            "preset": presentation_name,
            "crop": presentation_crop,
            "preset_crop": preset_source_crop,
            "letterbox_trim": {
                "top_ratio": round(trim_top, 4),
                "bottom_ratio": round(trim_bottom, 4),
            },
            "scores": average_scores,
            "is_screen_share": assessment["valid"],
            "assessment": assessment,
        },
        "zoom_grid": grid,
        "speaker_segments": segments,
        "speakers_detected": speakers_detected,
        "speaker_switches": max(0, len(visible_speaker_ids) - 1),
        "samples": observations,
    }


def speaker_overlay_geometry(layout: dict) -> list[dict]:
    """Resolve stable left/right speaker slots without swapping on active turns."""
    displayed_speakers = normalize_speaker_count(layout.get("displayed_speakers", 2))
    geometry = []
    for segment in layout.get("speaker_segments") or []:
        crop = segment.get("left_crop") or segment.get("crop") or {}
        try:
            source_width = max(2, int(crop["width"]))
            source_height = max(2, int(crop["height"]))
        except (KeyError, TypeError, ValueError):
            raise PipelineError("Таймлайн активных спикеров содержит некорректный crop.")
        secondary_crop = None
        if displayed_speakers == 2:
            secondary_crop = segment.get("right_crop") or segment.get("secondary_crop")
        has_secondary = isinstance(secondary_crop, dict) and bool(secondary_crop)
        max_width = SPEAKER_PAIR_MAX_WIDTH if has_secondary else SPEAKER_OVERLAY_MAX_WIDTH
        available_height = max(2, OUTPUT_HEIGHT - SPEAKER_OVERLAY_BOTTOM)
        scale = min(
            SPEAKER_OVERLAY_SCALE,
            max_width / source_width,
            available_height / source_height,
        )
        render_width = even(source_width * scale)
        render_height = even(source_height * scale)
        explicit_visible = segment.get("visible")
        visible = (
            segment.get("fallback") != "no_face_detected"
            if explicit_visible is None else bool(explicit_visible)
        )
        item = {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "crop": crop,
            "x": SPEAKER_OVERLAY_X,
            "bottom": SPEAKER_OVERLAY_BOTTOM,
            "width": render_width,
            "height": render_height,
            "scale": round(scale, 5),
            "visible": visible,
            "secondary_visible": False,
            "speaker_id": segment.get("left_speaker_id", segment.get("speaker_id")),
        }
        if has_secondary and visible:
            try:
                secondary_source_width = max(2, int(secondary_crop["width"]))
                secondary_source_height = max(2, int(secondary_crop["height"]))
            except (KeyError, TypeError, ValueError):
                raise PipelineError("Таймлайн второго спикера содержит некорректный crop.")
            secondary_scale = min(
                SPEAKER_OVERLAY_SCALE,
                SPEAKER_PAIR_MAX_WIDTH / secondary_source_width,
                available_height / secondary_source_height,
            )
            secondary_width = even(secondary_source_width * secondary_scale)
            secondary_height = even(secondary_source_height * secondary_scale)
            item.update({
                "secondary_crop": secondary_crop,
                "secondary_x": OUTPUT_WIDTH - SPEAKER_OVERLAY_X - secondary_width,
                "secondary_width": secondary_width,
                "secondary_height": secondary_height,
                "secondary_scale": round(secondary_scale, 5),
                "secondary_visible": True,
                "secondary_speaker_id": segment.get(
                    "right_speaker_id", segment.get("secondary_speaker_id")
                ),
            })
        geometry.append(item)
    return geometry


def speaker_safe_areas(layout: dict) -> list[dict]:
    visible = [item for item in speaker_overlay_geometry(layout) if item["visible"]]
    areas = [{
        "kind": "speaker",
        "x": SPEAKER_OVERLAY_X,
        "bottom": SPEAKER_OVERLAY_BOTTOM,
        "width": max((item["width"] for item in visible), default=0),
        "height": max((item["height"] for item in visible), default=0),
    }]
    secondary = [item for item in visible if item.get("secondary_visible")]
    if secondary:
        areas.append({
            "kind": "speaker_secondary",
            "x": min(item["secondary_x"] for item in secondary),
            "bottom": SPEAKER_OVERLAY_BOTTOM,
            "width": max(item["secondary_width"] for item in secondary),
            "height": max(item["secondary_height"] for item in secondary),
        })
    return areas


def build_broadcast_filter(layout: dict, ass_path: Path, playback_speed: float = 1.0) -> str:
    playback_speed = normalize_playback_speed(playback_speed)
    segments = layout.get("speaker_segments") or []
    geometry = speaker_overlay_geometry(layout)
    if not segments or len(segments) != len(geometry):
        raise PipelineError("Не удалось построить таймлайн спикеров эфира.")
    count = len(segments)
    parts = [
        "[0:v]fps=30,settb=AVTB,setsar=1[zoom]",
        f"[1:v]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},fps=30,settb=AVTB,setsar=1[motion]",
    ]
    if count > 1:
        zoom_labels = "".join(f"[z{index}]" for index in range(count))
        motion_labels = "".join(f"[m{index}]" for index in range(count))
        parts.extend([
            f"[zoom]split={count}{zoom_labels}",
            f"[motion]split={count}{motion_labels}",
        ])
    for index, item in enumerate(geometry):
        zoom_source = "[zoom]" if count == 1 else f"[z{index}]"
        motion_source = "[motion]" if count == 1 else f"[m{index}]"
        start, end = item["start"], item["end"]
        parts.append(
            f"{motion_source}trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS[bg{index}]"
        )
        if not item["visible"]:
            parts.extend([
                f"{zoom_source}nullsink",
                f"[bg{index}]null[s{index}]",
            ])
            continue
        crop = item["crop"]
        border_color = str(layout.get("speaker_border_color") or SPEAKER_BORDER_COLOR)
        border = (
            f"drawbox=x=0:y=0:w=iw:h=ih:color={border_color}:"
            f"t={SPEAKER_BORDER_WIDTH}"
        )
        if item.get("secondary_visible"):
            parts.append(
                f"{zoom_source}trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
                f"split=2[spbase{index}][secondarybase{index}]"
            )
            primary_source = f"[spbase{index}]"
        else:
            primary_source = (
                f"{zoom_source}trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
            )
        parts.append(
            f"{primary_source}crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']},"
            f"scale={item['width']}:{item['height']}:flags=lanczos,setsar=1,"
            f"{border}[sp{index}]"
        )
        primary_output = f"[primary{index}]" if item.get("secondary_visible") else f"[s{index}]"
        parts.append(
            f"[bg{index}][sp{index}]overlay=x={item['x']}:"
            f"y=main_h-overlay_h-{item['bottom']}:eval=init:eof_action=pass:shortest=0,"
            f"setsar=1{primary_output}"
        )
        if item.get("secondary_visible"):
            secondary_crop = item["secondary_crop"]
            parts.append(
                f"[secondarybase{index}]crop={secondary_crop['width']}:{secondary_crop['height']}:"
                f"{secondary_crop['x']}:{secondary_crop['y']},"
                f"scale={item['secondary_width']}:{item['secondary_height']}:flags=lanczos,"
                f"setsar=1,{border}[secondary{index}]"
            )
            parts.append(
                f"[primary{index}][secondary{index}]overlay=x={item['secondary_x']}:"
                f"y=main_h-overlay_h-{item['bottom']}:eval=init:eof_action=pass:shortest=0,"
                f"setsar=1[s{index}]"
            )
    if count == 1:
        composed = "[s0]"
    else:
        concat_inputs = "".join(f"[s{index}]" for index in range(count))
        parts.append(f"{concat_inputs}concat=n={count}:v=1:a=0[composed]")
        composed = "[composed]"
    safe_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    subtitle_target = "[outv]" if playback_speed == 1.0 else "[captioned]"
    parts.append(
        f"{composed}subtitles=filename='{safe_ass}'{runtime_paths.subtitles_fontsdir_option()}{subtitle_target}"
    )
    if playback_speed != 1.0:
        parts.append(f"[captioned]setpts=PTS/{playback_speed:.2f}[outv]")
    return ";\n".join(parts) + ";\n"


def _render_broadcast(video_path: Path, motion_path: Path, output_path: Path, work_dir: Path,
                      transcript: dict, info: dict, layout: dict,
                      subtitle_style: str, callback: ProgressCallback,
                      playback_speed: float = 1.0) -> None:
    playback_speed = normalize_playback_speed(playback_speed)
    output_duration = float(info["duration"]) / playback_speed
    ass_path = work_dir / "subtitles.ass"
    write_ass(transcript["words"], ass_path, info["duration"], subtitle_style)
    apply_subtitle_margin(ass_path, subtitle_safe_margin(layout.get("safe_areas") or []))
    filter_path = work_dir / "broadcast.filter.txt"
    filter_path.write_text(
        build_broadcast_filter(layout, ass_path, playback_speed), encoding="utf-8",
    )
    log_path = work_dir / "ffmpeg.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    callback(82, "Собираю активного спикера и субтитры")
    command = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "warning", "-i", str(video_path),
        "-i", str(motion_path),
        "-filter_complex_script", str(filter_path), "-map", "[outv]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
    ]
    if playback_speed != 1.0:
        command.extend(["-filter:a:0", f"atempo={playback_speed:.2f}"])
    command.extend([
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        "-t", f"{output_duration:.3f}",
        "-progress", "pipe:1", "-nostats", str(output_path),
    ])
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=log_file,
            text=True, encoding="utf-8", errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            key, _, value = line.strip().partition("=")
            if key in {"out_time_ms", "out_time_us"}:
                try:
                    rendered = int(value) / 1_000_000
                    progress = 82 + min(16, int(rendered / output_duration * 16))
                    callback(progress, "Кодирую спикера и субтитры")
                except (ValueError, ZeroDivisionError):
                    pass
        return_code = process.wait()
    if return_code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1800:]
        raise PipelineError("FFmpeg не смог собрать эфирный ролик. " + tail)


def process_broadcast_with_transcript(video_path: Path, output_path: Path, work_dir: Path,
                                      options: dict, transcript: dict,
                                      callback: ProgressCallback,
                                      candidate: dict | None = None) -> dict:
    """Create a 9:16 HyperFrames Reel with the active Zoom speaker overlaid."""
    work_dir.mkdir(parents=True, exist_ok=True)
    callback(14, "Проверяю выбранный фрагмент эфира")
    info = media_info(video_path)
    playback_speed = normalize_playback_speed(options.get("playback_speed", 1.0))
    output_duration = float(info["duration"]) / playback_speed
    words = transcript.get("words") or []
    if not words:
        raise PipelineError("Для выбранного фрагмента не найдены пословные таймкоды.")
    (work_dir / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    layout = analyze_broadcast_layout(video_path, info, callback, transcript)
    speaker_count = normalize_speaker_count(options.get("speaker_count", 2))
    visual_preset = normalize_visual_preset(options.get("visual_preset", "editorial_lime"))
    layout["render_mode"] = "hyperframes_motion_overlay"
    layout["displayed_speakers"] = speaker_count
    layout["visual_preset"] = visual_preset
    layout["speaker_border_color"] = "0x" + visual_preset_accent(visual_preset).lstrip("#")
    layout["speaker_overlay"] = speaker_overlay_geometry(layout)
    safe_areas = speaker_safe_areas(layout)
    layout["safe_areas"] = safe_areas
    (work_dir / "broadcast_layout.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    visual_plan = build_visual_plan(
        transcript, info["duration"], work_dir, options, candidate, callback,
    )
    if layout["presentation"].get("is_screen_share", True):
        visual_plan = attach_presentation_frames(
            video_path, visual_plan, layout["presentation"]["crop"], work_dir, callback,
        )
    else:
        callback(60, "Демонстрации экрана нет: графика без кадров исходника")
        visual_plan = without_photographic_visuals(visual_plan, info["duration"])
    motion_path = render_hyperframes_visual(
        visual_plan, info["duration"], safe_areas, work_dir, callback,
        visual_preset=visual_preset,
    )
    requested_subtitle_style = options.get("subtitle_style", "pill")
    rendered_subtitle_style = (
        visual_plan.get("caption_style", requested_subtitle_style)
        if requested_subtitle_style == "pill"
        else requested_subtitle_style
    )
    _render_broadcast(
        video_path, motion_path, output_path, work_dir, transcript, info, layout,
        rendered_subtitle_style, callback, playback_speed,
    )
    callback(99, "Проверяю эфирный ролик")
    result = verify_output(output_path, output_duration)
    result.update({
        "words": len(words),
        "language": transcript.get("language", options.get("language", "auto")),
        "layout": "broadcast_motion",
        "visual_engine": "hyperframes",
        "visual_plan_version": visual_plan["version"],
        "visual_plan_provider": visual_plan.get("provider_used", "codex"),
        "catalog_caption_style": rendered_subtitle_style,
        "catalog_items_used": sum(
            3 for scene in visual_plan.get("scenes") or []
        ) + (1 if rendered_subtitle_style.startswith("caption-") else 0),
        "playback_speed": playback_speed,
        "speaker_count": speaker_count,
        "visual_preset": visual_preset,
        "source_duration": float(info["duration"]),
        "speakers_detected": layout["speakers_detected"],
        "speaker_switches": layout["speaker_switches"],
        "presentation_preset": layout["presentation"]["preset"],
    })
    callback(100, "Готово")
    return result
