from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import reel_factory.broadcast as broadcast_module
from reel_factory.broadcast import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SPEAKER_OVERLAY_BOTTOM,
    SPEAKER_BORDER_COLOR,
    SPEAKER_BORDER_WIDTH,
    SPEAKER_OVERLAY_MAX_WIDTH,
    SPEAKER_PAIR_MAX_WIDTH,
    SPEAKER_OVERLAY_SCALE,
    SPEAKER_OVERLAY_X,
    _select_speaker,
    _render_broadcast,
    build_broadcast_filter,
    build_speaker_segments,
    clamp_crop,
    fallback_speaker_crop,
    score_presentation_preset,
    select_presentation_preset,
    smooth_speaker_ids,
    smooth_speaker_timeline,
    speaker_overlay_geometry,
    speaker_safe_areas,
    speaker_crop_from_face,
    top_anchored_presentation_crop,
    trim_presentation_letterbox,
)


class CropHelperTests(unittest.TestCase):
    def assert_even_clamped(self, crop: dict, source_width: int, source_height: int) -> None:
        for key in ("x", "y", "width", "height"):
            self.assertIsInstance(crop[key], int)
            self.assertEqual(crop[key] % 2, 0, f"{key} must be even: {crop}")
        self.assertGreaterEqual(crop["x"], 0)
        self.assertGreaterEqual(crop["y"], 0)
        self.assertGreaterEqual(crop["width"], 2)
        self.assertGreaterEqual(crop["height"], 2)
        self.assertLessEqual(crop["x"] + crop["width"], source_width)
        self.assertLessEqual(crop["y"] + crop["height"], source_height)

    def test_clamp_crop_keeps_odd_and_out_of_range_input_h264_safe(self) -> None:
        crop = clamp_crop(-31, 999, 101, 103, 319, 179)

        self.assert_even_clamped(crop, 319, 179)
        self.assertEqual(crop["x"], 0)
        self.assertEqual(crop["y"] + crop["height"], 178)

    def test_face_and_fallback_crops_are_even_and_inside_source(self) -> None:
        source_width, source_height = 1919, 1079

        face_crop = speaker_crop_from_face(
            (source_width - 24, source_height - 18, 80, 90),
            source_width,
            source_height,
        )
        fallback_crop = fallback_speaker_crop(source_width, source_height)

        self.assert_even_clamped(face_crop, source_width, source_height)
        self.assert_even_clamped(fallback_crop, source_width, source_height)

    def test_small_zoom_face_crop_stays_inside_its_tile(self) -> None:
        crop = speaker_crop_from_face((1406, 134, 56, 56), 1920, 1080)

        self.assertEqual(crop["height"], 184)
        self.assertEqual(crop["width"], 150)
        self.assertGreaterEqual(crop["x"], 1280)
        self.assertLessEqual(crop["x"] + crop["width"], 1600)
        self.assertLessEqual(abs(crop["y"] - 88), 2)
        self.assertLessEqual(crop["y"] + crop["height"], 274)


class PresentationScoringTests(unittest.TestCase):
    def test_detailed_face_free_area_beats_plain_area_and_full_frame(self) -> None:
        height, width = 180, 320
        frame = np.full((height, width, 3), 112, dtype=np.uint8)
        yy, xx = np.indices((height, 212))
        checker = (((xx // 6) + (yy // 6)) % 2 * 255).astype(np.uint8)
        frame[:, :212] = np.repeat(checker[:, :, None], 3, axis=2)
        faces = [(246, 48, 52, 68)]

        detailed_score = score_presentation_preset(frame, faces, "left_66")
        plain_face_score = score_presentation_preset(frame, faces, "right_66")
        full_frame_score = score_presentation_preset(frame, faces, "full")

        self.assertGreater(detailed_score, plain_face_score)
        self.assertGreater(detailed_score, full_frame_score)

    def test_letterbox_trim_and_top_anchor_produce_native_slide_crop(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[171:904, :1280] = 240
        preset = {"x": 0, "y": 0, "width": 1280, "height": 1080}

        trimmed = trim_presentation_letterbox(frame, preset)
        fitted = top_anchored_presentation_crop(trimmed, 1920, 1080)

        self.assertLessEqual(abs(trimmed["y"] - 171), 4)
        self.assertEqual(fitted["x"], 0)
        self.assertLessEqual(abs(fitted["y"] - 171), 4)
        self.assertEqual(fitted["width"], 1280)
        self.assertLessEqual(abs(fitted["height"] - 720), 2)

    def test_near_tied_vertical_region_yields_to_clean_side_share(self) -> None:
        scores = {
            "full": 1.05,
            "left_78": 1.16,
            "right_78": 0.82,
            "left_66": 2.16,
            "right_66": 0.70,
            "top_78": 1.03,
            "bottom_78": 2.28,
        }

        self.assertEqual(select_presentation_preset(scores), "left_66")
        scores["bottom_78"] = 2.5
        self.assertEqual(select_presentation_preset(scores), "bottom_78")


class SpeakerTimelineTests(unittest.TestCase):
    def test_single_sample_false_switch_is_debounced(self) -> None:
        self.assertEqual(
            smooth_speaker_ids([0, 0, 1, 0, 0], minimum_run=2),
            [0, 0, 0, 0, 0],
        )

    def test_sustained_switch_moves_cut_to_start_of_confirmed_run(self) -> None:
        self.assertEqual(
            smooth_speaker_ids([0, 0, 0, 1, 1, 1, 1], minimum_run=2),
            [0, 0, 0, 1, 1, 1, 1],
        )

    def test_timeline_debounce_uses_elapsed_time_and_moves_cut_back(self) -> None:
        self.assertEqual(
            smooth_speaker_timeline(
                [0, 0, 1, 1, 1], [0.0, 0.5, 1.0, 1.55, 2.1], minimum_seconds=0.8,
            ),
            [0, 0, 1, 1, 1],
        )

    def test_strong_zoom_or_mouth_signal_switches_on_first_observation(self) -> None:
        self.assertEqual(
            smooth_speaker_timeline(
                [0, 0, 1, 1],
                [0.0, 0.3, 0.6, 0.9],
                minimum_seconds=0.65,
                strong_switches=[False, False, True, False],
            ),
            [0, 0, 1, 1],
        )

    def test_speaker_candidate_inside_presentation_is_excluded(self) -> None:
        candidates = [
            {"face": [120, 100, 140, 140], "score": 6.0},
            {"face": [860, 110, 100, 100], "score": 3.0},
        ]

        face, confidence, fallback = _select_speaker(
            candidates,
            {"x": 0, "y": 0, "width": 780, "height": 600},
            "left_78",
            previous_face=None,
            speech_active=True,
            source_width=1000,
            source_height=600,
        )

        self.assertEqual(face, [860, 110, 100, 100])
        self.assertEqual(confidence, 1.0)
        self.assertFalse(fallback)

    def test_silence_holds_previous_speaker(self) -> None:
        face, _, _ = _select_speaker(
            [{"face": [760, 90, 120, 120], "score": 5.0}],
            {"x": 0, "y": 0, "width": 650, "height": 600},
            "left_66",
            previous_face=[80, 90, 120, 120],
            speech_active=False,
            source_width=1000,
            source_height=600,
        )

        self.assertEqual(face, [80, 90, 120, 120])

    def test_full_presentation_prefers_small_edge_overlay(self) -> None:
        candidates = [
            {"face": [360, 120, 300, 300], "score": 6.0, "zoom_border": 0.0},
            {"face": [875, 20, 90, 90], "score": 2.0, "zoom_border": 0.0},
        ]

        face, _, fallback = _select_speaker(
            candidates,
            {"x": 0, "y": 0, "width": 1000, "height": 600},
            "full",
            previous_face=None,
            speech_active=True,
            source_width=1000,
            source_height=600,
        )

        self.assertEqual(face, [875, 20, 90, 90])
        self.assertFalse(fallback)

    def test_ambiguous_speech_does_not_select_unconfirmed_candidate(self) -> None:
        face, _, _ = _select_speaker(
            [
                {"face": [620, 90, 120, 120], "score": 2.0, "mouth_motion": 0.0, "zoom_border": 0.0},
                {"face": [800, 90, 120, 120], "score": 1.95, "mouth_motion": 0.0, "zoom_border": 0.0},
            ],
            {"x": 0, "y": 0, "width": 500, "height": 600},
            "left_66",
            previous_face=[620, 90, 120, 120],
            speech_active=True,
            source_width=1000,
            source_height=600,
        )

        self.assertIsNone(face)

    def test_a_to_b_observations_form_two_contiguous_segments(self) -> None:
        face_a = [80, 90, 120, 120]
        face_b = [760, 90, 120, 120]
        observations = [
            {"time": float(index), "face": face_a if index < 3 else face_b, "confidence": 0.8}
            for index in range(6)
        ]

        segments, speaker_count = build_speaker_segments(
            observations,
            source_width=1000,
            source_height=600,
            duration=6.0,
        )

        self.assertEqual(speaker_count, 2)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[-1]["end"], 6.0)
        self.assertEqual(segments[0]["end"], segments[1]["start"])
        self.assertNotEqual(segments[0]["speaker_id"], segments[1]["speaker_id"])
        self.assertLess(segments[0]["crop"]["x"], segments[1]["crop"]["x"])
        self.assertEqual(segments[0]["left_speaker_id"], segments[1]["left_speaker_id"])
        self.assertEqual(segments[0]["right_speaker_id"], segments[1]["right_speaker_id"])
        self.assertEqual(segments[0]["left_crop"], segments[1]["left_crop"])
        self.assertEqual(segments[0]["right_crop"], segments[1]["right_crop"])
        self.assertNotEqual(segments[0]["left_speaker_id"], segments[0]["right_speaker_id"])
        geometry = speaker_overlay_geometry({"speaker_segments": segments})
        self.assertEqual(geometry[0]["crop"], geometry[1]["crop"])
        self.assertEqual(geometry[0]["secondary_crop"], geometry[1]["secondary_crop"])

    def test_no_faces_produces_centered_fallback_segment(self) -> None:
        observations = [
            {"time": 0.0, "face": None, "confidence": 0.0},
            {"time": 1.0, "face": None, "confidence": 0.0},
        ]

        segments, speaker_count = build_speaker_segments(
            observations,
            source_width=1280,
            source_height=720,
            duration=2.0,
        )

        self.assertEqual(speaker_count, 0)
        self.assertEqual(len(segments), 1)
        self.assertIsNone(segments[0]["speaker_id"])
        self.assertEqual(segments[0]["fallback"], "no_face_detected")
        self.assertEqual(segments[0]["crop"], fallback_speaker_crop(1280, 720))

    def test_persistent_silent_participant_becomes_secondary_speaker(self) -> None:
        active = [80, 90, 120, 120]
        listener = [760, 90, 120, 120]
        observations = [
            {
                "time": float(index),
                "face": active,
                "participant_faces": [active, listener],
                "confidence": 0.8,
            }
            for index in range(3)
        ]

        segments, speaker_count = build_speaker_segments(
            observations, source_width=1000, source_height=600, duration=3.0,
        )

        self.assertEqual(speaker_count, 2)
        self.assertEqual(len(segments), 1)
        self.assertIsNotNone(segments[0].get("right_speaker_id"))
        self.assertGreater(segments[0]["right_crop"]["x"], segments[0]["left_crop"]["x"])

    def test_tracked_speaker_remains_visible_during_silence(self) -> None:
        face = [760, 90, 120, 120]
        observations = [
            {"time": 0.0, "face": face, "confidence": 0.8, "speech_active": True},
            {"time": 1.0, "face": face, "confidence": 0.0, "speech_active": False},
            {"time": 2.0, "face": face, "confidence": 0.0, "speech_active": False},
        ]

        segments, _ = build_speaker_segments(
            observations, source_width=1000, source_height=600, duration=3.0,
        )

        self.assertEqual(len(segments), 1)
        self.assertNotIn("visible", segments[0])
        self.assertTrue(speaker_overlay_geometry({"speaker_segments": segments})[0]["visible"])

class BroadcastFilterTests(unittest.TestCase):
    def test_speaker_overlay_is_lower_left_and_sixty_nine_percent_larger(self) -> None:
        self.assertEqual(OUTPUT_WIDTH, 1080)
        self.assertEqual(OUTPUT_HEIGHT, 1920)
        self.assertEqual(SPEAKER_OVERLAY_X, 54)
        self.assertEqual(SPEAKER_OVERLAY_BOTTOM, 144)
        self.assertEqual(SPEAKER_OVERLAY_SCALE, 1.69)
        self.assertEqual(SPEAKER_OVERLAY_MAX_WIDTH, 608)
        self.assertEqual(SPEAKER_PAIR_MAX_WIDTH, 432)
        self.assertEqual(SPEAKER_BORDER_COLOR, "0xC7F54A")
        self.assertEqual(SPEAKER_BORDER_WIDTH, 6)

        layout = {
            "speaker_segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "crop": {"x": 0, "y": 0, "width": 640, "height": 480},
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "crop": {"x": 640, "y": 0, "width": 200, "height": 180},
                },
            ],
        }

        geometry = speaker_overlay_geometry(layout)

        self.assertEqual((geometry[0]["width"], geometry[0]["height"]), (608, 456))
        self.assertEqual((geometry[1]["width"], geometry[1]["height"]), (338, 304))
        for item in geometry:
            self.assertLessEqual(item["scale"], 1.69)
            self.assertEqual(item["x"], 54)
            self.assertEqual(item["bottom"], 144)

        self.assertEqual(
            speaker_safe_areas(layout),
            [{"kind": "speaker", "x": 54, "bottom": 144, "width": 608, "height": 456}],
        )

    def test_filter_uses_motion_input_and_speaker_overlay_without_vstack(self) -> None:
        layout = {
            "speaker_segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "crop": {"x": 0, "y": 0, "width": 640, "height": 480},
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "crop": {"x": 640, "y": 0, "width": 200, "height": 180},
                },
            ],
        }

        filter_text = build_broadcast_filter(layout, Path("captions.ass"))

        self.assertIn("[0:v]fps=30,settb=AVTB,setsar=1[zoom]", filter_text)
        self.assertIn(
            "[1:v]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1080:1920,fps=30,settb=AVTB,setsar=1[motion]",
            filter_text,
        )
        self.assertIn("[zoom]split=2[z0][z1]", filter_text)
        self.assertIn("[motion]split=2[m0][m1]", filter_text)
        self.assertIn("trim=start=0.000:end=2.500", filter_text)
        self.assertIn("crop=640:480:0:0,scale=608:456:flags=lanczos", filter_text)
        self.assertIn("crop=200:180:640:0,scale=338:304:flags=lanczos", filter_text)
        self.assertIn(
            "overlay=x=54:y=main_h-overlay_h-144:eval=init:eof_action=pass:shortest=0",
            filter_text,
        )
        self.assertIn("concat=n=2:v=1:a=0[composed]", filter_text)
        self.assertIn("[composed]subtitles=filename=", filter_text)
        self.assertNotIn("vstack", filter_text)
        self.assertNotIn("scale=640:480", filter_text)
        self.assertEqual(filter_text.count("drawbox=x=0:y=0:w=iw:h=ih:color=0xC7F54A:t=6"), 2)

    def test_second_speaker_is_bordered_and_placed_on_the_right(self) -> None:
        layout = {
            "speaker_segments": [{
                "start": 0.0,
                "end": 3.0,
                "speaker_id": 0,
                "crop": {"x": 0, "y": 0, "width": 320, "height": 240},
                "secondary_speaker_id": 1,
                "secondary_crop": {"x": 320, "y": 0, "width": 320, "height": 240},
            }],
        }

        geometry = speaker_overlay_geometry(layout)[0]
        filter_text = build_broadcast_filter(layout, Path("captions.ass"))

        self.assertEqual((geometry["width"], geometry["height"]), (432, 324))
        self.assertEqual(geometry["x"], 54)
        self.assertTrue(geometry["secondary_visible"])
        self.assertEqual(geometry["secondary_x"], 594)
        self.assertEqual((geometry["secondary_width"], geometry["secondary_height"]), (432, 324))
        self.assertIn("split=2[spbase0][secondarybase0]", filter_text)
        self.assertIn("[primary0][secondary0]overlay=x=594", filter_text)
        self.assertEqual(filter_text.count("drawbox=x=0:y=0:w=iw:h=ih:color=0xC7F54A:t=6"), 2)
        self.assertEqual(
            speaker_safe_areas(layout),
            [
                {"kind": "speaker", "x": 54, "bottom": 144, "width": 432, "height": 324},
                {"kind": "speaker_secondary", "x": 594, "bottom": 144, "width": 432, "height": 324},
            ],
        )

    def test_one_speaker_mode_hides_the_stable_right_slot(self) -> None:
        layout = {
            "displayed_speakers": 1,
            "speaker_segments": [{
                "start": 0.0,
                "end": 3.0,
                "left_speaker_id": 0,
                "left_crop": {"x": 0, "y": 0, "width": 320, "height": 240},
                "right_speaker_id": 1,
                "right_crop": {"x": 320, "y": 0, "width": 320, "height": 240},
            }],
        }

        geometry = speaker_overlay_geometry(layout)[0]
        filter_text = build_broadcast_filter(layout, Path("captions.ass"))

        self.assertFalse(geometry["secondary_visible"])
        self.assertEqual((geometry["width"], geometry["height"]), (540, 406))
        self.assertNotIn("secondarybase", filter_text)
        self.assertEqual(filter_text.count("drawbox=x=0:y=0:w=iw:h=ih:color=0xC7F54A:t=6"), 1)

    def test_preset_accent_can_drive_speaker_border(self) -> None:
        layout = {
            "speaker_border_color": "0xAEFBC3",
            "speaker_segments": [{
                "start": 0.0,
                "end": 3.0,
                "crop": {"x": 0, "y": 0, "width": 320, "height": 240},
            }],
        }

        filter_text = build_broadcast_filter(layout, Path("captions.ass"))

        self.assertIn("drawbox=x=0:y=0:w=iw:h=ih:color=0xAEFBC3:t=6", filter_text)

    def test_hidden_speaker_segment_uses_only_motion_background(self) -> None:
        layout = {
            "speaker_segments": [{
                "start": 0.0,
                "end": 5.0,
                "speaker_id": None,
                "visible": False,
                "fallback": "silence",
                "crop": {"x": 0, "y": 0, "width": 640, "height": 480},
            }],
        }

        filter_text = build_broadcast_filter(layout, Path("captions.ass"))

        self.assertIn("[zoom]nullsink", filter_text)
        self.assertIn("[bg0]null[s0]", filter_text)
        self.assertNotIn("overlay=x=", filter_text)

    def test_speed_retimes_composited_video_after_subtitles(self) -> None:
        layout = {
            "speaker_segments": [{
                "start": 0.0,
                "end": 5.0,
                "crop": {"x": 0, "y": 0, "width": 640, "height": 480},
            }],
        }

        filter_text = build_broadcast_filter(layout, Path("captions.ass"), 1.25)

        self.assertIn("subtitles=filename=", filter_text)
        self.assertIn("[captioned]setpts=PTS/1.25[outv]", filter_text)

    def test_renderer_maps_audio_only_from_zoom_input_zero(self) -> None:
        layout = {
            "speaker_segments": [{
                "start": 0.0,
                "end": 5.0,
                "crop": {"x": 0, "y": 0, "width": 640, "height": 480},
            }],
        }
        captured: dict[str, list[str]] = {}

        class FakeProcess:
            stdout = iter(["out_time_ms=1000000\n"])

            @staticmethod
            def wait() -> int:
                return 0

        def fake_popen(command, **_kwargs):
            captured["command"] = command
            return FakeProcess()

        with tempfile.TemporaryDirectory() as raw_tmp:
            work_dir = Path(raw_tmp)
            with (
                patch.object(broadcast_module, "write_ass"),
                patch.object(broadcast_module.subprocess, "Popen", side_effect=fake_popen),
            ):
                _render_broadcast(
                    Path("zoom.mp4"),
                    Path("motion.mp4"),
                    work_dir / "result.mp4",
                    work_dir,
                    {"words": []},
                    {"duration": 5.0},
                    layout,
                    "pill",
                    lambda _progress, _message: None,
                    1.25,
                )

        command = captured["command"]
        input_positions = [index for index, token in enumerate(command) if token == "-i"]
        self.assertEqual([command[index + 1] for index in input_positions], ["zoom.mp4", "motion.mp4"])
        map_positions = [index for index, token in enumerate(command) if token == "-map"]
        self.assertEqual([command[index + 1] for index in map_positions], ["[outv]", "0:a:0?"])
        self.assertFalse(any(token.startswith("1:a") for token in command))
        self.assertIn("atempo=1.25", command)
        self.assertIn("4.000", command)


class BroadcastSafeZoneTests(unittest.TestCase):
    """Presentation validity, subtitle safe zone and tile-bound speaker crops."""

    GALLERY = {"x": 8, "y": 244, "width": 2544, "height": 952}  # 3x2 tiles of 848x476
    FACES = [(380, 330, 130, 160), (1240, 330, 130, 160)]

    def test_detect_zoom_grid_prefers_fewest_tiles_with_one_face_each(self) -> None:
        grid = broadcast_module.detect_zoom_grid(self.GALLERY, self.FACES)
        self.assertEqual((grid["cols"], grid["rows"]), (3, 2))
        tile = broadcast_module.grid_tile_for_face(grid, self.FACES[1])
        self.assertAlmostEqual(tile["x"], 856, delta=1)
        self.assertAlmostEqual(tile["height"], 476, delta=1)

    def test_detect_zoom_grid_rejects_non_gallery_layouts(self) -> None:
        wide = {"x": 0, "y": 0, "width": 2560, "height": 300}
        self.assertIsNone(broadcast_module.detect_zoom_grid(wide, self.FACES[:1]))
        self.assertIsNone(broadcast_module.detect_zoom_grid(None, self.FACES))

    def test_speaker_crop_excludes_ui_strip_and_neighbours(self) -> None:
        grid = broadcast_module.detect_zoom_grid(self.GALLERY, self.FACES)
        face = self.FACES[0]
        tile = broadcast_module.grid_tile_for_face(grid, face)
        inner = broadcast_module.inset_tile(tile)
        crop = broadcast_module.speaker_crop_in_tile([face] * 5, tile, 2560, 1440)
        self.assertGreaterEqual(crop["x"], inner["x"] - 2)
        self.assertGreaterEqual(crop["y"], inner["y"] - 2)
        self.assertLessEqual(crop["x"] + crop["width"], inner["x"] + inner["width"] + 2)
        self.assertLessEqual(crop["y"] + crop["height"], inner["y"] + inner["height"] + 2)
        self.assertLess(crop["y"] + crop["height"], tile["y"] + tile["height"] * 0.94)

    def test_speaker_crop_keeps_moving_face_inside_with_margin(self) -> None:
        grid = broadcast_module.detect_zoom_grid(self.GALLERY, self.FACES)
        base = self.FACES[1]
        faces = [(base[0] + dx, base[1], base[2], base[3]) for dx in (-120, -60, 0, 0, 0, 0, 60, 120)]
        tile = broadcast_module.grid_tile_for_face(grid, base)
        crop = broadcast_module.speaker_crop_in_tile(faces, tile, 2560, 1440)
        for x, y, w, h in faces:
            self.assertGreater(x - crop["x"], 0.2 * w)
            self.assertGreater(crop["x"] + crop["width"] - (x + w), 0.2 * w)
            self.assertGreater(y, crop["y"])
            self.assertLess(y + h, crop["y"] + crop["height"])

    def test_presentation_over_camera_tiles_is_not_a_screen_share(self) -> None:
        grid = broadcast_module.detect_zoom_grid(self.GALLERY, self.FACES)
        rects = [broadcast_module.grid_tile_for_face(grid, f) for f in self.FACES]
        crop = {"x": 496, "y": 316, "width": 1566, "height": 882}
        result = broadcast_module.assess_presentation(crop, rects, 0.05, 90.0)
        self.assertFalse(result["valid"])
        self.assertIn("camera_tiles_in_crop", result["reasons"])

    def test_dark_or_featureless_crop_is_not_a_screen_share(self) -> None:
        crop = {"x": 0, "y": 0, "width": 1600, "height": 900}
        self.assertIn("low_detail", broadcast_module.assess_presentation(crop, [], 0.003, 90.0)["reasons"])
        self.assertIn("dark", broadcast_module.assess_presentation(crop, [], 0.05, 10.0)["reasons"])

    def test_real_screen_share_beside_thumbnail_strip_stays_valid(self) -> None:
        crop = {"x": 0, "y": 0, "width": 1900, "height": 1080}
        strip = [{"x": 1930, "y": 40, "width": 600, "height": 340}]
        result = broadcast_module.assess_presentation(crop, strip, 0.04, 120.0)
        self.assertTrue(result["valid"])

    def test_subtitle_margin_clears_speaker_tiles(self) -> None:
        areas = [
            {"kind": "speaker", "x": 54, "bottom": 144, "width": 432, "height": 526},
            {"kind": "speaker_secondary", "x": 594, "bottom": 144, "width": 432, "height": 420},
        ]
        margin = broadcast_module.subtitle_safe_margin(areas)
        tile_top = OUTPUT_HEIGHT - 144 - 526
        caption_bottom = OUTPUT_HEIGHT - margin
        self.assertLessEqual(caption_bottom, tile_top - 20)
        self.assertGreater(caption_bottom - broadcast_module.SUBTITLE_BLOCK_HEIGHT, 0)

    def test_apply_subtitle_margin_rewrites_style_margin(self) -> None:
        from reel_factory.pipeline import write_ass
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.ass"
            write_ass([{"text": "да", "start": 0.0, "end": 0.5}], path, 1.0, "pill")
            broadcast_module.apply_subtitle_margin(path, 706)
            self.assertIn(",2,80,80,706,1", path.read_text(encoding="utf-8"))

    def test_without_photographic_visuals_disables_source_and_catalog(self) -> None:
        from reel_factory.hyperframes_visual import without_photographic_visuals
        plan = {"version": 1, "duration": 6.0, "scenes": [{
            "id": "s1", "start": 0.0, "end": 6.0, "kind": "claim", "headline": "Тезис",
            "use_source_frame": True, "source_frame": "/x.jpg", "visual_variant": "source",
        }]}
        with patch("reel_factory.hyperframes_visual.validate_visual_plan", side_effect=lambda p, d: p):
            out = without_photographic_visuals(plan, 6.0)
        scene = out["scenes"][0]
        self.assertFalse(scene["use_source_frame"])
        self.assertEqual(scene["source_frame"], "")
        self.assertTrue(scene["_catalog_disabled"])
        self.assertNotEqual(scene["visual_variant"], "source")


if __name__ == "__main__":
    unittest.main()
