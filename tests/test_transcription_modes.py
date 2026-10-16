from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from reel_factory import pipeline


class FakeWhisperModel:
    def __init__(self, segments: list[SimpleNamespace], info: SimpleNamespace) -> None:
        self.segments = segments
        self.info = info
        self.calls: list[tuple[str, dict]] = []

    def transcribe(self, video_path: str, **kwargs):
        self.calls.append((video_path, dict(kwargs)))
        return iter(self.segments), self.info


def transcription_info(duration: float = 5.0) -> SimpleNamespace:
    return SimpleNamespace(
        language="ru",
        language_probability=0.98,
        duration=duration,
        duration_after_vad=max(0.0, duration - 0.5),
    )


class FastTranscriptionPayloadTests(unittest.TestCase):
    def test_fast_alias_returns_empty_words_and_segment_timeline(self) -> None:
        model = FakeWhisperModel([
            SimpleNamespace(start=0.0, end=2.0, text=" Первый сегмент "),
            SimpleNamespace(start=2.0, end=5.0, text="Второй сегмент"),
        ], transcription_info())
        progress: list[tuple[int, str]] = []

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(pipeline.MODELS, "get", return_value=model), \
                mock.patch.object(pipeline.MODELS, "runtime_metadata", return_value={
                    "device": "cpu", "compute_type": "int8", "model": "base",
                }):
            work_dir = Path(temp_dir)
            payload = pipeline.transcribe_video(
                Path("zoom-recording.mp4"),
                work_dir,
                "auto",
                "base",
                lambda value, message: progress.append((value, message)),
                mode="fast",
            )

            persisted = json.loads(
                (work_dir / "transcript.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["words"], [])
        self.assertEqual(payload["segments"], [
            {"start": 0.0, "end": 2.0, "text": "Первый сегмент"},
            {"start": 2.0, "end": 5.0, "text": "Второй сегмент"},
        ])
        self.assertEqual(payload["text"], "Первый сегмент Второй сегмент")
        self.assertEqual(payload["transcription"]["mode"], "coarse")
        self.assertFalse(payload["transcription"]["word_timestamps"])
        self.assertEqual(persisted, payload)

        self.assertEqual(len(model.calls), 1)
        _, kwargs = model.calls[0]
        self.assertIsNone(kwargs["language"])
        self.assertFalse(kwargs["word_timestamps"])
        self.assertNotIn("without_timestamps", kwargs)
        self.assertNotIn("condition_on_previous_text", kwargs)
        self.assertTrue(kwargs["vad_filter"])
        self.assertEqual(progress[-1][0], 36)

    def test_original_five_argument_call_remains_word_accurate(self) -> None:
        word = SimpleNamespace(
            word=" Привет! ", start=0.1, end=0.6, probability=0.97,
        )
        model = FakeWhisperModel([
            SimpleNamespace(start=0.0, end=1.0, text="Привет!", words=[word]),
        ], transcription_info(duration=1.0))

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(pipeline.MODELS, "get", return_value=model), \
                mock.patch.object(pipeline.MODELS, "runtime_metadata", return_value={
                    "device": "cpu", "model": "base",
                }):
            payload = pipeline.transcribe_video(
                Path("podcast.mp4"), Path(temp_dir), "ru", "base",
                lambda progress, message: None,
            )

        self.assertEqual(payload["words"], [{
            "text": "Привет",
            "raw": "Привет!",
            "start": 0.1,
            "end": 0.6,
            "probability": 0.97,
        }])
        self.assertEqual(payload["transcription"]["mode"], "accurate")
        self.assertTrue(payload["transcription"]["word_timestamps"])
        _, kwargs = model.calls[0]
        self.assertEqual(kwargs["language"], "ru")
        self.assertIn("э-э", kwargs["initial_prompt"])
        self.assertNotIn("condition_on_previous_text", kwargs)
        self.assertTrue(kwargs["word_timestamps"])
        self.assertNotIn("without_timestamps", kwargs)


class DisfluencyPromptTests(unittest.TestCase):
    def run_transcribe(self, language: str, **kwargs) -> dict:
        model = FakeWhisperModel([
            SimpleNamespace(start=0.0, end=1.0, text="Ну, э-э.", words=[
                SimpleNamespace(word=" Ну,", start=0.0, end=0.3, probability=0.9)]),
        ], transcription_info(duration=1.0))
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(pipeline.MODELS, "get", return_value=model), \
                mock.patch.object(pipeline.MODELS, "runtime_metadata", return_value={"device": "cpu"}):
            pipeline.transcribe_video(
                Path("a.mp4"), Path(temp_dir), language, "small", lambda p, m: None, **kwargs)
        return model.calls[0][1]

    def test_prompt_is_language_aware_and_disableable(self) -> None:
        self.assertIn("как бы", self.run_transcribe("ru")["initial_prompt"])
        self.assertIn("you know", self.run_transcribe("en")["initial_prompt"])
        self.assertNotIn("initial_prompt", self.run_transcribe("ru", keep_disfluencies=False, glossary=[]))
        self.assertNotIn("initial_prompt", self.run_transcribe("auto"))

    def test_glossary_default_for_ru_custom_and_disabled(self) -> None:
        default = self.run_transcribe("ru")["initial_prompt"]
        self.assertIn("Термины: ИИ, ИИшка, нейросети, фулстек", default)
        self.assertIn("как бы", default)
        custom = self.run_transcribe("ru", glossary="Kubernetes, Альфа-Банк")["initial_prompt"]
        self.assertIn("Термины: Kubernetes, Альфа-Банк.", custom)
        self.assertNotIn("фулстек", custom)
        self.assertNotIn("Термины", self.run_transcribe("ru", glossary=[])["initial_prompt"])
        glossary_only = self.run_transcribe("ru", keep_disfluencies=False)["initial_prompt"]
        self.assertTrue(glossary_only.startswith("Термины:"))
        self.assertNotIn("Термины", self.run_transcribe("en")["initial_prompt"])


class TranscriptionTelemetryTests(unittest.TestCase):
    def test_legacy_and_structured_progress_are_monotonic(self) -> None:
        model = FakeWhisperModel([
            SimpleNamespace(start=0.0, end=1.0, text="Раз"),
            SimpleNamespace(start=1.0, end=3.0, text="Два"),
            SimpleNamespace(start=3.0, end=6.0, text="Три"),
        ], transcription_info(duration=6.0))
        legacy: list[int] = []
        telemetry: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(pipeline.MODELS, "get", return_value=model), \
                mock.patch.object(pipeline.MODELS, "runtime_metadata", return_value={
                    "device": "cpu", "model": "base",
                }):
            pipeline.transcribe_video(
                Path("broadcast.mp4"), Path(temp_dir), "ru", "base",
                lambda progress, message: legacy.append(progress),
                mode="coarse",
                telemetry_callback=telemetry.append,
            )

        events = [item for item in telemetry if item["event"] == "transcription_progress"]
        self.assertGreaterEqual(len(events), 4)
        self.assertEqual(legacy, sorted(legacy))
        self.assertEqual(
            [item["processed_seconds"] for item in events],
            sorted(item["processed_seconds"] for item in events),
        )
        self.assertEqual(
            [item["progress"] for item in events],
            sorted(item["progress"] for item in events),
        )
        self.assertEqual(
            [item["percent"] for item in events],
            sorted(item["percent"] for item in events),
        )
        self.assertTrue(all(item["total_seconds"] == 6.0 for item in events))
        self.assertEqual(events[-1]["processed_seconds"], 6.0)
        self.assertEqual(events[-1]["progress"], 1.0)
        self.assertEqual(events[-1]["percent"], 100)

    def test_selected_clip_wrapper_always_requests_accurate_mode(self) -> None:
        callback = mock.Mock()
        telemetry = mock.Mock()
        expected = {"words": [{"text": "точно"}]}

        with mock.patch.object(
            pipeline, "transcribe_video", return_value=expected,
        ) as transcriber:
            result = pipeline.transcribe_selected_clip(
                Path("selected.mp4"), Path("accurate"), "ru", "small", callback,
                telemetry_callback=telemetry,
            )

        self.assertIs(result, expected)
        transcriber.assert_called_once_with(
            Path("selected.mp4"), Path("accurate"), "ru", "small", callback,
            mode="accurate", telemetry_callback=telemetry,
        )


if __name__ == "__main__":
    unittest.main()
