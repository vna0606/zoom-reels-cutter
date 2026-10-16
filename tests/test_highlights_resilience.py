from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel_factory import highlights


def long_transcript(segment_count: int = 70, text_size: int = 1_050) -> dict:
    segments = [
        {
            "start": float(index * 60),
            "end": float((index + 1) * 60),
            "text": f"segment-{index:03d} " + (chr(65 + index % 26) * text_size),
        }
        for index in range(segment_count)
    ]
    return {
        "source": "long.mp4",
        "language": "ru",
        "language_probability": 0.99,
        "words": [],
        "segments": segments,
        "text": " ".join(item["text"] for item in segments),
    }


def word_transcript(count: int = 200) -> dict:
    words = [
        {"text": f"w{i}", "raw": f"w{i}.", "start": i * 0.5, "end": i * 0.5 + 0.4, "probability": 0.9}
        for i in range(count)
    ]
    return {
        "language": "ru", "words": words,
        "segments": [{"start": 0.0, "end": count * 0.5, "text": "w0 w1"}],
        "text": "w0 w1",
    }


def editorial_candidate(start: float, end: float, score: int = 80) -> dict:
    return {
        "title": "t", "description": "d", "hook": "h", "score": score, "rationale": "r",
        "critic": {"verdict": "pass"},
        "segments": [{"start": start, "end": end, "role": "hook", "sentence_from": "S0001", "sentence_to": "S0002"}],
        "cuts": [], "start": start, "end": end, "duration": end - start,
    }


class RejectedCandidatesTests(unittest.TestCase):
    OPTS = {"min_duration": 30, "max_duration": 90, "max_clips": 8}

    def test_rejected_candidates_are_kept_renderable_and_ordered_after_kept_by_chief_rank(self) -> None:
        tr = word_transcript(400)
        second = dict(editorial_candidate(10, 50), title="второй", chief_editor={"rank": 2, "decision": "publish"})
        first = dict(editorial_candidate(60, 100), title="первый", chief_editor={"rank": 1, "decision": "publish"})
        out_of_window = dict(editorial_candidate(110, 112 + 100), title="слишком длинный",
                             rejected={"stage": "chief_editor", "reason": "дубль", "scores": {"clarity": 6}})
        tiny = dict(editorial_candidate(0, 3), title="крошка", rejected={"stage": "x", "reason": "r"})
        got = highlights.normalize_candidates([second, out_of_window, first, tiny], 200.0, self.OPTS, tr["words"])
        self.assertEqual([c["title"] for c in got], ["первый", "второй", "слишком длинный"])
        self.assertEqual([c["id"] for c in got], ["c001", "c002", "c003"])
        self.assertIsNone(got[0]["rejected"])
        self.assertEqual(got[2]["rejected"]["reason"], "дубль")   # окно длительности для отсеянных не режет
        self.assertEqual(got[0]["chief_editor"]["rank"], 1)
        self.assertEqual(got[2]["status"], "suggested")

    def test_analyze_transcript_passes_rejected_through(self) -> None:
        transcript = word_transcript()
        kept = editorial_candidate(0, 40)

        class Composed(list):
            rejected = [dict(editorial_candidate(50, 90), rejected={"stage": "repair", "reason": "стык"})]

        with mock.patch.object(highlights.llm, "resolve_provider", return_value="claude"), \
                mock.patch.object(highlights.editorial, "compose_clips", return_value=Composed([kept])):
            with tempfile.TemporaryDirectory() as raw:
                got, _provider, _warn = highlights.analyze_transcript(
                    transcript, Path(raw), {"provider": "claude"}, lambda p, m: None)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[1]["rejected"]["stage"], "repair")


class EditorialAnalysisTests(unittest.TestCase):
    def test_provider_runs_compose_clips_with_progress_in_analysis_range(self) -> None:
        callbacks: list[tuple[int, str]] = []
        transcript = word_transcript()

        def fake_compose(tr, work_dir, options, *, provider=None, progress=None, **_):
            self.assertIs(tr, transcript)
            self.assertEqual(provider, "claude")
            progress(0, "start")
            progress(100, "end")
            return [editorial_candidate(0, 40)]

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(highlights.llm, "resolve_provider", return_value="claude"), \
                mock.patch.object(highlights.editorial, "compose_clips", side_effect=fake_compose):
            candidates, provider, warning = highlights.analyze_transcript(
                transcript, Path(temp_dir), {"provider": "auto", "max_clips": 3},
                lambda progress, message: callbacks.append((progress, message)),
            )
        self.assertEqual(provider, "claude")
        self.assertIsNone(warning)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(all(72 <= progress <= 94 for progress, _ in callbacks))
        self.assertEqual(callbacks[-1][0], 94)

    def test_transcript_without_words_is_a_clear_error(self) -> None:
        transcript = {"words": [], "segments": [{"start": 0, "end": 5, "text": "x"}]}
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(highlights.llm, "resolve_provider", return_value="codex"):
            with self.assertRaisesRegex(highlights.PipelineError, "пословных"):
                highlights.analyze_transcript(transcript, Path(temp_dir), {"provider": "codex"}, lambda *_a: None)


class NormalizeCandidatesTests(unittest.TestCase):
    def test_new_contract_preserved_duration_from_assembly_and_sorted(self) -> None:
        transcript = word_transcript()
        weak = editorial_candidate(0, 40, score=50)
        strong = editorial_candidate(10, 60, score=90)
        strong["segments"].append({"start": 70.0, "end": 80.0, "role": "payoff"})
        strong["cuts"] = [{"start": 20.0, "end": 22.0, "reason": "filler"}]
        out = highlights.normalize_candidates(
            [weak, strong], 100.0, {"min_duration": 30, "max_duration": 90, "max_clips": 8},
            transcript["words"],
        )
        self.assertEqual([c["id"] for c in out], ["c001", "c002"])
        self.assertEqual([c["score"] for c in out], [90, 50])
        top = out[0]
        self.assertEqual(len(top["segments"]), 2)
        self.assertEqual(top["cuts"][0]["reason"], "filler")
        self.assertEqual(top["critic"], {"verdict": "pass"})
        self.assertEqual(top["rationale"], "r")
        self.assertEqual((top["start"], top["end"]), (10.0, 80.0))
        self.assertLess(top["duration"], 60.0)  # 50 + 10 минус вырез 2 с, без пауз
        self.assertGreater(top["duration"], 50.0)

    def test_duration_window_and_max_clips(self) -> None:
        words = word_transcript(400)["words"]
        cands = [editorial_candidate(0, 5), editorial_candidate(10, 50, 60), editorial_candidate(60, 100, 70)]
        out = highlights.normalize_candidates(
            cands, 200.0, {"min_duration": 30, "max_duration": 90, "max_clips": 1}, words)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["score"], 70)

    def test_final_window_is_085_min_to_max_strict(self) -> None:
        words = word_transcript(400)["words"]
        opts = {"min_duration": 30, "max_duration": 90, "max_clips": 8}
        under = editorial_candidate(0, 25)    # 25 с < 25.5
        edge = editorial_candidate(40, 66)    # 26 с >= 25.5
        tolerated = editorial_candidate(0, 90)    # ровно max
        over = editorial_candidate(0, 95)         # > 90 с: допуска больше нет
        out = highlights.normalize_candidates([under, edge, over, tolerated], 200.0, opts, words)
        self.assertEqual(sorted(round(c["duration"]) for c in out), [26, 90])

    def test_legacy_candidate_without_segments(self) -> None:
        legacy = {"start": 10, "end": 50, "title": "old", "score": 70}
        out = highlights.normalize_candidates([legacy], 100.0, {"min_duration": 30, "max_duration": 90})
        self.assertEqual(len(out[0]["segments"]), 1)
        self.assertEqual(out[0]["duration"], 40.0)
        self.assertEqual(out[0]["cuts"], [])
        playback = highlights.candidate_playback({"start": 1, "end": 2})
        self.assertEqual(playback["segments"][0]["start"], 1.0)


class ProviderFailureTests(unittest.TestCase):
    def test_ai_failure_never_falls_back_to_heuristic(self) -> None:
        transcript = word_transcript()
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(highlights.llm, "resolve_provider", return_value="claude"), \
                mock.patch.object(
                    highlights.editorial, "compose_clips", side_effect=highlights.llm.LLMError("сбой"),
                ), \
                mock.patch.object(highlights, "heuristic_highlights") as heuristic:
            with self.assertRaises(highlights.PipelineError):
                highlights.analyze_transcript(
                    transcript, Path(temp_dir), {"provider": "auto"}, lambda *_args: None,
                )
        heuristic.assert_not_called()

    def test_no_ready_provider_raises_russian_error(self) -> None:
        transcript = word_transcript()
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(highlights.llm, "provider_status", return_value={
                    "claude": {"installed": False, "ready": False, "model": "opus"},
                    "codex": {"installed": False, "ready": False, "model": None},
                }):
            with self.assertRaisesRegex(highlights.llm.LLMError, "Нет доступного ИИ"):
                highlights.analyze_transcript(
                    transcript, Path(temp_dir), {"provider": "auto"}, lambda *_args: None,
                )

    def test_explicit_heuristic_runs_and_returns_new_contract(self) -> None:
        transcript = {
            "language": "ru", "words": [],
            "segments": [{"start": i * 10.0, "end": i * 10.0 + 10, "text": f"важно почему {i}?"} for i in range(12)],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            candidates, provider, warning = highlights.analyze_transcript(
                transcript, Path(temp_dir), {"provider": "heuristic"}, lambda *_args: None,
            )
        self.assertEqual(provider, "heuristic")
        self.assertTrue(candidates)
        self.assertEqual(len(candidates[0]["segments"]), 1)
        self.assertEqual(candidates[0]["cuts"], [])


class RetryAnalysisStateTests(unittest.TestCase):
    def retryable_project(self) -> dict:
        return {
            "id": "h123456789ab",
            "status": "failed",
            "progress": 100,
            "phase": "analysis",
            "phase_progress": 100,
            "input": "input/source.mp4",
            "transcript": "work/transcript.json",
            "source": {"duration": 900.0},
            "options": {"provider": "codex", "model": "base"},
            "provider_used": None,
            "warning": "old warning",
            "error": "provider unavailable",
            "error_stage": "analysis",
            "candidates": [{"id": "c001"}],
            "paused": True,
        }

    def test_retry_requires_reusable_transcript_and_analysis_stage(self) -> None:
        project = self.retryable_project()
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            transcript_path = project_dir / project["transcript"]
            transcript_path.parent.mkdir(parents=True)
            transcript_path.write_text("{}", encoding="utf-8")

            self.assertTrue(highlights.can_retry_analysis(project, project_dir))
            project["error_stage"] = "transcription"
            self.assertFalse(highlights.can_retry_analysis(project, project_dir))
            project["error_stage"] = "analysis"
            transcript_path.unlink()
            self.assertFalse(highlights.can_retry_analysis(project, project_dir))

    def test_prepare_retry_preserves_source_and_transcript_for_persistence(self) -> None:
        project = self.retryable_project()
        preserved = {
            key: json.loads(json.dumps(project[key]))
            for key in ("input", "transcript", "source", "options")
        }

        prepared = highlights.prepare_analysis_retry(project)
        persisted = json.loads(json.dumps(prepared))

        self.assertIs(prepared, project)
        for key, value in preserved.items():
            self.assertEqual(persisted[key], value)
        self.assertEqual(persisted["status"], "queued")
        self.assertEqual(persisted["progress"], 58)
        self.assertEqual(persisted["phase"], "analysis")
        self.assertEqual(persisted["phase_progress"], 0)
        self.assertEqual(persisted["candidates"], [])
        self.assertIsNone(persisted["error"])
        self.assertIsNone(persisted["error_stage"])
        self.assertFalse(persisted["paused"])


class SelectedClipTranscriptTests(unittest.TestCase):
    def test_missing_global_words_triggers_accurate_clip_transcript_before_render(self) -> None:
        events: list[str] = []
        exact_transcript = {
            "language": "ru",
            "language_probability": 1.0,
            "words": [{"text": "точно", "start": 0.1, "end": 0.5, "probability": 0.99}],
            "segments": [{"start": 0.0, "end": 1.0, "text": "точно"}],
            "text": "точно",
        }

        def fake_assemble(*args, **kwargs) -> dict:
            events.append("assemble")
            return {"duration": 10.0, "pieces": 1}

        def fake_transcribe(*args, **kwargs) -> dict:
            events.append("accurate_transcript")
            return exact_transcript

        def fake_render(*args, **kwargs) -> dict:
            events.append("render")
            self.assertIs(args[4], exact_transcript)
            return {"ok": True}

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(highlights.assembly, "assemble_clip", side_effect=fake_assemble), \
                mock.patch.object(highlights, "transcribe_video", side_effect=fake_transcribe) as transcriber, \
                mock.patch.object(highlights, "process_video_with_transcript", side_effect=fake_render):
            work_dir = Path(temp_dir) / "clip"
            result = highlights.render_highlight(
                Path("source.mp4"),
                {
                    "language": "ru",
                    "words": [],
                    "segments": [{"start": 10.0, "end": 20.0, "text": "coarse"}],
                },
                {"start": 10.0, "end": 20.0},
                Path("result.mp4"),
                work_dir,
                {"project_mode": "podcast", "language": "auto", "model": "small"},
                lambda progress, message: None,
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(events, ["assemble", "accurate_transcript", "render"])
        transcriber.assert_called_once()
        args, kwargs = transcriber.call_args
        self.assertEqual(args[0], work_dir / "horizontal.mp4")
        self.assertEqual(args[1], work_dir / "accurate_transcription")
        self.assertEqual(args[2], "ru")
        self.assertEqual(args[3], "small")
        self.assertEqual(kwargs["mode"], "accurate")

    def test_multi_piece_clip_is_assembled_and_transcript_remapped(self) -> None:
        words = [
            {"text": "а", "start": 100.0, "end": 100.4, "probability": 0.9},
            {"text": "ну", "start": 100.5, "end": 100.8, "probability": 0.9},
            {"text": "б", "start": 101.0, "end": 101.4, "probability": 0.9},
            {"text": "в", "start": 10.0, "end": 10.4, "probability": 0.9},
            {"text": "г", "start": 10.5, "end": 10.9, "probability": 0.9},
        ]
        words.sort(key=lambda w: w["start"])
        transcript = {"language": "ru", "words": words, "segments": [{"start": 10.0, "end": 101.4, "text": "x"}]}
        candidate = {
            "start": 10.0, "end": 101.4,
            "segments": [
                {"start": 100.0, "end": 101.4, "role": "hook"},
                {"start": 10.0, "end": 10.9, "role": "payoff"},
            ],
            "cuts": [{"start": 100.5, "end": 100.8, "reason": "filler"}],
        }
        captured = {}

        def fake_assemble(ffmpeg, source, intervals, destination, **kwargs):
            captured["intervals"] = list(intervals)
            return {"duration": 2.0, "pieces": len(intervals)}

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(highlights.assembly, "assemble_clip", side_effect=fake_assemble), \
                mock.patch.object(highlights, "transcribe_video") as transcriber, \
                mock.patch.object(
                    highlights, "process_video_with_transcript", return_value={"ok": True},
                ) as renderer:
            highlights.render_highlight(
                Path("source.mp4"), transcript, candidate, Path("result.mp4"), Path(temp_dir),
                {"project_mode": "podcast", "language": "ru", "model": "base"},
                lambda progress, message: None,
            )
        transcriber.assert_not_called()
        # хук (после вырезания «ну» — два интервала), затем кусок из начала записи
        self.assertGreaterEqual(len(captured["intervals"]), 2)
        self.assertGreater(captured["intervals"][0][0], 99.0)
        self.assertLess(captured["intervals"][-1][1], 12.0)
        remapped = renderer.call_args.args[4]
        self.assertEqual([w["text"] for w in remapped["words"]], ["а", "б", "в", "г"])
        self.assertLess(remapped["words"][-1]["end"], 3.0)  # уже на новой временной шкале

    def test_legacy_candidate_without_segments_still_renders(self) -> None:
        global_transcript = {
            "language": "ru",
            "words": [{"text": "готово", "start": 10.1, "end": 10.5, "probability": 0.9}],
            "segments": [{"start": 10.0, "end": 11.0, "text": "готово"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(
                    highlights.assembly, "assemble_clip", return_value={"duration": 1.0, "pieces": 1},
                ) as assembler, \
                mock.patch.object(highlights, "transcribe_video") as transcriber, \
                mock.patch.object(
                    highlights, "process_video_with_transcript", return_value={"ok": True},
                ) as renderer:
            highlights.render_highlight(
                Path("source.mp4"), global_transcript,
                {"start": 10.0, "end": 11.0}, Path("result.mp4"), Path(temp_dir),
                {"project_mode": "podcast", "language": "ru", "model": "base"},
                lambda progress, message: None,
            )

        transcriber.assert_not_called()
        assembler.assert_called_once()
        rendered_transcript = renderer.call_args.args[4]
        self.assertAlmostEqual(rendered_transcript["words"][0]["start"], 0.05, delta=0.06)


class SourceClipExportTests(unittest.TestCase):
    def test_source_clip_uses_stream_copy_without_filters_or_transcoding(self) -> None:
        completed = mock.Mock(returncode=0, stderr=b"")
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(highlights.subprocess, "run", return_value=completed) as run:
            destination = Path(temp_dir) / "fragment.mkv"
            highlights.extract_source_clip(
                Path("source.mkv"), destination, 12.345, 45.678,
            )

        command = run.call_args.args[0]
        self.assertIn(["-c", "copy"], [command[index:index + 2] for index in range(len(command) - 1)])
        self.assertIn(["-ss", "12.345"], [command[index:index + 2] for index in range(len(command) - 1)])
        self.assertIn(["-t", "33.333"], [command[index:index + 2] for index in range(len(command) - 1)])
        self.assertNotIn("-vf", command)
        self.assertNotIn("-filter_complex", command)
        self.assertNotIn("libx264", command)
        self.assertEqual(command[-1], str(destination))


if __name__ == "__main__":
    unittest.main()
