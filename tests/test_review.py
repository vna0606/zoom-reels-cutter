from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from reel_factory import assembly, editorial, highlights, llm, review
from tests.test_editorial import make_transcript

try:
    from reel_factory.pipeline import FFMPEG
except Exception:  # pragma: no cover
    FFMPEG = None


def synthetic_video(path: Path, seconds: int = 10, size: str = "360x640") -> None:
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=16000:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True)


def fake_review(**over):
    base = {"verdict": "publish", "scores": {"hook": 8, "idea_disclosure": 8, "flow": 8, "audio": 8},
            "issues": [], "summary": "ок", "caption": "Подпись"}
    base.update(over)
    return base


class SplicesAndFrameTimesTest(unittest.TestCase):
    def test_splice_times_follow_assembled_timeline_and_kind(self):
        ivs = [(0.0, 3.0), (10.0, 13.0), (13.2, 17.0), (5.0, 6.0)]
        sp = review.splice_points(ivs)
        self.assertEqual([s["t"] for s in sp], [3.0, 6.0, 9.8])
        self.assertEqual([s["kind"] for s in sp], ["seam", "cut", "seam"])   # назад по записи — тоже стык
        self.assertEqual(review.splice_points(ivs, speed=2.0)[0]["t"], 1.5)

    def test_frame_times_grid_plus_two_frames_per_splice_and_cap(self):
        splices = review.splice_points([(0, 3), (10, 13), (13.2, 17)])
        times = review.frame_times(10.0, splices)
        ts = [f["t"] for f in times]
        self.assertEqual(ts, sorted(ts))
        spliced = [(f["splice"], f["side"]) for f in times if f["splice"] is not None]
        self.assertIn((1, "-"), spliced)
        self.assertIn((1, "+"), spliced)
        self.assertTrue(any(abs(f["t"] - 2.7) < 0.01 for f in times))
        self.assertTrue(any(abs(f["t"] - 3.3) < 0.01 for f in times))
        self.assertTrue(any(abs(f["t"] - 4.0) < 0.01 and f["splice"] is None for f in times))   # сетка раз в 2 с
        many = [{"index": k, "t": 1.0 + k * 1.1, "gap": 0.2, "kind": "cut"} for k in range(1, 60)]
        capped = review.frame_times(80.0, many, max_frames=30)
        self.assertLessEqual(len(capped), 30)


@unittest.skipUnless(FFMPEG, "нужен ffmpeg")
class FramesAndSheetsTest(unittest.TestCase):
    def test_extract_frames_and_contact_sheet_on_synthetic_video(self):
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "v.mp4"
            synthetic_video(video)
            self.assertAlmostEqual(review.media_duration(video), 10.0, delta=0.2)
            splices = review.splice_points([(0, 3), (10, 13), (13.2, 17)])
            frames = review.extract_frames(video, Path(raw) / "frames", review.frame_times(10.0, splices))
            self.assertGreaterEqual(len(frames), 9)
            self.assertTrue(all(Path(f["path"]).is_file() for f in frames))
            spans = []
            sheets = review.make_contact_sheets(frames, Path(raw) / "sheets", spans_out=spans)
            self.assertGreaterEqual(len(sheets), 1)
            self.assertEqual(len(spans), len(sheets))
            cv2 = review._cv2()
            for sheet in sheets:
                img = cv2.imread(str(sheet))
                self.assertIsNotNone(img)
                self.assertLessEqual(img.shape[1], 1600)
            # вертикальное видео: 5x2 тайлов, листов ceil(кадров/10)
            self.assertEqual(len(sheets), -(-len(frames) // 10))
            first = cv2.imread(str(sheets[0]))
            self.assertGreater(first.shape[0], first.shape[1] * 0.5)
            # подписи не пустые: плашка с таймкодом отличается от фона
            self.assertGreater(int(first[:20, :300].std()), 5)

    def test_horizontal_video_uses_4x3_grid(self):
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "h.mp4"
            synthetic_video(video, seconds=6, size="640x360")
            frames = review.extract_frames(video, Path(raw) / "f", review.frame_times(6.0, []))
            sheets = review.make_contact_sheets(frames, Path(raw) / "s")
            img = review._cv2().imread(str(sheets[0]))
            self.assertEqual(img.shape[1], 1600)
            self.assertLess(img.shape[0], 1000)

    def test_load_audio_returns_samples(self):
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "v.mp4"
            synthetic_video(video, seconds=3)
            samples, sr = review.load_audio(video)
            self.assertEqual(sr, 16000)
            self.assertAlmostEqual(len(samples) / sr, 3.0, delta=0.2)
            self.assertGreater(float(abs(samples).max()), 0.1)


class AudioChecksTest(unittest.TestCase):
    def tone(self, seconds, amp=0.3, sr=16000, f=300):
        np = review._np()
        t = np.arange(int(seconds * sr)) / sr
        return (amp * np.sin(2 * np.pi * f * t)).astype("float32")

    def test_loudness_jump_and_dip_at_splice_are_reported(self):
        np = review._np()
        sr = 16000
        samples = np.concatenate([self.tone(2, 0.3), self.tone(2, 0.01)])   # на стыке 2.0 с: -30 дБ
        splices = [{"index": 1, "t": 2.0, "gap": 8.0, "kind": "seam"}]
        words = [{"text": "а", "start": 1.5, "end": 1.95}, {"text": "б", "start": 2.05, "end": 2.5}]
        res = review.audio_findings(samples, sr, splices, words, 4.0)
        notes = " ".join(f["note"] for f in res["findings"])
        self.assertIn("скачок громкости речи на стыке 1", notes)
        smooth = review.audio_findings(self.tone(4, 0.3), sr, splices, words, 4.0)
        self.assertFalse([f for f in smooth["findings"] if "громкост" in f["note"]])
        self.assertEqual(len(smooth["splice_levels"]), 1)

    def test_silence_gap_over_06_and_abrupt_ending(self):
        np = review._np()
        sr = 16000
        samples = np.concatenate([self.tone(1, 0.3), np.zeros(int(1.0 * sr), dtype="float32"), self.tone(1.0, 0.3)])
        words = [{"text": "а", "start": 0.1, "end": 0.9}, {"text": "б", "start": 2.1, "end": 2.98}]
        res = review.audio_findings(samples, sr, [], words, 3.0)
        kinds = {f["kind"] for f in res["findings"]}
        self.assertIn("silence", kinds)
        self.assertIn("ending", kinds)   # речь и звук не затухают к последней секунде
        faded = np.concatenate([self.tone(1, 0.3), np.zeros(int(0.5 * sr), dtype="float32")])
        ok = review.audio_findings(faded, sr, [], [{"text": "а", "start": 0.1, "end": 0.9}], 1.5)
        self.assertNotIn("ending", {f["kind"] for f in ok["findings"]})

    def test_clipped_and_doubled_words_from_transcript_comparison(self):
        exp = [{"text": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in
               enumerate("мы пришли в компанию и увидели что здесь всё плохо".split())]
        # «увидели» срезано рядом со стыком на 2.5 с
        act = [w for w in exp if w["text"] != "увидели"]
        res = review.compare_transcripts(exp, act, [{"index": 1, "t": 2.5, "gap": 5.0, "kind": "seam"}])
        notes = " | ".join(f["note"] for f in res["findings"])
        self.assertIn("срезано слово «увидели»", notes)
        self.assertLess(res["match_ratio"], 1.0)
        # «что» задвоено на стыке
        doubled = exp[:6] + [dict(exp[6], start=exp[6]["start"] + 0.45)] + exp[6:]
        notes = " | ".join(f["note"] for f in review.compare_transcripts(
            exp, doubled, [{"index": 1, "t": 2.5, "gap": 5.0, "kind": "seam"}])["findings"])
        self.assertIn("задвоенное слово «что»", notes)
        clean = review.compare_transcripts(exp, exp, [{"index": 1, "t": 2.5, "gap": 5.0, "kind": "seam"}])
        self.assertEqual(clean["findings"], [])
        self.assertEqual(clean["match_ratio"], 1.0)
        # слово пропало далеко от стыка — это ошибка распознавания, не обрезка
        far = review.compare_transcripts(exp, act, [{"index": 1, "t": 40.0, "gap": 5.0, "kind": "seam"}])
        self.assertFalse([f for f in far["findings"] if f["kind"] == "clipped_word"])


class SchemaAndFixTest(unittest.TestCase):
    def test_schema_accepts_fake_review_and_rejects_bad_enum(self):
        llm.validate_schema(fake_review(suggested_fix={
            "description": "убрать хвост", "edits": [{"op": "drop_last_sentences", "count": 1}]}), review.REVIEW_SCHEMA)
        with self.assertRaises(ValueError):
            llm.validate_schema(fake_review(verdict="maybe"), review.REVIEW_SCHEMA)
        with self.assertRaises(ValueError):
            llm.validate_schema(fake_review(scores={"hook": 11, "idea_disclosure": 1, "flow": 1, "audio": 1}),
                                review.REVIEW_SCHEMA)

    def test_verdict_thresholds_from_scores(self):
        v = review.verdict_from_scores
        self.assertEqual(v({"hook": 8, "idea_disclosure": 7, "flow": 6, "audio": 9}), "publish")
        self.assertEqual(v({"hook": 8, "idea_disclosure": 5, "flow": 8, "audio": 9}), "needs_edit")
        self.assertEqual(v({"hook": 8, "idea_disclosure": 4, "flow": 8, "audio": 9}), "needs_edit")
        self.assertEqual(v({"hook": 8, "idea_disclosure": 8, "flow": 3, "audio": 9}), "reject")
        self.assertEqual(v({"hook": 3, "idea_disclosure": 8, "flow": 8, "audio": 9}), "reject")
        self.assertEqual(v({"hook": 8, "idea_disclosure": 8, "flow": 8, "audio": 3, "visual": 9}), "reject")
        # LLM-вердикт не важен: считаем по оценкам; для publish правка отбрасывается
        got = review._clean_review({"verdict": "reject", "scores": {"hook": 7, "idea_disclosure": 7, "flow": 7, "audio": 7},
                                    "suggested_fix": {"description": "d", "edits": [{"op": "drop_piece", "piece": 1}]}})
        self.assertEqual(got["verdict"], "publish")
        self.assertIsNone(got["suggested_fix"])

    def test_prompt_is_audio_only_with_duration_window(self):
        prompt = review.build_prompt({"title": "T", "format": "thesis", "thesis": "t"}, [], [], [], [], None, [], 120.0,
                                     window=(25.5, 90.0))
        self.assertIn("Картинку НЕ оцениваешь", prompt)
        self.assertIn("ДОПУСТИМАЯ ДЛИТЕЛЬНОСТЬ: 26–90 с", prompt)
        self.assertIn("ВНЕ ОКНА", prompt)

    def test_clean_review_normalizes_and_sorts(self):
        got = review._clean_review({"verdict": "needs_edit", "scores": {"hook": 7, "idea_disclosure": 6, "flow": 5, "audio": 4},
                                    "issues": [{"t": 9, "kind": "ending", "note": "обрыв"}, {"t": 1.5, "kind": "hook", "note": "слабо"}],
                                    "suggested_fix": {"description": "d", "edits": [
                                        {"op": "drop_last_sentences", "count": 9}, {"op": "bogus"}]}})
        self.assertEqual([i["t"] for i in got["issues"]], [1.5, 9.0])
        self.assertEqual(got["suggested_fix"]["edits"], [{"op": "drop_last_sentences", "piece": None, "count": 3}])

    def make_candidate(self, tr, pieces):
        sents = editorial.build_sentences(tr)
        words = tr["words"]
        segs = []
        for a, b, role in pieces:
            segs.append({"start": words[sents[a - 1]["w0"]]["start"], "end": words[sents[b - 1]["w1"]]["end"], "role": role,
                         "sentence_from": f"S{a:04d}", "sentence_to": f"S{b:04d}"})
        return {"format": "thesis", "segments": segs, "cuts": [], "title": "T", "thesis": "t"}

    def test_apply_suggested_fix_variants(self):
        tr, _ = make_transcript()
        cand = self.make_candidate(tr, [(2, 4, "hook"), (10, 13, "payoff")])
        opts = {"min_duration": 8, "max_duration": 60}
        fix = lambda *edits: {"description": "d", "edits": list(edits)}
        upd = review.apply_suggested_fix(cand, fix({"op": "drop_last_sentences", "count": 1}), tr, opts)
        self.assertEqual([(s["sentence_from"], s["sentence_to"]) for s in upd["segments"]], [("S0002", "S0004"), ("S0010", "S0012")])
        self.assertEqual(upd["segments"][-1]["role"], "payoff")
        upd = review.apply_suggested_fix(cand, fix({"op": "extend_piece_start", "piece": 2, "count": 2}), tr, opts)
        self.assertEqual(upd["segments"][1]["sentence_from"], "S0008")
        upd = review.apply_suggested_fix(cand, fix({"op": "drop_first_sentences", "count": 1}), tr, opts)
        self.assertEqual(upd["segments"][0]["sentence_from"], "S0003")
        upd = review.apply_suggested_fix(cand, fix({"op": "drop_piece", "piece": 1}), tr, opts)
        self.assertEqual(len(upd["segments"]), 1)
        self.assertAlmostEqual(upd["duration"], assembly.assembled_duration(
            assembly.keep_intervals({"segments": upd["segments"], "cuts": upd["cuts"]}, tr["words"])), places=3)
        # несуществующий кусок, слишком короткий результат, нет id — правка невыразима
        self.assertIsNone(review.apply_suggested_fix(cand, fix({"op": "drop_piece", "piece": 5}), tr, opts))
        self.assertIsNone(review.apply_suggested_fix(cand, fix({"op": "drop_piece", "piece": 1}, {"op": "drop_piece", "piece": 2}), tr, opts))
        self.assertIsNone(review.apply_suggested_fix(cand, fix({"op": "drop_last_sentences", "count": 3}),
                                                      tr, {"min_duration": 60, "max_duration": 90}))
        bare = dict(cand, segments=[{"start": 0.0, "end": 9.0, "role": "point"}])
        self.assertIsNone(review.apply_suggested_fix(bare, fix({"op": "drop_last_sentences", "count": 1}), tr, opts))
        self.assertIsNone(review.apply_suggested_fix(cand, None, tr, opts))

    def test_cuts_outside_new_segments_are_dropped(self):
        tr, idx = make_transcript()
        cand = self.make_candidate(tr, [(2, 6, "payoff")])
        w = tr["words"]
        cand["cuts"] = [{"start": w[idx[(2, 1)]]["start"], "end": w[idx[(2, 1)]]["end"], "reason": "filler"},
                        {"start": w[idx[(6, 1)]]["start"], "end": w[idx[(6, 1)]]["end"], "reason": "filler"}]
        upd = review.apply_suggested_fix(cand, {"description": "d", "edits": [{"op": "drop_last_sentences", "count": 1}]},
                                         tr, {"min_duration": 8, "max_duration": 60})
        self.assertEqual(len(upd["cuts"]), 1)


@unittest.skipUnless(FFMPEG, "нужен ffmpeg")
class ReviewVideoTest(unittest.TestCase):
    def test_review_video_end_to_end_with_fakes(self):
        tr, _ = make_transcript(6)
        cand = {"format": "thesis", "title": "Клип", "thesis": "Тезис",
                "segments": [{"start": 0.0, "end": 10.0, "role": "point", "sentence_from": "S0001", "sentence_to": "S0003"}],
                "cuts": []}
        sent = {}

        def fake_llm(prompt, schema, work_dir, *, task, provider=None, effort="high", timeout=1800, images=None):
            sent.update(prompt=prompt, task=task, effort=effort, images=images)
            return fake_review(issues=[{"t": 3.0, "kind": "flow", "note": "стык"}])

        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "out.mp4"
            synthetic_video(video)
            acts = [{"text": "Предложение", "start": 0.2, "end": 0.8}, {"text": "номер", "start": 0.9, "end": 1.2}]
            report = review.review_video(
                video, candidate=cand, transcript=tr, work_dir=Path(raw) / "rv", options={"model": "small"},
                intervals=[(0.0, 3.0), (10.0, 13.0), (13.2, 17.0)],
                llm=fake_llm, transcribe=lambda v: {"words": acts})
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["verdict"], "publish")
            self.assertEqual(set(report["scores"]), set(review.SCORE_KEYS))
            self.assertEqual(report["issues"][0]["t"], 3.0)
            self.assertEqual(sent["task"], "video_review")
            self.assertEqual(sent["effort"], "high")
            self.assertFalse(sent["images"])          # картинку не оцениваем: кадры и листы не готовятся
            self.assertEqual(report["frames"], 0)
            self.assertNotIn("visual", report["scores"])
            self.assertNotIn("КОНТАКТНЫЕ ЛИСТЫ", sent["prompt"])
            self.assertNotIn("visual", sent["prompt"].split("ЧТО ОЦЕНИТЬ")[1].split("ВЕРДИКТ")[0])
            self.assertIn("СТЫКИ СКЛЕЙКИ", sent["prompt"])
            self.assertIn("РАСШИФРОВКА ЗВУКА ГОТОВОГО РОЛИКА", sent["prompt"])
            self.assertIn("[00:00.2] Предложение номер", sent["prompt"])
            self.assertIn("Кусок 1 [S0001–S0003]", sent["prompt"])
            self.assertEqual(len(report["splices"]), 2)
            self.assertTrue((Path(raw) / "rv" / "video_review.json").is_file())
            # опция review_visual=True возвращает кадры, листы и оценку visual
            vis = review.review_video(
                video, candidate=cand, transcript=tr, work_dir=Path(raw) / "rv3", options={"review_visual": True},
                intervals=[(0.0, 3.0), (10.0, 13.0)],
                llm=lambda prompt, schema, wd, **k: dict(fake_review(), scores=dict(fake_review()["scores"], visual=8)) or sent.update(imgs=k.get("images")),
                transcribe=lambda v: {"words": acts})
            self.assertTrue(vis["sheets"] and vis["frames"] > 0)
            self.assertIn("visual", vis["scores"])
            # сбой ИИ не роняет отсмотр: автонаходки остаются, статус error
            def boom(*a, **k):
                raise RuntimeError("нет сети")
            failed = review.review_video(video, candidate=cand, transcript=tr, work_dir=Path(raw) / "rv2",
                                         options={}, intervals=[(0.0, 10.0)], llm=boom, transcribe=lambda v: {"words": acts})
            self.assertEqual(failed["status"], "error")
            self.assertIsNone(failed["verdict"])


class AutoFixFlowTest(unittest.TestCase):
    """render_highlight_reviewed: не более одной авто-правки, исправленная версия принимается только если не хуже."""

    def setUp(self):
        self.tr, _ = make_transcript()
        sents = editorial.build_sentences(self.tr)
        w = self.tr["words"]
        seg = lambda a, b, role: {"start": w[sents[a - 1]["w0"]]["start"], "end": w[sents[b - 1]["w1"]]["end"],
                                  "role": role, "sentence_from": f"S{a:04d}", "sentence_to": f"S{b:04d}"}
        self.cand = {"format": "thesis", "title": "T", "thesis": "t", "cuts": [],
                     "segments": [seg(2, 4, "hook"), seg(10, 13, "payoff")]}
        self.opts = {"video_review": True, "min_duration": 8, "max_duration": 60}
        self.rendered = []

    def renderer(self, source, transcript, candidate, output, work_dir, options, callback):
        self.rendered.append((Path(output).name, [s["sentence_to"] for s in candidate["segments"]]))
        Path(output).write_bytes(b"video-" + str(len(self.rendered)).encode())
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        callback(100, "ok")
        return {"visual_engine": "x"}

    def run_flow(self, reviews, opts=None):
        calls = []

        def reviewer(output, *, candidate, transcript, work_dir, options, task="video_review", callback=None):
            calls.append((task, [s["sentence_to"] for s in candidate["segments"]]))
            return reviews[len(calls) - 1]

        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "c001.mp4"
            result = highlights.render_highlight_reviewed(
                Path("src.mp4"), self.tr, self.cand, out, Path(raw) / "w", dict(self.opts, **(opts or {})),
                lambda p, m: None, renderer=self.renderer, reviewer=reviewer)
            content = out.read_bytes()
            leftovers = [p.name for p in Path(raw).glob("*.autofix*")]
        return result, calls, content, leftovers

    def needs_edit(self, **kw):
        return dict(fake_review(verdict="needs_edit", scores={"hook": 6, "idea_disclosure": 5, "flow": 6, "audio": 8},
                                suggested_fix={"description": "убрать хвост", "edits": [{"op": "drop_last_sentences", "count": 1}]}),
                    status="ok", **kw)

    def test_review_disabled_just_renders(self):
        result, calls, content, _ = self.run_flow([], {"video_review": False})
        self.assertEqual(calls, [])
        self.assertNotIn("review", result)
        self.assertEqual(len(self.rendered), 1)

    def test_publish_verdict_no_fix(self):
        result, calls, _, _ = self.run_flow([dict(fake_review(), status="ok")])
        self.assertEqual(result["review"]["verdict"], "publish")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("auto_fix", result)
        self.assertEqual(len(self.rendered), 1)

    def test_needs_edit_applies_one_fix_rerenders_and_keeps_better_version(self):
        better = dict(fake_review(), status="ok")
        result, calls, content, leftovers = self.run_flow([self.needs_edit(), better])
        self.assertEqual([c[0] for c in calls], ["video_review", "video_review_2"])
        self.assertEqual(len(self.rendered), 2)             # ровно один перерендер
        self.assertTrue(result["auto_fix"]["kept"])
        self.assertEqual(result["review"]["verdict"], "publish")
        self.assertEqual(result["review_before_fix"]["verdict"], "needs_edit")
        self.assertEqual([s["sentence_to"] for s in result["candidate_update"]["segments"]], ["S0004", "S0012"])
        self.assertEqual(content, b"video-2")               # файл заменён исправленной версией
        self.assertEqual(leftovers, [])

    def test_fix_that_is_worse_is_discarded_and_second_needs_edit_is_not_fixed_again(self):
        worse = dict(fake_review(verdict="needs_edit", scores={"hook": 2, "idea_disclosure": 2, "flow": 2, "audio": 2},
                                 suggested_fix={"description": "ещё", "edits": [{"op": "drop_last_sentences", "count": 1}]}),
                     status="ok")
        result, calls, content, leftovers = self.run_flow([self.needs_edit(), worse])
        self.assertEqual(len(calls), 2)                      # нет третьей итерации
        self.assertEqual(len(self.rendered), 2)
        self.assertFalse(result["auto_fix"]["kept"])
        self.assertNotIn("candidate_update", result)
        self.assertEqual(content, b"video-1")                # осталась исходная версия
        self.assertEqual(leftovers, [])

    def test_inexpressible_fix_is_recorded_and_not_applied(self):
        review1 = dict(self.needs_edit(), suggested_fix=None)
        result, calls, _, _ = self.run_flow([review1])
        self.assertEqual(len(self.rendered), 1)
        self.assertFalse(result["auto_fix"]["applied"])
        bad = dict(self.needs_edit(), suggested_fix={"description": "x", "edits": [{"op": "drop_piece", "piece": 9}]})
        result, calls, _, _ = self.run_flow([bad])
        self.assertFalse(result["auto_fix"]["applied"])

    def test_autofix_can_be_switched_off(self):
        result, calls, _, _ = self.run_flow([self.needs_edit()], {"video_review_autofix": False})
        self.assertEqual(len(self.rendered), 1)
        self.assertNotIn("auto_fix", result)

    def test_reviewer_crash_keeps_the_rendered_clip(self):
        def reviewer(*a, **k):
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "c.mp4"
            result = highlights.render_highlight_reviewed(
                Path("s.mp4"), self.tr, self.cand, out, Path(raw) / "w", self.opts, lambda p, m: None,
                renderer=self.renderer, reviewer=reviewer)
            self.assertTrue(out.is_file())
        self.assertEqual(result["review"]["status"], "error")


if __name__ == "__main__":
    unittest.main()
