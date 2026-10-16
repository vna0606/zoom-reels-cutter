from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from reel_factory import assembly

ROOT = Path(__file__).resolve().parents[1]


def find_ffmpeg():
    for local in (ROOT / ".tools" / "ffmpeg" / "ffmpeg", ROOT / ".tools" / "dl" / "t" / "ffmpeg"):
        if local.is_file():
            return str(local)
    return shutil.which("ffmpeg")


def w(text, start, end):
    return {"text": text, "start": start, "end": end, "probability": 0.9}


WORDS = [
    w("а", 0.0, 0.3),
    w("э-э", 0.35, 0.6),   # филлер
    w("вот", 0.65, 0.9),
    w("дело", 0.95, 1.3),
    w("в", 3.3, 3.5),       # пауза 2 с перед этим словом
    w("том", 3.55, 3.8),
    w("далее", 5.0, 5.4),
    w("идёт", 5.45, 5.8),
]


def cand(segments, cuts=()):
    return {"segments": [dict(start=s, end=e, role="point", sentence_from="S1", sentence_to="S1") for s, e in segments],
            "cuts": [dict(start=s, end=e, reason="filler") for s, e in cuts]}


class KeepIntervalsTest(unittest.TestCase):
    def test_tail_pad_adds_room_after_last_word_without_touching_next_word(self):
        c = cand([(5.0, 5.8)])
        plain = assembly.keep_intervals(c, WORDS)
        padded = assembly.keep_intervals(c, WORDS, tail_pad=0.3)
        self.assertAlmostEqual(padded[-1][1] - plain[-1][1], 0.25, delta=0.06)   # до +0.3 с от конца слова
        self.assertEqual(padded[0][0], plain[0][0])
        # следующее слово близко: хвост ограничен его началом
        near = [w("раз", 0.0, 0.5), w("два", 0.6, 1.0)]
        tight = assembly.keep_intervals(cand([(0.0, 0.5)]), near, tail_pad=0.3)
        self.assertLessEqual(tight[-1][1], 0.6 - 0.04 + 1e-9)
        self.assertEqual(assembly.keep_intervals(c, WORDS), plain)   # по умолчанию без хвоста

    def test_filler_cut_in_middle_and_pause_compression(self):
        c = cand([(0.0, 3.8)], cuts=[(0.35, 0.6)])
        iv = assembly.keep_intervals(c, WORDS)
        self.assertEqual(len(iv), 3)
        # слово-филлер не попадает ни в один интервал
        for s, e in iv:
            self.assertFalse(s < 0.6 and e > 0.35 and not (e <= 0.35 or s >= 0.6))
        self.assertLessEqual(iv[0][1], 0.35)
        self.assertGreaterEqual(iv[1][0], 0.6)
        # граница между «а» и «вот» лежит в паузах (середина), а не на словах
        self.assertGreater(iv[0][1], 0.3)
        self.assertLess(iv[1][0], 0.65)
        # длинная пауза (>= 0.9 с) 1.3..3.3 сжата до max_gap=0.35 с
        self.assertAlmostEqual(iv[2][0] - iv[1][1], 2.0 - 0.35, delta=0.01)
        self.assertGreaterEqual(iv[0][0], 0.0)

    def test_padding_never_enters_neighbouring_word(self):
        words = [w("раз", 0.0, 0.5), w("два", 0.52, 1.0), w("три", 1.02, 1.5)]
        iv = assembly.keep_intervals(cand([(0.52, 1.0)]), words, pad=0.2)
        self.assertEqual(len(iv), 1)
        self.assertGreaterEqual(iv[0][0], 0.5)
        self.assertLessEqual(iv[0][1], 1.02)
        self.assertLessEqual(iv[0][0], 0.52)
        self.assertGreaterEqual(iv[0][1], 1.0)

    def test_reordered_segments_keep_playback_order(self):
        c = cand([(5.0, 5.8), (0.0, 1.3)])
        iv = assembly.keep_intervals(c, WORDS)
        self.assertEqual(len(iv), 2)
        self.assertGreater(iv[0][0], 4.5)
        self.assertLess(iv[1][1], 2.0)
        self.assertEqual(c["segments"][0]["start"], 5.0)

    def test_no_words_fallback(self):
        c = cand([(10.0, 20.0)], cuts=[(12.0, 13.0)])
        iv = assembly.keep_intervals(c, [])
        self.assertEqual(iv, [(10.0, 12.0), (13.0, 20.0)])
        self.assertAlmostEqual(assembly.assembled_duration(iv), 9.0)
        self.assertEqual(assembly.source_spans(c), [(10.0, 20.0)])

    def test_short_pieces_dropped(self):
        words = [w("а", 0.0, 0.05), w("б", 5.0, 5.5)]
        iv = assembly.keep_intervals(cand([(0.0, 0.05), (5.0, 5.5)]), words, pad=0.0, min_piece=0.12)
        self.assertEqual(len(iv), 1)


class RemapTest(unittest.TestCase):
    def setUp(self):
        self.transcript = {
            "language": "ru",
            "words": WORDS,
            "segments": [{"start": 0.0, "end": 1.3, "text": "а э-э вот дело"},
                         {"start": 3.3, "end": 3.8, "text": "в том"},
                         {"start": 5.0, "end": 5.8, "text": "далее идёт"}],
            "text": "а э-э вот дело в том далее идёт",
        }
        self.cand = cand([(5.0, 5.8), (0.0, 3.8)], cuts=[(0.35, 0.6)])
        self.iv = assembly.keep_intervals(self.cand, WORDS)

    def test_remap(self):
        out = assembly.remap_transcript(self.transcript, self.iv)
        texts = [x["text"] for x in out["words"]]
        self.assertNotIn("э-э", texts)
        self.assertEqual(texts[:2], ["далее", "идёт"])
        self.assertEqual(texts[2:], ["а", "вот", "дело", "в", "том"])
        starts = [x["start"] for x in out["words"]]
        self.assertEqual(starts, sorted(starts))
        self.assertLess(starts[0], 0.15)
        total = assembly.assembled_duration(self.iv)
        self.assertLessEqual(out["words"][-1]["end"], total + 1e-6)
        self.assertGreater(out["words"][-1]["end"], total - 0.2)
        self.assertEqual(set(out), {"language", "words", "segments", "text"})
        self.assertTrue(out["segments"])
        self.assertEqual(" ".join(texts), out["text"])
        for s in out["segments"]:
            self.assertLessEqual(s["start"], s["end"])


@unittest.skipUnless(find_ffmpeg(), "ffmpeg не найден")
class AssembleClipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = find_ffmpeg()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = Path(cls.tmp.name) / "src.mp4"
        cmd = [cls.ffmpeg, "-y", "-loglevel", "error",
               "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=12",
               "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
               "-af", "volume='if(lt(mod(t,3),2),1,0)':eval=frame",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.src)]
        subprocess.run(cmd, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def probe(self, path):
        probe = shutil.which("ffprobe") or str(Path(self.ffmpeg).with_name("ffprobe"))
        res = subprocess.run([probe, "-v", "error", "-show_entries", "stream=codec_type:format=duration",
                              "-of", "json", str(path)], capture_output=True, text=True, check=True)
        return json.loads(res.stdout)

    def test_three_out_of_order_intervals(self):
        dest = Path(self.tmp.name) / "out.mp4"
        iv = [(8.0, 10.0), (1.0, 3.5), (5.0, 6.0)]
        res = assembly.assemble_clip(self.ffmpeg, self.src, iv, dest)
        self.assertEqual(res["pieces"], 3)
        self.assertAlmostEqual(res["duration"], 5.5, delta=0.1)
        info = self.probe(dest)
        types = sorted(s["codec_type"] for s in info["streams"])
        self.assertEqual(types, ["audio", "video"])
        self.assertAlmostEqual(float(info["format"]["duration"]), 5.5, delta=0.1)

    def test_single_interval(self):
        dest = Path(self.tmp.name) / "one.mp4"
        res = assembly.assemble_clip(self.ffmpeg, self.src, [(2.0, 4.0)], dest)
        self.assertAlmostEqual(res["duration"], 2.0, delta=0.1)

    def test_many_intervals(self):
        dest = Path(self.tmp.name) / "many.mp4"
        iv = [(0.2 + 0.19 * i, 0.2 + 0.19 * i + 0.2) for i in range(60)]
        calls = []
        res = assembly.assemble_clip(self.ffmpeg, self.src, iv, dest, callback=lambda p, m: calls.append(p))
        self.assertEqual(res["pieces"], 60)
        self.assertAlmostEqual(res["duration"], 60 * 0.2, delta=0.2)
        self.assertEqual(calls[-1], 100)

    def _captured_script(self, **kwargs):
        from unittest import mock
        seen = {}
        real = assembly.build_filter_script

        def spy(*a, **kw):
            seen.update(kw)
            seen["script"] = real(*a, **kw)[1]
            return real(*a, **kw)
        dest = Path(self.tmp.name) / "mode.mp4"
        with mock.patch.object(assembly, "build_filter_script", spy):
            assembly.assemble_clip(self.ffmpeg, self.src, [(1.0, 3.0), (6.0, 8.0)], dest,
                                   gains=[0.0, 6.0], **kwargs)
        return seen

    def test_raw_mode_has_no_gain_or_normalization_filters(self):
        seen = self._captured_script()
        script = seen["script"]
        for bad in ("volume", "loudnorm", "dynaudnorm"):
            self.assertNotIn(bad, script)
        self.assertIn("afade=t=in:st=0:d=0.0150", script)
        self.assertIn("[outv][outa]", script)

    def test_leveled_mode_still_applies_gain_and_loudnorm(self):
        seen = self._captured_script(audio_mode="leveled")
        self.assertIn("volume=6.00dB", seen["script"])
        self.assertIn("loudnorm", seen["script"])

    def test_error_is_russian(self):
        with self.assertRaises(RuntimeError) as ctx:
            assembly.assemble_clip(self.ffmpeg, Path(self.tmp.name) / "missing.mp4", [(0, 1)],
                                   Path(self.tmp.name) / "x.mp4")
        self.assertIn("Не удалось", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class AudioPolishTest(unittest.TestCase):
    def levels(self, spec):
        """spec: [(секунды, дБ)] -> уровни окон по 10 мс."""
        out = []
        for seconds, db in spec:
            out += [db] * int(round(seconds / assembly.FRAME_SEC))
        return out

    def test_silent_edges_longer_than_015_are_trimmed(self):
        lv = self.levels([(0.5, -80), (2.0, -20), (0.6, -80)])
        s, e = assembly.trim_edge_silence(lv, 100.0, (100.0, 103.1), -20.0)
        self.assertAlmostEqual(s, 100.0 + 0.5 - assembly.LEAD_KEEP, places=2)
        self.assertAlmostEqual(e, 100.0 + 2.5 + assembly.TAIL_KEEP, places=2)
        # итоговые тихие края не длиннее допуска
        self.assertLessEqual(0.5 - (s - 100.0), 0.15)

    def test_short_silence_is_left_alone(self):
        lv = self.levels([(0.1, -80), (2.0, -20), (0.1, -80)])
        self.assertEqual(assembly.trim_edge_silence(lv, 0.0, (0.0, 2.2), -20.0), (0.0, 2.2))

    def test_trailing_tail_of_foreign_word_before_pause_is_dropped(self):
        # 0.2 с обрывка чужого слова, 0.5 с тишины, дальше речь: обрывок + пауза отрезаются
        lv = self.levels([(0.2, -22), (0.5, -80), (2.0, -20)])
        s, e = assembly.trim_edge_silence(lv, 0.0, (0.0, 2.7), -20.0)
        self.assertAlmostEqual(s, 0.7 - assembly.LEAD_KEEP, places=2)
        self.assertEqual(e, 2.7)

    def test_last_piece_keeps_tail_air(self):
        lv = self.levels([(2.0, -20), (0.6, -80)])
        self.assertEqual(assembly.trim_edge_silence(lv, 0.0, (0.0, 2.6), -20.0, trim_end=False)[1], 2.6)

    def test_piece_gains_towards_median_with_clamp_and_deadband(self):
        gains = assembly.piece_gains([-20.0, -31.0, -20.5, None, -45.0, -19.0])
        self.assertEqual(gains[0], 0.0)           # в пределах 1.5 дБ
        self.assertEqual(gains[1], 9.0)           # зажато
        self.assertEqual(gains[3], 0.0)
        self.assertEqual(gains[4], 9.0)
        self.assertEqual(assembly.piece_gains([]), [])

    def test_filter_has_piece_fades_gain_and_final_loudnorm(self):
        _, script = assembly.build_filter_script([(0.0, 2.0), (5.0, 8.0)], has_audio=True, crossfade=0.012,
                                                 normalize="loudnorm", gains=[0.0, 4.5])
        self.assertIn("afade=t=in", script)
        self.assertIn("volume=4.50dB", script)
        self.assertNotIn("volume=0.00dB", script)
        self.assertIn("concat=n=2:v=1:a=1[outv][cata]", script)
        self.assertIn("[cata]loudnorm=I=-16:TP=-1.5:LRA=11[outa]", script)
        _, plain = assembly.build_filter_script([(0.0, 2.0)], has_audio=True, crossfade=0.012)
        self.assertIn("[outv][outa]", plain)
        self.assertNotIn("loudnorm", plain)


class SnapToQuietTest(unittest.TestCase):
    def test_quietest_point_is_center_of_quietest_40ms(self):
        lv = [-20.0] * 100
        for i in range(40, 44):
            lv[i] = -70.0                     # 40 мс тишины на 0.40–0.44 с
        c = assembly.quietest_point(lv, 10.0, 10.2, 10.7)
        self.assertAlmostEqual(c, 10.0 + 0.42, places=2)

    def test_quietest_point_none_when_region_too_small(self):
        self.assertIsNone(assembly.quietest_point([-20.0] * 100, 0.0, 0.2, 0.22))

    def test_default_audio_mode_is_raw_with_15ms_fade(self):
        import inspect
        params = inspect.signature(assembly.assemble_clip).parameters
        self.assertEqual(params["audio_mode"].default, "raw")
        self.assertEqual(assembly.RAW_CROSSFADE, 0.015)
        self.assertEqual(assembly.normalize_audio_mode(None), "raw")
        self.assertEqual(assembly.normalize_audio_mode("LEVELED"), "leveled")
        self.assertEqual(assembly.normalize_audio_mode("junk"), "raw")


class SeamGainsTest(unittest.TestCase):
    def test_gains_equalize_level_across_each_seam(self):
        edges = [(-20.0, -20.0), (-30.0, -30.0), (-22.0, -22.0)]       # второй кусок тише на 10 дБ
        g = assembly.seam_gains(edges, deadband=0.0)
        levels = [(e[0] + gi, e[1] + gi) for e, gi in zip(edges, g)]
        self.assertAlmostEqual(levels[0][1], levels[1][0], places=1)
        self.assertAlmostEqual(levels[1][1], levels[2][0], places=1)

    def test_step_and_total_are_clamped_unknown_edges_keep_gain(self):
        g = assembly.seam_gains([(-10.0, -10.0), (-60.0, -60.0), (None, None), (-20.0, -20.0)], max_step=12.0,
                                max_total=15.0, deadband=0.0)
        self.assertTrue(all(abs(x) <= 15.0 for x in g))
        self.assertEqual(assembly.seam_gains([], ), [])
        self.assertEqual(assembly.seam_gains([(-20.0, -20.0)]), [0.0])
