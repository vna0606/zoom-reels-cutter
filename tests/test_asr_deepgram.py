from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest import mock

from reel_factory import asr_deepgram, pipeline

FAKE_KEY = "dg_test_FAKEKEY_1234567890abcdef"

FAKE = {
    "metadata": {"duration": 6.0},
    "results": {
        "channels": [{
            "detected_language": "ru", "language_confidence": 0.97,
            "alternatives": [{
                "transcript": "Привет, мир. Это тест.",
                "words": [
                    {"word": "привет", "punctuated_word": "Привет,", "start": 0.1, "end": 0.5, "confidence": 0.99},
                    {"word": "мир", "punctuated_word": "мир.", "start": 0.6, "end": 1.0, "confidence": 0.95},
                    {"word": "это", "punctuated_word": "Это", "start": 2.0, "end": 2.3, "confidence": 0.9},
                    {"word": "тест", "punctuated_word": "тест.", "start": 2.4, "end": 3.0, "confidence": 0.8},
                    {"word": "...", "punctuated_word": "...", "start": 3.1, "end": 3.2, "confidence": 0.5},
                ],
            }],
        }],
        "utterances": [
            {"start": 0.1, "end": 1.0, "transcript": "Привет, мир."},
            {"start": 2.0, "end": 3.0, "transcript": "Это тест."},
        ],
    },
}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(body.encode()))


class MappingTests(unittest.TestCase):
    def test_mapping_matches_transcript_format(self) -> None:
        out = asr_deepgram.map_response(FAKE, pipeline.clean_word, requested_language="ru")
        self.assertEqual(out["words"][0], {"text": "Привет", "raw": "Привет,", "start": 0.1, "end": 0.5,
                                           "probability": 0.99})
        self.assertEqual(len(out["words"]), 4)  # слово из одной пунктуации отброшено
        self.assertEqual(out["segments"][1], {"start": 2.0, "end": 3.0, "text": "Это тест."})
        self.assertEqual(out["language"], "ru")
        self.assertEqual(out["text"], "Привет, мир. Это тест.")

    def test_paragraph_fallback_and_detected_language(self) -> None:
        data = json.loads(json.dumps(FAKE))
        del data["results"]["utterances"]
        data["results"]["channels"][0]["alternatives"][0]["paragraphs"] = {"paragraphs": [
            {"sentences": [{"text": "Привет, мир.", "start": 0.1, "end": 1.0}]}]}
        out = asr_deepgram.map_response(data, pipeline.clean_word, requested_language="auto")
        self.assertEqual(out["segments"], [{"start": 0.1, "end": 1.0, "text": "Привет, мир."}])
        self.assertEqual((out["language"], out["language_probability"]), ("ru", 0.97))

    def test_build_sentences_compatible(self) -> None:
        from reel_factory import editorial
        out = asr_deepgram.map_response(FAKE, pipeline.clean_word, requested_language="ru")
        sentences = editorial.build_sentences(out)
        self.assertGreaterEqual(len(sentences), 2)


class ParamsTests(unittest.TestCase):
    def test_params(self) -> None:
        params = asr_deepgram.build_params("ru", ["ИИ", "ИИ", "HR"] + [f"t{i}" for i in range(80)])
        self.assertIn(("model", "nova-3"), params)
        self.assertIn(("language", "ru"), params)
        terms = [v for k, v in params if k == "keyterm"]
        self.assertEqual(terms[:2], ["ИИ", "HR"])
        self.assertEqual(len(terms), 50)
        auto = asr_deepgram.build_params("auto", None)
        self.assertIn(("detect_language", "true"), auto)
        self.assertNotIn("language", [k for k, _ in auto])
        for flag in ("smart_format", "punctuate", "filler_words", "utterances", "paragraphs"):
            self.assertIn((flag, "true"), auto)


class RequestTests(unittest.TestCase):
    def test_retries_on_5xx_then_ok(self) -> None:
        calls = []

        def post(url, data, key, timeout):
            calls.append(url)
            if len(calls) < 3:
                raise http_error(503)
            return FAKE
        with mock.patch.object(asr_deepgram, "_post", post):
            out = asr_deepgram.request_deepgram(b"x", asr_deepgram.build_params("ru", ["ИИ"]), FAKE_KEY,
                                                sleep=lambda s: None)
        self.assertEqual(out, FAKE)
        self.assertEqual(len(calls), 3)
        self.assertIn("keyterm=%D0%98%D0%98", calls[0])

    def test_gives_up_after_retries_and_hides_key(self) -> None:
        def post(url, data, key, timeout):
            raise urllib.error.URLError(f"boom Authorization: Token {key}")
        with mock.patch.object(asr_deepgram, "_post", post):
            with self.assertRaises(asr_deepgram.DeepgramError) as ctx:
                asr_deepgram.request_deepgram(b"x", [("model", "nova-3")], FAKE_KEY, sleep=lambda s: None)
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_auth_error_not_retried(self) -> None:
        calls = []

        def post(*a):
            calls.append(1)
            raise http_error(401, f"bad {FAKE_KEY}")
        with mock.patch.object(asr_deepgram, "_post", post):
            with self.assertRaises(asr_deepgram.DeepgramError) as ctx:
                asr_deepgram.request_deepgram(b"x", [], FAKE_KEY, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)
        self.assertIn("ключ", str(ctx.exception))
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_keyterm_rejection_retries_without_keyterms(self) -> None:
        urls = []

        def post(url, *a):
            urls.append(url)
            if "keyterm" in url:
                raise http_error(400, "keyterm is not supported for this language")
            return FAKE
        with mock.patch.object(asr_deepgram, "_post", post):
            asr_deepgram.request_deepgram(b"x", asr_deepgram.build_params("ru", ["ИИ"]), FAKE_KEY)
        self.assertEqual(len(urls), 2)
        self.assertNotIn("keyterm", urls[1])

    def test_redact(self) -> None:
        text = asr_deepgram.redact(f"x Authorization: Token {FAKE_KEY} y {FAKE_KEY}", FAKE_KEY)
        self.assertNotIn(FAKE_KEY, text)


class KeyLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        # the developer's real <repo>/.env must not leak into these tests
        patcher = mock.patch.object(asr_deepgram, "DOTENV_FILE", Path("/nonexistent/.env"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dotenv_file_in_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dotenv = Path(tmp) / ".env"
            dotenv.write_text(f"# comment\nREEL_LLM_PROVIDER=claude\nDEEPGRAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
            with mock.patch.object(asr_deepgram, "DOTENV_FILE", dotenv), \
                    mock.patch.object(asr_deepgram, "SECRETS_FILE", Path(tmp) / "none"), \
                    mock.patch.dict("os.environ", {"DEEPGRAM_API_KEY": ""}):
                self.assertEqual(asr_deepgram.load_api_key(), FAKE_KEY)
            dotenv.write_text("DEEPGRAM_API_KEY=\n", encoding="utf-8")  # empty value = not set
            with mock.patch.object(asr_deepgram, "DOTENV_FILE", dotenv), \
                    mock.patch.object(asr_deepgram, "SECRETS_FILE", Path(tmp) / "none"), \
                    mock.patch.dict("os.environ", {"DEEPGRAM_API_KEY": ""}):
                self.assertFalse(asr_deepgram.key_available())

    def test_env_then_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "deepgram.env"
            f.write_text(f"DEEPGRAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
            with mock.patch.object(asr_deepgram, "SECRETS_FILE", f), mock.patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop("DEEPGRAM_API_KEY", None)
                self.assertEqual(asr_deepgram.load_api_key(), FAKE_KEY)
                os.environ["DEEPGRAM_API_KEY"] = "from_env_key"
                self.assertEqual(asr_deepgram.load_api_key(), "from_env_key")
            with mock.patch.object(asr_deepgram, "SECRETS_FILE", Path(tmp) / "none"), \
                    mock.patch.dict("os.environ", {"DEEPGRAM_API_KEY": ""}):
                self.assertFalse(asr_deepgram.key_available())


def fake_extract(ffmpeg, video, dest):
    Path(dest).write_bytes(b"mp3")
    return Path(dest)


class PipelineWiringTests(unittest.TestCase):
    def run_transcribe(self, provider, post, key=FAKE_KEY, local=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        video = Path(tmp.name) / "v.mp4"
        video.write_bytes(b"v")
        messages: list[str] = []
        patches = [mock.patch.object(asr_deepgram, "load_api_key", lambda: key),
                   mock.patch.object(asr_deepgram, "extract_audio", fake_extract),
                   mock.patch.object(asr_deepgram, "_post", post)]
        if local is not None:
            patches.append(mock.patch.object(pipeline, "_transcribe_once", local))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        try:
            payload = pipeline.transcribe_video(video, Path(tmp.name) / "w", "ru", "base",
                                                lambda p, m: messages.append(m), glossary=["ИИ"],
                                                asr_provider=provider)
        except Exception as error:
            return None, messages, error, Path(tmp.name)
        return payload, messages, None, Path(tmp.name)

    def test_deepgram_payload_and_cache(self) -> None:
        calls = []
        payload, messages, error, tmp = self.run_transcribe("deepgram", lambda *a: calls.append(1) or FAKE)
        self.assertIsNone(error)
        self.assertEqual(payload["transcription"]["provider"], "deepgram")
        self.assertEqual(len(payload["words"]), 4)
        self.assertTrue(any("Загружаю аудио в Deepgram" in m for m in messages))
        self.assertTrue(any("Deepgram распознаёт" in m for m in messages))
        self.assertTrue((tmp / "w" / "deepgram_response.json").is_file())
        self.assertNotIn(FAKE_KEY, (tmp / "w" / "deepgram_response.json").read_text(encoding="utf-8"))
        self.assertNotIn(FAKE_KEY, (tmp / "w" / "transcript.json").read_text(encoding="utf-8"))
        # повторный запуск: из кэша
        with mock.patch.object(asr_deepgram, "load_api_key", lambda: FAKE_KEY), \
                mock.patch.object(asr_deepgram, "_post", lambda *a: self.fail("повторная оплата")):
            again = pipeline.transcribe_video(tmp / "v.mp4", tmp / "w", "ru", "base", lambda p, m: None,
                                              glossary=["ИИ"], asr_provider="deepgram")
        self.assertTrue(again["transcription"]["cached_response"])
        self.assertEqual(len(calls), 1)

    def test_explicit_mode_raises_russian_error(self) -> None:
        def post(*a):
            raise http_error(401)
        _, messages, error, _ = self.run_transcribe("deepgram", post)
        self.assertIsInstance(error, pipeline.PipelineError)
        self.assertIn("Deepgram", str(error))
        self.assertNotIn(FAKE_KEY, str(error))

    def test_auto_falls_back_to_local_with_warning(self) -> None:
        def post(*a):
            raise http_error(401)
        from types import SimpleNamespace as NS
        info = NS(language="ru", language_probability=0.9, duration=3.0, duration_after_vad=3.0)

        def local(*a, **k):
            return {"info": info, "words": [{"text": "а", "raw": "а", "start": 0, "end": 1, "probability": 1}],
                    "segments": [{"start": 0, "end": 1, "text": "а"}], "text": "а", "pipeline": "sequential",
                    "batch_size": 1, "elapsed_seconds": 1.0, "emit_progress": lambda *x, **y: None}
        with mock.patch.object(pipeline.MODELS, "get", lambda name: object()):
            payload, messages, error, tmp = self.run_transcribe("auto", post, local=local)
        self.assertIsNone(error, error)
        self.assertEqual(payload["transcription"]["provider"], "local")
        self.assertIn("Deepgram", payload["transcription"]["asr_warning"])
        self.assertNotIn(FAKE_KEY, payload["transcription"]["asr_warning"])
        self.assertTrue(any("Внимание" in m for m in messages))

    def test_auto_without_key_is_local(self) -> None:
        with mock.patch.object(asr_deepgram, "load_api_key", lambda: None):
            self.assertEqual(pipeline.resolve_asr_provider("auto"), "local")
            self.assertEqual(pipeline.resolve_asr_provider("local"), "local")
        with mock.patch.object(asr_deepgram, "load_api_key", lambda: FAKE_KEY):
            self.assertEqual(pipeline.resolve_asr_provider("auto"), "deepgram")


if __name__ == "__main__":
    unittest.main()


class OverlapSanitizeTests(unittest.TestCase):
    def test_overlapping_word_times_are_made_monotonic(self) -> None:
        from reel_factory.asr_deepgram import map_response
        data = {"results": {"channels": [{"alternatives": [{"transcript": "кризис. Ему", "words": [
            {"word": "кризис", "punctuated_word": "кризис.", "start": 12.4, "end": 13.6, "confidence": 0.9},
            {"word": "ему", "punctuated_word": "Ему", "start": 12.795, "end": 13.115, "confidence": 0.9},
        ]}]}]}}
        words = map_response(data, lambda s: s.strip(".").lower())["words"]
        self.assertLessEqual(words[0]["end"], words[1]["start"])
        self.assertGreater(words[0]["end"], words[0]["start"])
