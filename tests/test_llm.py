from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from reel_factory import llm
from reel_factory.pipeline import PipelineError

SCHEMA = {
    "type": "object",
    "required": ["ok", "n"],
    "properties": {
        "ok": {"type": "boolean"},
        "n": {"type": "integer", "minimum": 0, "maximum": 10},
        "tag": {"type": "string", "enum": ["a", "b"]},
        "items": {"type": "array", "items": {"type": "object", "required": ["x"]}},
    },
}
READY = {
    "claude": {"installed": True, "ready": True, "model": "opus"},
    "codex": {"installed": True, "ready": True, "model": None},
}


def status(claude: bool, codex: bool) -> dict:
    return {
        "claude": {"installed": True, "ready": claude, "model": "opus"},
        "codex": {"installed": True, "ready": codex, "model": None},
    }


class ValidatorTests(unittest.TestCase):
    def test_accepts_valid_and_rejects_invalid(self) -> None:
        llm.validate_schema({"ok": True, "n": 3, "tag": "a", "items": [{"x": 1}]}, SCHEMA)
        for bad in (
            {"ok": True},
            {"ok": "yes", "n": 1},
            {"ok": True, "n": True},
            {"ok": True, "n": 11},
            {"ok": True, "n": 1, "tag": "z"},
            {"ok": True, "n": 1, "items": [{}]},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                llm.validate_schema(bad, SCHEMA)

    def test_extract_json_handles_fences_and_chatter(self) -> None:
        self.assertEqual(llm.extract_json_object('```json\n{"a": {"b": 1}}\n```'), {"a": {"b": 1}})
        self.assertEqual(llm.extract_json_object('Вот ответ: {"a": 1} Готово.'), {"a": 1})
        with self.assertRaises(ValueError):
            llm.extract_json_object("нет json")


class ResolveProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("REEL_LLM_PROVIDER", "REEL_CLAUDE_MODEL", "REEL_CODEX_MODEL"):
            os.environ.pop(name, None)
        cfg = mock.patch.object(llm, "_config", return_value={})
        cfg.start()
        self.addCleanup(cfg.stop)

    def test_auto_prefers_claude_then_codex(self) -> None:
        with mock.patch.object(llm, "provider_status", return_value=status(True, True)):
            self.assertEqual(llm.resolve_provider(None), "claude")
            self.assertEqual(llm.resolve_provider("auto"), "claude")
        with mock.patch.object(llm, "provider_status", return_value=status(False, True)):
            self.assertEqual(llm.resolve_provider("auto"), "codex")

    def test_env_default_and_explicit_override(self) -> None:
        os.environ["REEL_LLM_PROVIDER"] = "codex"
        with mock.patch.object(llm, "provider_status", return_value=status(True, True)):
            self.assertEqual(llm.resolve_provider(None), "codex")
            self.assertEqual(llm.resolve_provider("claude"), "claude")

    def test_nothing_ready_raises_llm_error_subclass_of_pipeline_error(self) -> None:
        with mock.patch.object(llm, "provider_status", return_value=status(False, False)):
            with self.assertRaises(llm.LLMError) as caught:
                llm.resolve_provider("auto")
            self.assertIsInstance(caught.exception, PipelineError)
            with self.assertRaises(llm.LLMError):
                llm.resolve_provider("claude")
            with self.assertRaises(llm.LLMError):
                llm.resolve_provider("heuristic")

    def test_model_configuration_priority(self) -> None:
        self.assertEqual(llm.claude_model(), "opus")
        with mock.patch.object(llm, "_config", return_value={"claude_model": "sonnet", "codex_model": "gpt-x"}):
            self.assertEqual(llm.claude_model(), "sonnet")
            self.assertEqual(llm.codex_model(), "gpt-x")
            os.environ["REEL_CLAUDE_MODEL"] = "haiku"
            self.assertEqual(llm.claude_model(), "haiku")


class RunLlmJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        for patcher in (
            mock.patch.object(llm, "provider_status", return_value=READY),
            mock.patch.object(llm, "_config", return_value={}),
            mock.patch.object(llm.runtime_paths, "claude_path", return_value="/bin/claude"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def envelope(**kwargs) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=json.dumps(kwargs), stderr="")

    def test_claude_structured_output_cache_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            with mock.patch.object(
                llm.subprocess, "run",
                return_value=self.envelope(structured_output={"ok": True, "n": 1}, result="{}"),
            ) as run:
                first = llm.run_llm_json("p", SCHEMA, work, task="t", provider="claude")
                second = llm.run_llm_json("p", SCHEMA, work, task="t", provider="claude")
                third = llm.run_llm_json("other", SCHEMA, work, task="t", provider="claude")
            self.assertEqual(first, {"ok": True, "n": 1})
            self.assertEqual(first, second)
            self.assertEqual(third, first)
            self.assertEqual(run.call_count, 2)  # second call was a cache hit
            command = run.call_args_list[0].args[0]
            for flag in ("-p", "--json-schema", "--effort", "--model"):
                self.assertIn(flag, command)
            self.assertEqual(command[command.index("--model") + 1], "opus")
            self.assertEqual(command[command.index("--effort") + 1], "high")
            self.assertIn("JSON-схем", run.call_args_list[0].kwargs["input"])
            self.assertTrue((work / "t.claude.json").is_file())
            self.assertTrue((work / "t.claude.log").is_file())

    def test_retries_once_with_error_then_succeeds(self) -> None:
        replies = [
            self.envelope(result='```json\n{"ok": "no"}\n```'),
            self.envelope(result='{"ok": true, "n": 2}'),
        ]
        with tempfile.TemporaryDirectory() as raw, \
                mock.patch.object(llm.subprocess, "run", side_effect=replies) as run:
            result = llm.run_llm_json("p", SCHEMA, Path(raw), task="t", provider="claude")
        self.assertEqual(result, {"ok": True, "n": 2})
        self.assertIn("не прошёл проверку", run.call_args_list[1].kwargs["input"])

    def test_second_failure_raises(self) -> None:
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            llm.subprocess, "run", return_value=self.envelope(result="не json"),
        ):
            with self.assertRaises(llm.LLMError):
                llm.run_llm_json("p", SCHEMA, Path(raw), task="t", provider="claude")

    def test_codex_command_uses_schema_effort_and_model(self) -> None:
        os.environ["REEL_CODEX_MODEL"] = "gpt-test"
        self.addCleanup(os.environ.pop, "REEL_CODEX_MODEL", None)

        def fake_run(command, **kwargs):
            out = Path(command[command.index("--output-last-message") + 1])
            out.write_text('{"ok": true, "n": 4}', encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="log")

        with tempfile.TemporaryDirectory() as raw, \
                mock.patch.object(llm.runtime_paths, "codex_command", return_value=["codex"]), \
                mock.patch("reel_factory.codex_cli.codex_environment", return_value={}), \
                mock.patch.object(llm.subprocess, "run", side_effect=fake_run) as run:
            result = llm.run_llm_json("p", SCHEMA, Path(raw), task="t", provider="codex", effort="medium")
            command = run.call_args.args[0]
        self.assertEqual(result, {"ok": True, "n": 4})
        self.assertIn("--output-schema", command)
        self.assertIn('model_reasoning_effort="medium"', command)
        self.assertEqual(command[command.index("--model") + 1], "gpt-test")
        self.assertIn("read-only", command)

    def test_claude_images_enable_read_tool_add_dir_and_cache_depends_on_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw) / "w"
            pics = Path(raw) / "pics"
            pics.mkdir()
            img = pics / "sheet1.jpg"
            img.write_bytes(b"aaa")
            with mock.patch.object(
                llm.subprocess, "run",
                return_value=self.envelope(structured_output={"ok": True, "n": 1}, result="{}"),
            ) as run:
                llm.run_llm_json("p", SCHEMA, work, task="t", provider="claude", images=[img])
                llm.run_llm_json("p", SCHEMA, work, task="t", provider="claude", images=[img])   # кэш
                self.assertEqual(run.call_count, 1)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--tools") + 1], "Read")
                self.assertEqual(command[command.index("--add-dir") + 1], str(pics.resolve()))
                sent = run.call_args.kwargs["input"]
                self.assertIn(str(img.resolve()), sent)
                self.assertIn("Read", sent)
                img.write_bytes(b"bbb")   # другое содержимое под тем же именем -> кэш не должен совпасть
                llm.run_llm_json("p", SCHEMA, work, task="t", provider="claude", images=[img])
                self.assertEqual(run.call_count, 2)
                llm.run_llm_json("p", SCHEMA, work, task="t2", provider="claude")
                text_command = run.call_args.args[0]
                self.assertEqual(text_command[text_command.index("--tools") + 1], "")
                self.assertNotIn("--add-dir", text_command)

    def test_codex_images_passed_with_repeatable_i_flag(self) -> None:
        def fake_run(command, **kwargs):
            Path(command[command.index("--output-last-message") + 1]).write_text('{"ok": true, "n": 4}', encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="log")

        with tempfile.TemporaryDirectory() as raw:
            imgs = []
            for name in ("a.jpg", "b.jpg"):
                (Path(raw) / name).write_bytes(name.encode())
                imgs.append(Path(raw) / name)
            with mock.patch.object(llm.runtime_paths, "codex_command", return_value=["codex"]), \
                    mock.patch("reel_factory.codex_cli.codex_environment", return_value={}), \
                    mock.patch.object(llm.subprocess, "run", side_effect=fake_run) as run:
                llm.run_llm_json("p", SCHEMA, Path(raw) / "w", task="t", provider="codex", images=imgs)
                command = run.call_args.args[0]
        positions = [i for i, x in enumerate(command) if x == "-i"]
        self.assertEqual([command[i + 1] for i in positions], [str(p.resolve()) for p in imgs])
        self.assertEqual(command[command.index("exec") + 1], "-")
        self.assertNotEqual(command[positions[-1] + 2], "-")

    def test_image_hashes_in_fingerprint_only_when_images_given(self) -> None:
        plain = llm._fingerprint("p", SCHEMA, "claude", "opus", "high")
        self.assertEqual(plain, llm._fingerprint("p", SCHEMA, "claude", "opus", "high", []))
        self.assertNotEqual(plain, llm._fingerprint("p", SCHEMA, "claude", "opus", "high", ["abc"]))


if __name__ == "__main__":
    unittest.main()
