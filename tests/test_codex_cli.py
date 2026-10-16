from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel_factory.codex_cli import (
    codex_environment,
    codex_home,
    codex_runtime_home,
    codex_sqlite_home,
)


class CodexEnvironmentTests(unittest.TestCase):
    def test_explicit_codex_home_is_used_only_as_the_credential_source(self) -> None:
        with mock.patch.dict(os.environ, {"CODEX_HOME": r"D:\codex-state"}, clear=False), \
             mock.patch("reel_factory.codex_cli._prepare_codex_runtime_home", return_value=Path(r"D:\runtime")), \
             mock.patch.object(Path, "mkdir"):
            self.assertEqual(codex_home(), Path(r"D:\codex-state"))
            self.assertEqual(codex_environment()["CODEX_HOME"], r"D:\runtime")

    def test_profile_defaults_to_dot_codex(self) -> None:
        with mock.patch.dict(os.environ, {"USERPROFILE": r"C:\Users\example"}, clear=False):
            os.environ.pop("CODEX_HOME", None)
            self.assertEqual(codex_home(), Path(r"C:\Users\example") / ".codex")

    def test_sqlite_state_is_separate_from_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            state_home = root / "state"
            runtime_home = root / "runtime"
            with mock.patch(
                "reel_factory.codex_cli.codex_sqlite_home", return_value=state_home,
            ), mock.patch(
                "reel_factory.codex_cli._prepare_codex_runtime_home", return_value=runtime_home,
            ):
                env = codex_environment()

            self.assertEqual(state_home, Path(env["CODEX_SQLITE_HOME"]))
            self.assertNotEqual(Path(env["CODEX_HOME"]), state_home)
            self.assertTrue(state_home.is_dir())

    def test_runtime_home_defaults_to_workspace_and_is_not_the_desktop_profile(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REEL_FACTORY_CODEX_HOME", None)
            runtime_home = codex_runtime_home()

        self.assertEqual(runtime_home, Path(__file__).resolve().parents[1] / "workspace" / "codex-runtime")
        self.assertNotEqual(runtime_home, codex_home())

    def test_environment_copies_auth_and_owns_all_temp_directories(self) -> None:
        with self.subTest("isolated runtime"):
            with tempfile.TemporaryDirectory() as raw_tmp:
                root = Path(raw_tmp)
                credentials = root / "credentials"
                runtime = root / "runtime"
                sqlite = root / "sqlite"
                credentials.mkdir()
                (credentials / "auth.json").write_text('{"tokens":"test"}', encoding="utf-8")
                with mock.patch.dict(
                    os.environ,
                    {
                        "CODEX_HOME": str(credentials),
                        "REEL_FACTORY_CODEX_HOME": str(runtime),
                        "CODEX_SQLITE_HOME": str(sqlite),
                    },
                    clear=False,
                ):
                    env = codex_environment()

                self.assertEqual(Path(env["CODEX_HOME"]), runtime)
                self.assertEqual((runtime / "auth.json").read_text(encoding="utf-8"), '{"tokens":"test"}')
                self.assertTrue((runtime / "tmp" / "arg0").is_dir())
                self.assertEqual(Path(env["TEMP"]), runtime / "tmp" / "process")
                self.assertEqual(env["TEMP"], env["TMP"])

    def test_explicit_sqlite_home_wins(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CODEX_SQLITE_HOME": r"D:\codex-sqlite"},
            clear=False,
        ), mock.patch.object(Path, "mkdir") as mkdir, mock.patch(
            "reel_factory.codex_cli._prepare_codex_runtime_home",
            return_value=Path(r"D:\runtime"),
        ):
            env = codex_environment()

        self.assertEqual(env["CODEX_SQLITE_HOME"], r"D:\codex-sqlite")
        self.assertGreaterEqual(mkdir.call_count, 2)


if __name__ == "__main__":
    unittest.main()
