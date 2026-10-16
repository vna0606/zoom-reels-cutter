from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel_factory import runtime_paths


class RuntimePathsTests(unittest.TestCase):
    def test_env_override_wins_for_ffmpeg_and_node(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fake = Path(raw) / "tool"
            fake.write_text("x", encoding="utf-8")
            with mock.patch.dict(os.environ, {"REEL_FFMPEG": str(fake), "REEL_NODE": str(fake)}):
                self.assertEqual(runtime_paths.ffmpeg_path(), str(fake))
                self.assertEqual(runtime_paths.node_path(), str(fake))

    def test_fontsdir_option_is_filter_escaped_and_overridable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.dict(os.environ, {"REEL_FONTS_DIR": raw}):
                option = runtime_paths.subtitles_fontsdir_option()
        self.assertTrue(option.startswith(":fontsdir='"))
        self.assertNotIn("C:\\Windows", option)

    def test_filter_path_escapes_windows_drive_colon(self) -> None:
        self.assertEqual(runtime_paths.ffmpeg_filter_path("C:\\a\\b.ass"), "C\\:/a/b.ass")

    def test_codex_command_none_when_missing(self) -> None:
        with mock.patch.object(runtime_paths, "node_path", return_value=None), \
                mock.patch.object(runtime_paths, "_which", return_value=None), \
                mock.patch.dict(os.environ, {"REEL_CODEX_BIN": ""}):
            self.assertIsNone(runtime_paths.codex_command())


if __name__ == "__main__":
    unittest.main()
