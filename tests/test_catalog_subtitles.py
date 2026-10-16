from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reel_factory.pipeline import write_ass


class CatalogSubtitleTests(unittest.TestCase):
    def test_catalog_caption_style_changes_ass_profile_and_keeps_word_timing(self) -> None:
        words = [{"start": 0.1, "end": 0.45, "text": "Проверка"}]
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "captions.ass"
            write_ass(words, path, 1.0, "caption-kinetic-slam")
            rendered = path.read_text(encoding="utf-8-sig")

        self.assertIn("Style: Word,Arial Black,92", rendered)
        self.assertIn("\\fscx132", rendered)
        self.assertIn("Dialogue: 0,0:00:00.10,0:00:00.53", rendered)

    def test_unknown_catalog_caption_style_falls_back_to_legacy_pill(self) -> None:
        words = [{"start": 0.0, "end": 0.2, "text": "Тест"}]
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "captions.ass"
            write_ass(words, path, 0.5, "unknown-caption")
            rendered = path.read_text(encoding="utf-8-sig")

        self.assertIn("Style: Word,Arial,82", rendered)


if __name__ == "__main__":
    unittest.main()
