from __future__ import annotations

import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HYPERFRAMES_ROOT = PROJECT_ROOT / "hyperframes" / "broadcast-motion"
COMPOSITION_PATH = HYPERFRAMES_ROOT / "index.html"
PACKAGE_PATH = HYPERFRAMES_ROOT / "package.json"


class _CompositionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html_attributes: dict[str, str | None] = {}
        self.root_attributes: dict[str, str | None] = {}
        self.attributes_by_id: dict[str, dict[str, str | None]] = {}
        self.remote_resources: list[str] = []
        self.script_sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "html":
            self.html_attributes = attributes
        if attributes.get("id") == "root":
            self.root_attributes = attributes
        if attributes.get("id"):
            self.attributes_by_id[str(attributes["id"])] = attributes
        if tag == "script" and attributes.get("src"):
            self.script_sources.append(str(attributes["src"]))
        for name in ("src", "href"):
            value = attributes.get(name)
            if value and re.match(r"^https?://", value, flags=re.IGNORECASE):
                self.remote_resources.append(value)


class HyperFramesCompositionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = COMPOSITION_PATH.read_text(encoding="utf-8")
        cls.parser = _CompositionParser()
        cls.parser.feed(cls.source)

    def test_portrait_root_and_motion_stage_have_literal_twelve_second_contract(self) -> None:
        html_attributes = self.parser.html_attributes
        root_attributes = self.parser.root_attributes
        stage_attributes = self.parser.attributes_by_id["motion-stage"]

        self.assertEqual(html_attributes.get("lang"), "ru")
        self.assertEqual(html_attributes.get("data-resolution"), "portrait")
        self.assertEqual(root_attributes.get("data-start"), "0")
        self.assertEqual(root_attributes.get("data-duration"), "12")
        self.assertEqual(root_attributes.get("data-width"), "1080")
        self.assertEqual(root_attributes.get("data-height"), "1920")
        self.assertEqual(root_attributes.get("data-composition-id"), "main")
        self.assertIn("clip", str(stage_attributes.get("class")).split())
        self.assertEqual(stage_attributes.get("data-start"), "0")
        self.assertEqual(stage_attributes.get("data-duration"), "12")
        self.assertEqual(stage_attributes.get("data-track-index"), "1")

    def test_motion_plan_is_valid_json_and_covers_template_timeline(self) -> None:
        match = re.search(
            r'<script\s+id="motion-plan"\s+type="application/json">(.*?)</script>',
            self.source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        plan = json.loads(match.group(1))

        self.assertEqual(plan["version"], 3)
        self.assertEqual(plan["duration"], 12)
        self.assertTrue(plan["scenes"])
        self.assertEqual(plan["scenes"][0]["start"], 0)
        self.assertEqual(plan["scenes"][-1]["end"], 12)
        for previous, current in zip(plan["scenes"], plan["scenes"][1:]):
            self.assertEqual(previous["end"], current["start"])

    def test_composition_registers_exactly_one_paused_main_gsap_timeline(self) -> None:
        self.assertEqual(self.source.count("gsap.timeline({ paused: true })"), 1)
        self.assertEqual(self.source.count('window.__timelines["main"] = tl'), 1)
        self.assertIn("window.__timelines = window.__timelines || {}", self.source)
        self.assertEqual(self.parser.script_sources, ["assets/gsap.min.js"])
        self.assertTrue((HYPERFRAMES_ROOT / "assets" / "gsap.min.js").is_file())

    def test_composition_is_self_contained_for_offline_rendering(self) -> None:
        self.assertEqual(self.parser.remote_resources, [])
        self.assertNotRegex(self.source, r"<script[^>]+src=[\"']https?://")
        self.assertNotRegex(self.source, r"url\([\"']?https?://")

        package = json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))
        version = package["devDependencies"]["hyperframes"]
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        self.assertTrue((HYPERFRAMES_ROOT / "presets" / "editorial_lime" / "style.css").is_file())
        self.assertTrue((HYPERFRAMES_ROOT / "presets" / "itbooster" / "style.css").is_file())

    def test_template_has_distinct_motion_worlds_and_source_frame_treatment(self) -> None:
        for token in (
            "poster-layout",
            "quote-layout",
            "metric-layout",
            "split-layout",
            "flow-layout",
            "network-layout",
            "ledger-layout",
            "signal-layout",
            "source-layout",
            "composition-split-stage",
            "motion_recipe",
            "source-frame-wrap",
            "Кадр из эфира",
        ):
            self.assertIn(token, self.source)
        self.assertIn("const buildVisual = (scene, sceneIndex)", self.source)
        self.assertIn('variant === "source" && scene.source_frame', self.source)

    def test_network_layout_is_responsive_inside_split_stage(self) -> None:
        self.assertIn("grid-template-columns: minmax(0, .44fr) minmax(0, .56fr)", self.source)
        self.assertIn(".composition-split-stage .visual-shell { min-width: 0; }", self.source)
        self.assertIn(".network-center { left: 27%; top: 33%; width: 46%; height: 34%;", self.source)
        self.assertIn(".network-node { width: 38%;", self.source)
        self.assertIn('makeSvg("network-svg", "0 0 100 100")', self.source)
        self.assertNotIn(".network-center { left: 310px", self.source)

    def test_dynamic_copy_height_and_poster_text_have_safe_layout_fallbacks(self) -> None:
        self.assertIn("grid-template-rows: max-content minmax(0, 1fr)", self.source)
        self.assertIn(".scene-layer:not(.composition-split-stage) .scene-copy { padding-bottom: 64px; }", self.source)
        self.assertNotIn(".variant-radial .scene-inner, .variant-signal .scene-inner { grid-template-rows: 320px 1fr; }", self.source)
        self.assertIn("const fitPosterPanel = (element) =>", self.source)
        self.assertIn('if (scene._layout_safe) layer.classList.add("layout-safe")', self.source)

    def test_full_bleed_child_owns_background_and_lower_zone_is_reserved(self) -> None:
        root_css = re.search(r"#root\s*\{([^}]*)}", self.source, flags=re.DOTALL)
        canvas_css = re.search(r"\.frame-fill\s*\{([^}]*)}", self.source, flags=re.DOTALL)
        scene_css = re.search(r"\.scene-inner\s*\{([^}]*)}", self.source, flags=re.DOTALL)
        self.assertIsNotNone(root_css)
        self.assertIsNotNone(canvas_css)
        self.assertIsNotNone(scene_css)
        self.assertNotIn("background:", root_css.group(1))
        self.assertIn("position: absolute", canvas_css.group(1))
        self.assertIn("inset: 0", canvas_css.group(1))
        self.assertIn("background:", canvas_css.group(1))

        top = int(re.search(r"top:\s*(\d+)px", scene_css.group(1)).group(1))
        height = int(re.search(r"height:\s*(\d+)px", scene_css.group(1)).group(1))
        self.assertLessEqual(top + height, 1272)
        self.assertGreaterEqual(1920 - top - height, 648)


if __name__ == "__main__":
    unittest.main()
