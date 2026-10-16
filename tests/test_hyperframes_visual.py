from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import reel_factory.hyperframes_visual as hyperframes_visual


def _scene(start: float, end: float, *, kind: str = "statement", headline: str = "Тезис") -> dict:
    return {
        "start": start,
        "end": end,
        "kind": kind,
        "eyebrow": "Раздел",
        "headline": headline,
        "body": "Пояснение",
        "primary": "Главное",
        "secondary": "Контекст",
        "items": [],
        "visual_variant": hyperframes_visual.KIND_VISUAL_VARIANTS[kind],
        "composition": "left-led",
        "motion_recipe": "reveal",
        "accent_word": "",
        "visual_concept": "Проверяемая визуальная идея",
        "use_source_frame": False,
        "catalog_block": hyperframes_visual.KIND_CATALOG_BLOCKS[kind],
        "catalog_motion": hyperframes_visual.KIND_CATALOG_MOTIONS[kind],
        "catalog_transition": "fade-through",
        "catalog_reason": "The catalog recipe matches the scene meaning",
    }


class VisualPlanValidationTests(unittest.TestCase):
    def test_validation_preserves_fragment_title_as_timed_visual_hook(self) -> None:
        normalized = hyperframes_visual.validate_visual_plan({
            "version": hyperframes_visual.VISUAL_PLAN_VERSION,
            "hook_title": "Сеньора заменили дешёвым фулстаком",
            "scenes": [_scene(0.0, 8.0)],
        }, 8.0)

        self.assertEqual(normalized["hook_title"], "Сеньора заменили дешёвым фулстаком")
        self.assertEqual(normalized["hook_duration"], hyperframes_visual.HOOK_TITLE_DURATION)

    def test_validation_falls_back_to_first_headline_when_fragment_title_is_missing(self) -> None:
        normalized = hyperframes_visual.validate_visual_plan({
            "version": hyperframes_visual.VISUAL_PLAN_VERSION,
            "scenes": [_scene(0.0, 6.0, headline="Первый смысловой тезис")],
        }, 6.0)

        self.assertEqual(normalized["hook_title"], "Первый смысловой тезис")

    def test_layout_errors_repair_only_the_scene_active_at_the_failure_time(self) -> None:
        plan = hyperframes_visual.validate_visual_plan({
            "version": hyperframes_visual.VISUAL_PLAN_VERSION,
            "scenes": [
                _scene(0.0, 4.0, headline="Первая сцена"),
                _scene(4.0, 8.0, kind="network", headline="Плотная сеть"),
            ],
        }, 8.0)
        payload = {
            "lint": {"errorCount": 0},
            "runtime": {"errorCount": 0},
            "motion": {"errorCount": 0},
            "contrast": {"errorCount": 0},
            "layout": {"findings": [{"severity": "error", "time": 6.0}]},
        }

        repaired, scene_ids = hyperframes_visual._layout_repair_plan(plan, payload, 8.0)

        self.assertEqual(scene_ids, ["s002"])
        self.assertFalse(repaired["scenes"][0]["_layout_safe"])
        self.assertTrue(repaired["scenes"][1]["_layout_safe"])
        self.assertTrue(repaired["scenes"][1]["_catalog_disabled"])
        self.assertEqual(repaired["scenes"][1]["visual_variant"], "flow")
        self.assertEqual(repaired["scenes"][1]["composition"], "stacked")

    def test_adjacent_equal_variants_are_diversified(self) -> None:
        first = _scene(0.0, 4.0, kind="statement")
        second = _scene(4.0, 8.0, kind="statement")
        first["visual_variant"] = "kinetic"
        second["visual_variant"] = "kinetic"

        normalized = hyperframes_visual.validate_visual_plan(
            {"version": hyperframes_visual.VISUAL_PLAN_VERSION, "scenes": [first, second]}, 8.0,
        )

        self.assertEqual(normalized["scenes"][0]["visual_variant"], "poster")
        self.assertNotEqual(
            normalized["scenes"][0]["visual_variant"],
            normalized["scenes"][1]["visual_variant"],
        )

    def test_validation_normalizes_edges_and_small_boundary_drift(self) -> None:
        normalized = hyperframes_visual.validate_visual_plan(
            {
                "version": hyperframes_visual.VISUAL_PLAN_VERSION,
                "scenes": [
                    _scene(0.2, 4.8, headline="Первый тезис"),
                    _scene(5.2, 10.2, kind="comparison", headline="Второй тезис"),
                ],
            },
            duration=10.0,
        )

        self.assertEqual(normalized["duration"], 10.0)
        self.assertEqual(normalized["scenes"][0]["start"], 0.0)
        self.assertEqual(normalized["scenes"][-1]["end"], 10.0)
        self.assertEqual(normalized["scenes"][0]["end"], 5.0)
        self.assertEqual(normalized["scenes"][1]["start"], 5.0)
        self.assertEqual([scene["id"] for scene in normalized["scenes"]], ["s001", "s002"])

    def test_validation_rejects_large_timeline_gap(self) -> None:
        with self.assertRaisesRegex(ValueError, "large gap or overlap"):
            hyperframes_visual.validate_visual_plan(
                {
                    "version": hyperframes_visual.VISUAL_PLAN_VERSION,
                    "scenes": [
                        _scene(0.0, 3.0),
                        _scene(5.0, 10.0),
                    ],
                },
                duration=10.0,
            )

    def test_catalog_choices_are_normalized_and_recorded(self) -> None:
        scene = _scene(0.0, 6.0, kind="stat")
        scene["catalog_block"] = "missing-block"
        scene["catalog_motion"] = "missing-motion"
        scene["catalog_transition"] = "missing-transition"
        normalized = hyperframes_visual.validate_visual_plan({
            "version": hyperframes_visual.VISUAL_PLAN_VERSION,
            "caption_style": "missing-caption",
            "scenes": [scene],
        }, 6.0)

        selected = normalized["scenes"][0]
        self.assertIn(selected["catalog_block"], hyperframes_visual.KIND_CATALOG_BLOCK_POOLS["stat"])
        self.assertIn(selected["catalog_motion"], hyperframes_visual.KIND_CATALOG_MOTION_POOLS["stat"])
        self.assertIn(selected["catalog_transition"], hyperframes_visual.CATALOG_TRANSITION_FALLBACKS)
        self.assertIn(selected["catalog_icon"], hyperframes_visual.KIND_ICON_POOLS["stat"])
        self.assertEqual(normalized["caption_style"], hyperframes_visual.CATALOG_CAPTION_DEFAULT)

    def test_semantic_fallback_uses_real_catalog_blocks_and_iconify_icons(self) -> None:
        selected_blocks = {
            hyperframes_visual._scene_catalog_choice(kind, f"{kind} grounded scene", index)
            for index, kind in enumerate(sorted(hyperframes_visual.ALLOWED_SCENE_KINDS))
        }
        selected_icons = {
            hyperframes_visual._scene_icon_choice(kind, f"{kind} grounded scene", index)
            for index, kind in enumerate(sorted(hyperframes_visual.ALLOWED_SCENE_KINDS))
        }

        self.assertGreaterEqual(len(selected_blocks), 6)
        self.assertGreaterEqual(len(selected_icons), 6)
        for block in selected_blocks:
            self.assertIn(block, hyperframes_visual.CATALOG_RUNTIME_BLOCKS)
            self.assertIsNotNone(hyperframes_visual._catalog_source_info(block, "block"))
        for icon in selected_icons:
            self.assertIn(icon, hyperframes_visual._icon_catalog())

    def test_catalog_motion_with_unbound_demo_claims_is_replaced(self) -> None:
        scene = _scene(0.0, 6.0)
        scene["catalog_motion"] = "pull-back-reveal"

        normalized = hyperframes_visual.validate_visual_plan({
            "version": hyperframes_visual.VISUAL_PLAN_VERSION,
            "scenes": [scene],
        }, 6.0)

        self.assertEqual(normalized["scenes"][0]["catalog_motion"], "headline-slam")

    def test_fallback_is_deterministic_grounded_and_covers_full_clip(self) -> None:
        transcript = {
            "segments": [
                {"start": 0.0, "end": 5.8, "text": "Конверсия выросла на 25% за месяц."},
                {"start": 5.8, "end": 12.0, "text": "Команда повторила этот процесс на втором проекте."},
            ]
        }
        candidate = {"title": "Проверяем рост", "hook": "Результат за один месяц"}

        first = hyperframes_visual.fallback_visual_plan(transcript, 12.0, candidate)
        second = hyperframes_visual.fallback_visual_plan(transcript, 12.0, candidate)

        self.assertEqual(first, second)
        self.assertEqual(first["scenes"][0]["start"], 0.0)
        self.assertEqual(first["scenes"][-1]["end"], 12.0)
        for previous, current in zip(first["scenes"], first["scenes"][1:]):
            self.assertEqual(previous["end"], current["start"])
        self.assertEqual(first["scenes"][0]["kind"], "stat")
        self.assertEqual(first["scenes"][0]["primary"], "25%")
        self.assertIn("25%", first["scenes"][0]["body"])
        self.assertEqual(first["hook_title"], candidate["title"])

    def test_fallback_requests_source_frame_for_chat_or_graph_evidence(self) -> None:
        plan = hyperframes_visual.fallback_visual_plan(
            {"segments": [{"start": 0.0, "end": 6.0, "text": "На экране переписка из чата и график роста."}]},
            6.0,
        )

        self.assertTrue(plan["scenes"][0]["use_source_frame"])
        self.assertEqual(plan["scenes"][0]["visual_variant"], "source")


class SourceFrameTests(unittest.TestCase):
    def test_materialization_wires_animated_title_hook_and_reports_its_motion(self) -> None:
        plan = hyperframes_visual.validate_visual_plan({
            "version": hyperframes_visual.VISUAL_PLAN_VERSION,
            "hook_title": "Почему <дорогого> сеньора заменили универсальным разработчиком",
            "scenes": [_scene(0.0, 8.0)],
        }, 8.0)
        with tempfile.TemporaryDirectory() as raw_tmp:
            materialized = hyperframes_visual.materialize_hyperframes_project(
                plan, 8.0, [], Path(raw_tmp),
            )
            project_dir = materialized["project_dir"]
            html = (project_dir / "index.html").read_text(encoding="utf-8")
            selection = json.loads(
                (project_dir / "catalog_selection.json").read_text(encoding="utf-8")
            )

        self.assertIn('id="hook-card"', html)
        self.assertIn('"hook_title":"Почему \\u003cдорогого> сеньора', html)
        self.assertIn('const hookTitleWords =', html)
        self.assertIn('tl.set(hookCard, { autoAlpha: 1 }, 0)', html)
        self.assertEqual(selection["hook"]["title"], plan["hook_title"])
        self.assertEqual(
            selection["hook"]["motion"],
            ["waterfall-entry", "gradient-text-sweep", "ambient-glow-bloom"],
        )

    def test_requested_source_frame_is_extracted_and_attached(self) -> None:
        frame = np.full((180, 320, 3), 245, dtype=np.uint8)
        frame[30:150:12, 24:296] = 20

        class FakeCapture:
            def isOpened(self):
                return True

            def set(self, *_args):
                return True

            def read(self):
                return True, frame.copy()

            def release(self):
                return None

        scene = _scene(0.0, 6.0, headline="Показываю переписку из чата")
        scene["use_source_frame"] = True
        scene["visual_variant"] = "source"
        plan = hyperframes_visual.validate_visual_plan({
            "version": hyperframes_visual.VISUAL_PLAN_VERSION, "scenes": [scene],
        }, 6.0)

        with tempfile.TemporaryDirectory() as raw_tmp:
            work_dir = Path(raw_tmp)

            def fake_imwrite(path, _image, _params):
                Path(path).write_bytes(b"jpeg")
                return True

            with (
                patch.object(hyperframes_visual.cv2, "VideoCapture", return_value=FakeCapture()),
                patch.object(hyperframes_visual.cv2, "imwrite", side_effect=fake_imwrite),
            ):
                attached = hyperframes_visual.attach_presentation_frames(
                    Path("source.mp4"),
                    plan,
                    {"x": 0, "y": 0, "width": 320, "height": 180},
                    work_dir,
                    lambda *_args: None,
                )

            source_path = Path(attached["scenes"][0]["source_frame"])
            self.assertTrue(source_path.is_file())
            self.assertEqual(attached["scenes"][0]["visual_variant"], "source")

            materialized = hyperframes_visual.materialize_hyperframes_project(
                attached, 6.0, [], work_dir,
            )
            copied = materialized["project_dir"] / "assets" / "source_frames" / "s001.jpg"
            self.assertTrue(copied.is_file())
            html = (materialized["project_dir"] / "index.html").read_text(encoding="utf-8")
            self.assertIn("assets/source_frames/s001.jpg", html)
            self.assertIn('href="preset/style.css"', html)
            self.assertTrue((materialized["project_dir"] / "preset" / "style.css").is_file())
            selection = json.loads(
                (materialized["project_dir"] / "catalog_selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                selection["scenes"][0]["block"],
                hyperframes_visual.KIND_CATALOG_BLOCKS["statement"],
            )
            self.assertEqual(selection["caption_style"], hyperframes_visual.CATALOG_CAPTION_DEFAULT)

    def test_itbooster_preset_materializes_its_frame_style_and_font(self) -> None:
        plan = hyperframes_visual.validate_visual_plan({
            "version": hyperframes_visual.VISUAL_PLAN_VERSION,
            "scenes": [_scene(0.0, 6.0)],
        }, 6.0)
        with tempfile.TemporaryDirectory() as raw_tmp:
            materialized = hyperframes_visual.materialize_hyperframes_project(
                plan, 6.0, [], Path(raw_tmp), visual_preset="itbooster",
            )
            project_dir = materialized["project_dir"]
            style = (project_dir / "preset" / "style.css").read_text(encoding="utf-8")
            frame = (project_dir / "frame.md").read_text(encoding="utf-8")

            self.assertEqual(materialized["visual_preset"], "itbooster")
            self.assertIn("#aefbc3", style.lower())
            self.assertIn("Onest Variable", frame)
            self.assertTrue((project_dir / "preset" / "assets" / "onest-cyrillic-wght-normal.woff2").is_file())


class CatalogRuntimeTests(unittest.TestCase):
    def test_catalog_dependencies_resolve_from_template_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            project_dir = Path(raw_tmp)
            source_path = (
                hyperframes_visual.HYPERFRAMES_TEMPLATE
                / "compositions"
                / "notes-reveal.html"
            )
            source = source_path.read_text(encoding="utf-8")
            hyperframes_visual._copy_catalog_dependencies(source_path, project_dir, source)
            assets = project_dir / "assets"

            self.assertTrue((assets / "permanent-marker-400.woff2").is_file())
            self.assertTrue((assets / "courier-prime-700.woff2").is_file())

    def test_materialization_mounts_isolated_retimed_catalog_compositions(self) -> None:
        plan = {
            "version": hyperframes_visual.VISUAL_PLAN_VERSION,
            "scenes": [
                _scene(0.0, 4.0),
                _scene(4.0, 10.0),
            ],
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            materialized = hyperframes_visual.materialize_hyperframes_project(
                plan, 10.0, [], Path(raw_tmp),
            )
            project_dir = materialized["project_dir"]
            selection = json.loads(
                (project_dir / "catalog_selection.json").read_text(encoding="utf-8")
            )
            runtimes = [scene["runtime"] for scene in selection["scenes"]]

            self.assertEqual(len({runtime["composition_id"] for runtime in runtimes}), 2)
            self.assertEqual(len({runtime["src"] for runtime in runtimes}), 2)
            for runtime, expected_duration in zip(runtimes, (4.0, 6.0)):
                source = (project_dir / runtime["src"]).read_text(encoding="utf-8")
                self.assertIn(f'data-composition-id="{runtime["composition_id"]}"', source)
                self.assertIn(f'data-duration="{expected_duration:.3f}"', source)
                self.assertNotIn("https://cdn.jsdelivr.net", source)
                self.assertTrue(runtime["variables"])

            html = (project_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("data-composition-src", html)
            self.assertIn("buildCatalogVisual", html)
            self.assertIn("buildSceneIcon", html)
            self.assertTrue((project_dir / "icon-catalog.json").is_file())
            for scene in selection["scenes"]:
                self.assertTrue(scene["icon"])
                self.assertTrue(scene["runtime"])


class VisualPlanCacheTests(unittest.TestCase):
    def test_visual_llm_runner_delegates_to_provider_layer_with_high_effort(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            with patch.object(
                hyperframes_visual.llm, "run_llm_json", return_value={"version": 3},
            ) as run:
                payload = hyperframes_visual._run_llm("prompt", {"type": "object"}, root, "claude")

        self.assertEqual(payload, {"version": 3})
        args, kwargs = run.call_args
        self.assertEqual(args[0], "prompt")
        self.assertEqual(kwargs["task"], "visual_plan")
        self.assertEqual(kwargs["provider"], "claude")
        self.assertEqual(kwargs["effort"], "high")

    def test_identical_request_uses_codex_cache_and_transcript_change_invalidates_it(self) -> None:
        transcript = {
            "segments": [{"start": 0.0, "end": 8.0, "text": "Первый проверяемый тезис."}]
        }
        changed_transcript = {
            "segments": [{"start": 0.0, "end": 8.0, "text": "Другой проверяемый тезис."}]
        }
        callbacks: list[tuple[int, str]] = []

        def codex_plan(prompt, _schema, _work_dir, _provider):
            headline = "Другой тезис" if "Другой проверяемый тезис" in prompt else "Первый тезис"
            return {
                "version": hyperframes_visual.VISUAL_PLAN_VERSION,
                "scenes": [_scene(0.0, 8.0, headline=headline)],
            }

        with tempfile.TemporaryDirectory() as raw_tmp:
            work_dir = Path(raw_tmp)
            with (
                patch.object(hyperframes_visual.llm, "resolve_provider", return_value="codex"),
                patch.object(hyperframes_visual, "_run_llm", side_effect=codex_plan) as run_codex,
            ):
                first = hyperframes_visual.build_visual_plan(
                    transcript,
                    8.0,
                    work_dir,
                    {"provider": "heuristic"},
                    {"title": "Тест"},
                    lambda progress, message: callbacks.append((progress, message)),
                )
                cached = hyperframes_visual.build_visual_plan(
                    transcript,
                    8.0,
                    work_dir,
                    {"provider": "heuristic"},
                    {"title": "Тест"},
                    lambda progress, message: callbacks.append((progress, message)),
                )
                changed = hyperframes_visual.build_visual_plan(
                    changed_transcript,
                    8.0,
                    work_dir,
                    {"provider": "heuristic"},
                    {"title": "Тест"},
                    lambda progress, message: callbacks.append((progress, message)),
                )

            self.assertEqual(first, cached)
            self.assertNotEqual(first["fingerprint"], changed["fingerprint"])
            self.assertEqual(run_codex.call_count, 2)
            self.assertEqual(first["provider_used"], "codex")
            self.assertTrue((work_dir / "visual_plan.json").is_file())
            self.assertEqual(
                (work_dir / "visual_plan.request.sha256").read_text(encoding="utf-8").strip(),
                changed["fingerprint"],
            )
        self.assertEqual(callbacks[-1], (61, "Визуальный план готов"))

    def test_codex_failure_uses_local_catalog_plan_when_cache_is_missing(self) -> None:
        transcript = {"segments": [{"start": 0.0, "end": 6.0, "text": "Проверяемый тезис."}]}
        with tempfile.TemporaryDirectory() as raw_tmp:
            work_dir = Path(raw_tmp)
            with patch.object(
                hyperframes_visual.llm, "resolve_provider", return_value="codex",
            ), patch.object(
                hyperframes_visual, "_run_llm", side_effect=RuntimeError("offline"),
            ):
                plan = hyperframes_visual.build_visual_plan(
                    transcript, 6.0, work_dir, {"provider": "auto"}, None, lambda *_args: None,
                )

            self.assertEqual(plan["provider_used"], "local-catalog-resilient")
            self.assertEqual(plan["scenes"][0]["catalog_motion"], "headline-slam")
            self.assertTrue((work_dir / "visual_plan.codex.error.log").is_file())
            self.assertTrue((work_dir / "visual_plan.fallback.log").is_file())

    def test_codex_failure_reuses_valid_clip_local_semantic_plan(self) -> None:
        transcript = {"segments": [{"start": 0.0, "end": 6.0, "text": "Сетевой сбой."}]}
        with tempfile.TemporaryDirectory() as raw_tmp:
            work_dir = Path(raw_tmp)
            previous = hyperframes_visual.validate_visual_plan({
                "version": hyperframes_visual.VISUAL_PLAN_VERSION,
                "scenes": [_scene(0.0, 6.0, headline="Сохранённый смысловой план")],
            }, 6.0)
            (work_dir / "visual_plan.json").write_text(
                json.dumps(previous, ensure_ascii=False), encoding="utf-8",
            )
            (work_dir / "visual_plan.request.sha256").write_text("stale", encoding="utf-8")

            with patch.object(
                hyperframes_visual.llm, "resolve_provider", return_value="codex",
            ), patch.object(
                hyperframes_visual, "_run_llm", side_effect=RuntimeError("stream disconnected"),
            ):
                plan = hyperframes_visual.build_visual_plan(
                    transcript, 6.0, work_dir, {"provider": "auto"}, None, lambda *_args: None,
                )

            self.assertEqual(plan["provider_used"], "codex-cache-resilient")
            self.assertEqual(plan["scenes"][0]["headline"], "Сохранённый смысловой план")

    def test_strict_schema_requires_every_declared_scene_property(self) -> None:
        scene_schema = hyperframes_visual._visual_plan_schema()["properties"]["scenes"]["items"]
        self.assertEqual(set(scene_schema["required"]), set(scene_schema["properties"]))
        self.assertIn("caption_style", hyperframes_visual._visual_plan_schema()["required"])


class LocalCliResolutionTests(unittest.TestCase):
    def test_media_tools_resolve_explicit_ffmpeg_and_local_ffprobe(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            ffmpeg = root / "ffmpeg-custom.exe"
            ffprobe = root / "ffprobe.exe"
            ffmpeg.write_bytes(b"ffmpeg")
            ffprobe.write_bytes(b"ffprobe")

            with (
                patch.dict(
                    os.environ,
                    {
                        "HYPERFRAMES_FFMPEG_PATH": str(ffmpeg),
                        "HYPERFRAMES_FFPROBE_PATH": "",
                    },
                    clear=False,
                ),
                patch.object(hyperframes_visual.runtime_paths, "ffprobe_path", return_value=str(ffprobe)),
            ):
                resolved_ffmpeg, resolved_ffprobe = hyperframes_visual.resolve_hyperframes_media_tools()

        self.assertEqual(resolved_ffmpeg, ffmpeg.resolve())
        self.assertEqual(resolved_ffprobe, ffprobe.resolve())

    def test_media_tools_fail_clearly_when_ffprobe_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            ffmpeg = root / "ffmpeg-custom.exe"
            ffmpeg.write_bytes(b"ffmpeg")

            with (
                patch.dict(
                    os.environ,
                    {
                        "HYPERFRAMES_FFMPEG_PATH": str(ffmpeg),
                        "HYPERFRAMES_FFPROBE_PATH": "",
                    },
                    clear=False,
                ),
                patch.object(hyperframes_visual.runtime_paths, "ffprobe_path", return_value=None),
            ):
                with self.assertRaisesRegex(hyperframes_visual.PipelineError, "FFprobe"):
                    hyperframes_visual.resolve_hyperframes_media_tools()

    def test_resolver_returns_only_bundled_node_and_pinned_local_cli(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            template = root / "broadcast-motion"
            node = root / "runtime" / "node.exe"
            cli = template / "node_modules" / "hyperframes" / "bin" / "hyperframes.mjs"
            node.parent.mkdir(parents=True)
            node.write_text("local node", encoding="utf-8")
            cli.parent.mkdir(parents=True)
            cli.write_text("local cli", encoding="utf-8")
            (template / "package.json").write_text(
                json.dumps({
                    "devDependencies": {
                        "hyperframes": hyperframes_visual.HYPERFRAMES_VERSION,
                    }
                }),
                encoding="utf-8",
            )

            with (
                patch.object(hyperframes_visual.runtime_paths, "node_path", return_value=str(node)),
                patch.object(hyperframes_visual, "HYPERFRAMES_TEMPLATE", template),
                patch.object(hyperframes_visual, "ISOLATED_HYPERFRAMES_RUNTIME", root / "missing-runtime"),
            ):
                command = hyperframes_visual.resolve_hyperframes_command()

        self.assertEqual(command, [str(node), str(cli)])
        joined = " ".join(command).lower()
        for forbidden in ("npx", "pnpx", "dlx", "http://", "https://"):
            self.assertNotIn(forbidden, joined)

    def test_resolver_prefers_readable_isolated_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            template = root / "broadcast-motion"
            node = root / "node.exe"
            linked_cli = template / "node_modules" / "hyperframes" / "bin" / "hyperframes.mjs"
            runtime = root / "runtime"
            isolated_cli = runtime / "node_modules" / "hyperframes" / "bin" / "hyperframes.mjs"
            node.write_text("local node", encoding="utf-8")
            linked_cli.parent.mkdir(parents=True)
            linked_cli.write_text("linked cli", encoding="utf-8")
            isolated_cli.parent.mkdir(parents=True)
            isolated_cli.write_text("isolated cli", encoding="utf-8")
            (template / "package.json").write_text(
                json.dumps({"devDependencies": {"hyperframes": hyperframes_visual.HYPERFRAMES_VERSION}}),
                encoding="utf-8",
            )

            with (
                patch.object(hyperframes_visual.runtime_paths, "node_path", return_value=str(node)),
                patch.object(hyperframes_visual, "HYPERFRAMES_TEMPLATE", template),
                patch.object(hyperframes_visual, "ISOLATED_HYPERFRAMES_RUNTIME", runtime),
            ):
                command = hyperframes_visual.resolve_hyperframes_command()

        self.assertEqual(command, [str(node), str(isolated_cli)])

    def test_transient_hyperframes_failure_is_retried(self) -> None:
        failed = subprocess.CompletedProcess(["node"], 1, stdout="Error: EPERM operation not permitted")
        passed = subprocess.CompletedProcess(["node"], 0, stdout="ok")
        with (
            patch.object(hyperframes_visual.subprocess, "run", side_effect=[failed, passed]) as run,
            patch.object(hyperframes_visual.time, "sleep"),
        ):
            result = hyperframes_visual._run_hyperframes_process(
                ["node"], cwd=Path.cwd(), env={}, timeout=5,
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(run.call_count, 2)

    def test_semantic_hyperframes_failure_is_not_retried(self) -> None:
        failed = subprocess.CompletedProcess(["node"], 1, stdout="content_overlap")
        with patch.object(hyperframes_visual.subprocess, "run", return_value=failed) as run:
            result = hyperframes_visual._run_hyperframes_process(
                ["node"], cwd=Path.cwd(), env={}, timeout=5,
            )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(run.call_count, 1)

    def test_resolver_rejects_unpinned_project_version(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            template = root / "broadcast-motion"
            node = root / "node.exe"
            template.mkdir(parents=True)
            node.write_text("local node", encoding="utf-8")
            (template / "package.json").write_text(
                json.dumps({"devDependencies": {"hyperframes": "0.0.0"}}),
                encoding="utf-8",
            )

            with (
                patch.object(hyperframes_visual.runtime_paths, "node_path", return_value=str(node)),
                patch.object(hyperframes_visual, "HYPERFRAMES_TEMPLATE", template),
                patch.object(hyperframes_visual, "ISOLATED_HYPERFRAMES_RUNTIME", root / "missing-runtime"),
            ):
                with self.assertRaisesRegex(hyperframes_visual.PipelineError, "Ожидался локальный"):
                    hyperframes_visual.resolve_hyperframes_command()


@unittest.skipUnless(
    os.environ.get("RUN_HYPERFRAMES_SMOKE") == "1",
    "set RUN_HYPERFRAMES_SMOKE=1 to run the local browser/render smoke",
)
class HyperFramesCliSmokeTests(unittest.TestCase):
    def test_checked_template_with_pinned_local_cli(self) -> None:
        command = hyperframes_visual.resolve_hyperframes_command() + [
            "check",
            "--json",
            "--snapshots",
            "--at",
            "1,5,9,11.5",
            "--timeout",
            "30000",
        ]
        joined = " ".join(command).lower()
        for forbidden in ("npx", "pnpx", "dlx", "http://", "https://"):
            self.assertNotIn(forbidden, joined)

        env = hyperframes_visual.runtime_paths.subprocess_env()
        result = subprocess.run(
            command,
            cwd=hyperframes_visual.HYPERFRAMES_TEMPLATE,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, (result.stdout or "")[-4000:])


if __name__ == "__main__":
    unittest.main()
