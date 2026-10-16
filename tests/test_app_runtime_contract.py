from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import unittest
import uuid
import warnings
from contextlib import contextmanager
from email.utils import formatdate
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from reel_factory.runtime_identity import artifact_set_version, artifact_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"


@contextmanager
def isolated_app_module():
    """Load app.py with a temporary ROOT so manager restore cannot touch real projects."""
    with tempfile.TemporaryDirectory() as raw_tmp:
        temp_root = Path(raw_tmp)
        copied_app = temp_root / "app.py"
        shutil.copyfile(APP_PATH, copied_app)
        module_name = f"_reel_factory_isolated_app_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, copied_app)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not create isolated app module spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                spec.loader.exec_module(module)
            yield module
        finally:
            manager = getattr(module, "MANAGER", None)
            if manager is not None:
                manager.executor.shutdown(wait=True, cancel_futures=True)
            sys.modules.pop(module_name, None)


class AppRuntimeContractTests(unittest.TestCase):
    def test_health_exposes_frozen_identity_pid_and_busy_state(self) -> None:
        with isolated_app_module() as app:
            captured: dict = {}
            handler = object.__new__(app.AppHandler)
            handler.path = "/api/health"
            handler.send_json = lambda data, status=HTTPStatus.OK: captured.update(
                {"data": data, "status": status}
            )

            handler.do_GET()

            self.assertEqual(captured["status"], HTTPStatus.OK)
            self.assertEqual(captured["data"], {
                "ok": True,
                "service": "Reel Factory",
                "build_id": app.BUILD_ID,
                "workspace_id": app.WORKSPACE_ID,
                "pid": os.getpid(),
                "busy": False,
            })

    def test_http_server_exclusively_owns_an_ephemeral_listener(self) -> None:
        with isolated_app_module() as app:
            server = app.ReelFactoryHTTPServer(("127.0.0.1", 0), app.AppHandler)
            competitor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.assertFalse(server.allow_reuse_address)
                with self.assertRaises(OSError):
                    competitor.bind(server.server_address)
            finally:
                competitor.close()
                server.server_close()

    def test_busy_shutdown_returns_conflict_before_touching_server(self) -> None:
        with isolated_app_module() as app:
            payload = json.dumps({
                "pid": os.getpid(),
                "workspace_id": app.WORKSPACE_ID,
                "build_id": app.BUILD_ID,
            }).encode("utf-8")
            captured: dict = {}
            handler = object.__new__(app.AppHandler)
            handler.path = "/api/shutdown"
            handler.client_address = ("127.0.0.1", 50000)
            handler.headers = {
                "X-Reel-Factory-Launcher": "1",
                "Content-Length": str(len(payload)),
            }
            handler.rfile = io.BytesIO(payload)
            handler.send_error_json = lambda message, status=HTTPStatus.BAD_REQUEST: captured.update(
                {"message": message, "status": status}
            )

            with patch.object(app, "runtime_busy", return_value=True) as busy:
                handler.do_POST()

            self.assertEqual(captured["status"], HTTPStatus.CONFLICT)
            self.assertIn("занят", captured["message"])
            busy.assert_called_once_with()

    def test_media_response_is_no_store_and_version_identified(self) -> None:
        with isolated_app_module() as app:
            media = app.ROOT / "sample.mp4"
            media.write_bytes(b"media-payload")
            response: dict = {"headers": {}}
            handler = object.__new__(app.AppHandler)
            handler.headers = {}
            handler.wfile = io.BytesIO()
            handler.send_response = lambda status: response.update({"status": status})
            handler.send_header = lambda name, value: response["headers"].update({name: value})
            handler.end_headers = lambda: None

            handler.send_file(media, "sample.mp4", "video/mp4", attachment=False)

            headers = response["headers"]
            self.assertEqual(response["status"], HTTPStatus.OK)
            self.assertEqual(headers["Cache-Control"], "no-store, no-cache, must-revalidate, max-age=0")
            self.assertEqual(headers["Pragma"], "no-cache")
            self.assertEqual(headers["Expires"], "0")
            self.assertEqual(headers["Last-Modified"], formatdate(media.stat().st_mtime, usegmt=True))
            self.assertEqual(headers["ETag"], f'"{artifact_version(media)}"')
            self.assertEqual(handler.wfile.getvalue(), b"media-payload")

    def test_public_media_urls_change_when_completed_output_is_replaced(self) -> None:
        with isolated_app_module() as app:
            project_id = "h00000000001"
            project_dir = app.HIGHLIGHTS_DIR / project_id
            output = project_dir / "output" / "c001.mp4"
            thumbnail = project_dir / "thumbnails" / "c001.jpg"
            output.parent.mkdir(parents=True)
            thumbnail.parent.mkdir(parents=True)
            output.write_bytes(b"old-output")
            thumbnail.write_bytes(b"thumbnail")
            initial_ns = time.time_ns() - 10_000_000_000
            os.utime(output, ns=(initial_ns, initial_ns))
            project = {
                "id": project_id,
                "created_at": "2026-08-20T00:00:00+00:00",
                "updated_at": "2026-08-20T00:00:00+00:00",
                "status": "completed",
                "project_mode": "broadcast",
                "options": {"subtitle_style": "pill"},
                "input": "input/source.mp4",
                "transcript": None,
                "youtube_url": "",
                "error_stage": None,
                "candidates": [{
                    "id": "c001",
                    "status": "completed",
                    "output": "output/c001.mp4",
                    "result": {"visual_engine": "hyperframes"},
                }],
            }

            before = app.HIGHLIGHTS.public(project)
            output.write_bytes(b"new-output-with-a-different-size")
            replacement_ns = time.time_ns()
            os.utime(output, ns=(replacement_ns, replacement_ns))
            after = app.HIGHLIGHTS.public(project)

            before_candidate = before["candidates"][0]
            after_candidate = after["candidates"][0]
            self.assertIn(f"?v={artifact_version(thumbnail)}", before_candidate["thumbnail_url"])
            self.assertNotEqual(before_candidate["download_url"], after_candidate["download_url"])
            self.assertNotEqual(before["zip_url"], after["zip_url"])
            self.assertIn(f"?v={artifact_version(output)}", after_candidate["download_url"])
            self.assertIn(
                f"?v={artifact_set_version([output])}",
                after["zip_url"],
            )

    def test_suggested_candidate_exposes_versioned_source_download(self) -> None:
        with isolated_app_module() as app:
            project_id = "h00000000004"
            project_dir = app.HIGHLIGHTS_DIR / project_id
            source = project_dir / "input" / "source.mkv"
            thumbnail = project_dir / "thumbnails" / "c001.jpg"
            source.parent.mkdir(parents=True)
            thumbnail.parent.mkdir(parents=True)
            source.write_bytes(b"source-video")
            thumbnail.write_bytes(b"thumbnail")
            project = {
                "id": project_id,
                "created_at": "2026-08-20T00:00:00+00:00",
                "updated_at": "2026-08-20T00:00:00+00:00",
                "status": "awaiting_selection",
                "project_mode": "broadcast",
                "options": {"subtitle_style": "pill"},
                "input": "input/source.mkv",
                "transcript": None,
                "youtube_url": "",
                "error_stage": None,
                "candidates": [{
                    "id": "c001", "status": "suggested", "start": 12.345, "end": 45.678,
                }],
            }

            public = app.HIGHLIGHTS.public(project)

            source_url = public["candidates"][0]["source_download_url"]
            self.assertIn(f"/api/highlights/{project_id}/source-files/c001", source_url)
            self.assertIn(artifact_version(source), source_url)
            self.assertNotIn("download_url", public["candidates"][0])

    def test_source_download_creates_unprocessed_clip_once_and_serves_it(self) -> None:
        with isolated_app_module() as app:
            project_id = "h00000000005"
            project_dir = app.HIGHLIGHTS_DIR / project_id
            source = project_dir / "input" / "source.mkv"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"source-video")
            project = {
                "id": project_id,
                "input": "input/source.mkv",
                "project_mode": "broadcast",
                "candidates": [{
                    "id": "c001", "status": "suggested", "start": 12.345, "end": 45.678,
                    "title": "Сильный момент",
                }],
            }
            app.HIGHLIGHTS.projects[project_id] = project
            handler = object.__new__(app.AppHandler)
            served: list[tuple[Path, str, str]] = []
            handler.send_file = lambda path, name, content_type: served.append((path, name, content_type))

            def fake_extract(source_path, destination, start, end):
                self.assertEqual(source_path, source)
                self.assertEqual((start, end), (12.345, 45.678))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"source-fragment")

            with patch.object(app, "extract_source_clip", side_effect=fake_extract) as extract:
                handler.send_highlight_source_file(project_id, "c001")
                handler.send_highlight_source_file(project_id, "c001")

            extract.assert_called_once()
            self.assertEqual(len(served), 2)
            self.assertEqual(served[0][0].read_bytes(), b"source-fragment")
            self.assertEqual(served[0][1], "Source_c001_Сильный_момент.mkv")
            self.assertEqual(served[0][2], "video/x-matroska")

    def test_start_render_clears_selected_legacy_metadata_but_preserves_old_file(self) -> None:
        with isolated_app_module() as app:
            project_id = "h00000000002"
            project_dir = app.HIGHLIGHTS_DIR / project_id
            old_output = project_dir / "output" / "c001.mp4"
            other_output = project_dir / "output" / "c002.mp4"
            old_output.parent.mkdir(parents=True)
            old_output.write_bytes(b"last-known-good")
            other_output.write_bytes(b"other-good")
            selected = {
                "id": "c001",
                "status": "completed",
                "selected": True,
                "output": "output/c001.mp4",
                "artifact_version": "legacy-version",
                "result": {"layout": "broadcast"},
                "error": "legacy error",
            }
            unselected = {
                "id": "c002",
                "status": "completed",
                "selected": False,
                "output": "output/c002.mp4",
                "artifact_version": "keep-version",
                "result": {"visual_engine": "hyperframes"},
                "error": None,
            }
            project = {
                "id": project_id,
                "created_at": "2026-08-20T00:00:00+00:00",
                "updated_at": "2026-08-20T00:00:00+00:00",
                "status": "completed",
                "progress": 100,
                "paused": False,
                "project_mode": "broadcast",
                "options": {"subtitle_style": "pill"},
                "input": "input/source.mp4",
                "transcript": None,
                "youtube_url": "",
                "error_stage": None,
                "candidates": [selected, unselected],
            }
            app.HIGHLIGHTS.projects[project_id] = project

            with (
                patch.object(app.HIGHLIGHTS.executor, "submit") as submit,
                patch.object(app, "provider_status", return_value={"codex": {"ready": True}}),
            ):
                public = app.HIGHLIGHTS.start_render(
                    project_id, ["c001"], "clean", 1.2, 1, "itbooster",
                )

            self.assertEqual(selected["status"], "queued")
            self.assertEqual(selected["result"], None)
            self.assertEqual(selected["error"], None)
            self.assertNotIn("output", selected)
            self.assertNotIn("artifact_version", selected)
            self.assertTrue(old_output.is_file())
            self.assertEqual(old_output.read_bytes(), b"last-known-good")
            self.assertEqual(unselected["result"], {"visual_engine": "hyperframes"})
            self.assertEqual(unselected["output"], "output/c002.mp4")
            self.assertNotIn("download_url", public["candidates"][0])
            self.assertEqual(project["options"]["playback_speed"], 1.2)
            self.assertEqual(project["options"]["speaker_count"], 1)
            self.assertEqual(project["options"]["visual_preset"], "itbooster")
            submit.assert_called_once()

    def test_rejected_candidate_can_be_rendered_anyway_and_video_review_option_is_stored(self) -> None:
        with isolated_app_module() as app:
            project_id = "h00000000007"
            project = {
                "id": project_id, "created_at": "2026-08-20T00:00:00+00:00", "updated_at": "2026-08-20T00:00:00+00:00",
                "status": "awaiting_selection", "progress": 100, "paused": False, "project_mode": "podcast",
                "options": {"subtitle_style": "pill"}, "input": "input/source.mp4", "transcript": "work/transcript.json",
                "candidates": [
                    {"id": "c001", "status": "suggested", "selected": False, "rejected": None},
                    {"id": "c002", "status": "suggested", "selected": False,
                     "rejected": {"stage": "chief_editor", "reason": "дубль", "scores": {}}},
                ],
            }
            app.HIGHLIGHTS.projects[project_id] = project
            with patch.object(app.HIGHLIGHTS.executor, "submit") as submit:
                app.HIGHLIGHTS.start_render(project_id, ["c002"], "pill", 1.0, 2, "editorial_lime", False)
            forced = project["candidates"][1]
            self.assertEqual(forced["status"], "queued")
            self.assertTrue(forced["rejected_forced"])
            self.assertEqual(forced["rejected"]["reason"], "дубль")        # причина отсева сохраняется
            self.assertIs(project["options"]["video_review"], False)
            self.assertNotIn("rejected_forced", project["candidates"][0])
            submit.assert_called_once()

    def test_options_default_video_review_on(self) -> None:
        with isolated_app_module() as app:
            self.assertIs(app.HIGHLIGHTS._options({})["video_review"], True)
            self.assertIs(app.HIGHLIGHTS._options({"video_review": "false"})["video_review"], False)

    def test_render_applies_autofix_candidate_update_and_stores_review(self) -> None:
        with isolated_app_module() as app:
            project_id = "h00000000008"
            project_dir = app.HIGHLIGHTS_DIR / project_id
            (project_dir / "input").mkdir(parents=True)
            (project_dir / "work").mkdir(parents=True)
            (project_dir / "input" / "source.mp4").write_bytes(b"s")
            (project_dir / "work" / "transcript.json").write_text(json.dumps({"words": []}), encoding="utf-8")
            candidate = {"id": "c001", "start": 0.0, "end": 3.0, "status": "queued", "selected": True, "segments": [1]}
            project = {"id": project_id, "created_at": "x", "updated_at": "x", "status": "rendering", "progress": 0,
                       "paused": False, "project_mode": "podcast", "options": {"video_review": True},
                       "input": "input/source.mp4", "transcript": "work/transcript.json", "candidates": [candidate]}
            app.HIGHLIGHTS.projects[project_id] = project

            def fake(_s, _t, _c, output, *_a, **_k):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"ok")
                return {"review": {"verdict": "publish"}, "candidate_update": {"segments": [2], "duration": 12.0}}

            with patch.object(app, "render_highlight_reviewed", side_effect=fake):
                app.HIGHLIGHTS._render(project_id, ["c001"])
            self.assertEqual(candidate["status"], "completed", candidate.get("error"))
            self.assertEqual(candidate["segments"], [2])
            self.assertEqual(candidate["result"]["review"]["verdict"], "publish")

    def test_failed_project_with_candidates_can_start_another_render(self) -> None:
        with isolated_app_module() as app:
            project_id = "h00000000003"
            project = {
                "id": project_id,
                "created_at": "2026-08-20T00:00:00+00:00",
                "updated_at": "2026-08-20T00:00:00+00:00",
                "status": "failed",
                "progress": 100,
                "paused": False,
                "project_mode": "broadcast",
                "options": {"subtitle_style": "pill"},
                "input": "input/source.mp4",
                "transcript": "work/transcript.json",
                "youtube_url": "",
                "error_stage": "rendering",
                "candidates": [
                    {"id": "c001", "status": "failed", "selected": True, "error": "render failed"},
                    {"id": "c002", "status": "suggested", "selected": False, "error": None},
                ],
            }
            app.HIGHLIGHTS.projects[project_id] = project

            with (
                patch.object(app.HIGHLIGHTS.executor, "submit") as submit,
                patch.object(app, "provider_status", return_value={"codex": {"ready": True}}),
            ):
                public = app.HIGHLIGHTS.start_render(project_id, ["c002"], "pill")

            self.assertEqual(public["status"], "rendering")
            self.assertEqual(project["candidates"][1]["status"], "queued")
            submit.assert_called_once()

    def test_render_automatically_retries_twice_and_succeeds_on_third_attempt(self) -> None:
        with isolated_app_module() as app:
            project_id = "h00000000006"
            project_dir = app.HIGHLIGHTS_DIR / project_id
            source = project_dir / "input" / "source.mp4"
            transcript_path = project_dir / "work" / "transcript.json"
            source.parent.mkdir(parents=True)
            transcript_path.parent.mkdir(parents=True)
            source.write_bytes(b"source")
            transcript_path.write_text(json.dumps({"words": []}), encoding="utf-8")
            candidate = {
                "id": "c001", "start": 0.0, "end": 3.0,
                "status": "queued", "selected": True,
            }
            project = {
                "id": project_id,
                "created_at": "2026-08-21T00:00:00+00:00",
                "updated_at": "2026-08-21T00:00:00+00:00",
                "status": "rendering",
                "progress": 0,
                "paused": False,
                "project_mode": "broadcast",
                "options": {"project_mode": "broadcast", "subtitle_style": "pill"},
                "input": "input/source.mp4",
                "transcript": "work/transcript.json",
                "error_stage": "rendering",
                "candidates": [candidate],
            }
            app.HIGHLIGHTS.projects[project_id] = project
            attempts = []

            def flaky_render(_source, _transcript, _candidate, output, *_args):
                attempts.append(output)
                if len(attempts) < 3:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"partial")
                    raise app.PipelineError(f"transient-{len(attempts)}")
                output.write_bytes(b"success")
                return {"visual_engine": "hyperframes"}

            with patch.object(app, "render_highlight", side_effect=flaky_render) as render:
                app.HIGHLIGHTS._render(project_id, ["c001"])

            self.assertEqual(render.call_count, 3)
            self.assertEqual(candidate["status"], "completed")
            self.assertEqual(candidate["render_attempt"], 3)
            self.assertEqual(len(candidate["retry_errors"]), 2)
            self.assertEqual((project_dir / "output" / "c001.mp4").read_bytes(), b"success")
            self.assertEqual(list((project_dir / "output").glob("*.rendering.mp4")), [])

    def test_batch_file_render_uses_the_same_two_automatic_retries(self) -> None:
        with isolated_app_module() as app:
            job_id = "j00000000001"
            job_dir = app.JOBS_DIR / job_id
            source = job_dir / "input" / "v001.mp4"
            output = job_dir / "output" / "v001.mp4"
            source.parent.mkdir(parents=True)
            output.parent.mkdir(parents=True)
            source.write_bytes(b"source")
            item = {
                "id": "v001", "name": "source.mp4",
                "input": "input/v001.mp4", "output": "output/v001.mp4",
                "status": "queued", "progress": 0,
            }
            job = {
                "id": job_id, "status": "queued", "progress": 0,
                "paused": False, "options": {}, "files": [item],
            }
            app.MANAGER.jobs[job_id] = job
            attempts = []

            def flaky_process(_source, destination, *_args):
                attempts.append(destination)
                if len(attempts) < 3:
                    raise app.PipelineError(f"transient-{len(attempts)}")
                destination.write_bytes(b"success")
                return {"layout": "stack"}

            with patch.object(app, "process_video", side_effect=flaky_process) as process:
                app.MANAGER._process(job_id)

            self.assertEqual(process.call_count, 3)
            self.assertEqual(item["status"], "completed")
            self.assertEqual(item["render_attempt"], 3)
            self.assertEqual(len(item["retry_errors"]), 2)
            self.assertEqual(output.read_bytes(), b"success")

    def test_source_contract_stages_replacement_and_double_checks_busy_shutdown(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn(".rendering{output.suffix}", source)
        self.assertIn("os.replace(staged_output, output)", source)
        self.assertIn('candidate["artifact_version"] = artifact_version(output)', source)
        shutdown = source[source.index('if path == "/api/shutdown"'):]
        shutdown = shutdown[:shutdown.index("job_control =")]
        self.assertEqual(shutdown.count("if runtime_busy():"), 2)


if __name__ == "__main__":
    unittest.main()
