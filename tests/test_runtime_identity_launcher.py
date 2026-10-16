from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import launch_reel_factory
from reel_factory import runtime_identity


class RuntimeIdentityTests(unittest.TestCase):
    def test_build_id_is_content_based_stable_and_sensitive_to_runtime_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            package = root / "reel_factory"
            package.mkdir()
            (root / "app.py").write_text("APP = 1\n", encoding="utf-8")
            (root / "launch_reel_factory.py").write_text("LAUNCHER = 1\n", encoding="utf-8")
            module = package / "feature.py"
            module.write_text("FEATURE = 1\n", encoding="utf-8")

            first = runtime_identity.compute_build_id(root)
            os.utime(module, None)
            same_content = runtime_identity.compute_build_id(root)
            module.write_text("FEATURE = 2\n", encoding="utf-8")
            changed = runtime_identity.compute_build_id(root)

        self.assertRegex(first, r"^[0-9a-f]{20}$")
        self.assertEqual(first, same_content)
        self.assertNotEqual(first, changed)

    def test_workspace_id_does_not_expose_path_and_distinguishes_checkouts(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first = runtime_identity.workspace_id(Path(first_tmp))
            repeated = runtime_identity.workspace_id(Path(first_tmp))
            second = runtime_identity.workspace_id(Path(second_tmp))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^[0-9a-f]{20}$")
        self.assertNotIn(first_tmp.casefold(), first.casefold())

    def test_artifact_version_and_versioned_url_change_after_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "clip.mp4"
            path.write_bytes(b"old")
            old = runtime_identity.artifact_version(path)
            path.write_bytes(b"new-and-longer")
            new = runtime_identity.artifact_version(path)

        self.assertNotEqual(old, new)
        self.assertEqual(runtime_identity.versioned_url("/media/c001", new), f"/media/c001?v={new}")
        self.assertEqual(
            runtime_identity.versioned_url("/media/c001?download=1", new),
            f"/media/c001?download=1&v={new}",
        )


class LauncherIdentityDecisionTests(unittest.TestCase):
    @staticmethod
    def health(
        *,
        build_id: str | None = None,
        workspace_id: str | None = None,
        busy: bool = False,
    ) -> dict:
        return {
            "ok": True,
            "service": "Reel Factory",
            "build_id": build_id or launch_reel_factory.BUILD_ID,
            "workspace_id": workspace_id or launch_reel_factory.WORKSPACE_ID,
            "pid": 43210,
            "busy": busy,
        }

    def test_current_instance_requires_matching_build_workspace_and_pid(self) -> None:
        self.assertTrue(launch_reel_factory.is_current_instance(self.health()))
        self.assertFalse(launch_reel_factory.is_current_instance(self.health(build_id="old-build")))
        self.assertFalse(launch_reel_factory.is_current_instance(self.health(workspace_id="foreign")))
        self.assertFalse(launch_reel_factory.is_current_instance({**self.health(), "pid": 0}))
        self.assertFalse(launch_reel_factory.is_current_instance(None))

    def test_launcher_keeps_current_instance_without_restart(self) -> None:
        current = self.health()
        with (
            patch.object(launch_reel_factory.sys, "argv", ["launch_reel_factory.py", "--no-open"]),
            patch.object(launch_reel_factory, "fetch_health", return_value=current),
            patch.object(launch_reel_factory, "request_shutdown") as shutdown,
            patch.object(launch_reel_factory, "start_server") as start,
            patch("builtins.print"),
        ):
            result = launch_reel_factory.main()

        self.assertEqual(result, 0)
        shutdown.assert_not_called()
        start.assert_not_called()

    def test_launcher_replaces_stale_build_from_same_workspace(self) -> None:
        stale = self.health(build_id="stale-build")
        current = self.health()
        process = Mock()
        process.poll.return_value = None
        with (
            patch.object(launch_reel_factory.sys, "argv", ["launch_reel_factory.py", "--no-open"]),
            patch.object(launch_reel_factory, "fetch_health", side_effect=[stale, current]),
            patch.object(launch_reel_factory, "request_shutdown", return_value=True) as shutdown,
            patch.object(launch_reel_factory, "wait_for_port_release", return_value=True),
            patch.object(launch_reel_factory, "start_server", return_value=process) as start,
            patch.object(launch_reel_factory.time, "sleep"),
            patch("builtins.print"),
        ):
            result = launch_reel_factory.main()

        self.assertEqual(result, 0)
        shutdown.assert_called_once_with(stale)
        start.assert_called_once_with()

    def test_launcher_refuses_to_stop_foreign_workspace(self) -> None:
        foreign = self.health(workspace_id="foreign-workspace")
        with (
            patch.object(launch_reel_factory.sys, "argv", ["launch_reel_factory.py", "--no-open"]),
            patch.object(launch_reel_factory, "fetch_health", return_value=foreign),
            patch.object(launch_reel_factory, "request_shutdown") as shutdown,
            patch.object(launch_reel_factory, "start_server") as start,
            patch("builtins.print"),
        ):
            result = launch_reel_factory.main()

        self.assertEqual(result, 1)
        shutdown.assert_not_called()
        start.assert_not_called()

    def test_launcher_defers_stale_build_restart_while_server_is_busy(self) -> None:
        busy_stale = self.health(build_id="stale-build", busy=True)
        with (
            patch.object(launch_reel_factory.sys, "argv", ["launch_reel_factory.py", "--no-open"]),
            patch.object(launch_reel_factory, "fetch_health", return_value=busy_stale),
            patch.object(launch_reel_factory, "request_shutdown") as shutdown,
            patch.object(launch_reel_factory, "start_server") as start,
            patch("builtins.print"),
        ):
            result = launch_reel_factory.main()

        self.assertEqual(result, 2)
        shutdown.assert_not_called()
        start.assert_not_called()

    def test_shutdown_request_is_scoped_and_contains_reported_identity(self) -> None:
        current = self.health()

        class Response:
            status = 202

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def read() -> bytes:
                return json.dumps({"stopping_pid": current["pid"]}).encode("utf-8")

        with patch.object(launch_reel_factory.urllib.request, "urlopen", return_value=Response()) as open_url:
            accepted = launch_reel_factory.request_shutdown(current, "http://127.0.0.1:8765/")

        self.assertTrue(accepted)
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8765/api/shutdown")
        self.assertEqual(request.get_header("X-reel-factory-launcher"), "1")
        self.assertEqual(json.loads(request.data), {
            "pid": current["pid"],
            "workspace_id": current["workspace_id"],
            "build_id": current["build_id"],
        })

        foreign = self.health(workspace_id="foreign")
        with patch.object(launch_reel_factory.urllib.request, "urlopen") as open_url:
            self.assertFalse(launch_reel_factory.request_shutdown(foreign))
        open_url.assert_not_called()


if __name__ == "__main__":
    unittest.main()
