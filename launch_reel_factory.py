from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

from reel_factory.runtime_identity import BUILD_ID, WORKSPACE_ID

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}/"


def _endpoint(url: str, path: str) -> str:
    return f"{url.rstrip('/')}/{path.lstrip('/')}"


def fetch_health(url: str = URL) -> dict | None:
    try:
        request = urllib.request.Request(
            _endpoint(url, "api/health"),
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
        )
        with urllib.request.urlopen(request, timeout=1.0) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def is_running() -> bool:
    """Compatibility helper retained for existing shortcuts and callers."""
    return fetch_health() is not None


def is_current_instance(health: dict | None) -> bool:
    return bool(
        health
        and health.get("ok") is True
        and health.get("service") == "Reel Factory"
        and health.get("workspace_id") == WORKSPACE_ID
        and health.get("build_id") == BUILD_ID
        and isinstance(health.get("pid"), int)
        and health["pid"] > 0
        and isinstance(health.get("busy"), bool)
    )


def _is_owned_instance(health: dict | None) -> bool:
    """Only an instance from this exact checkout is eligible for shutdown."""
    return bool(
        health
        and health.get("ok") is True
        and health.get("service") == "Reel Factory"
        and health.get("workspace_id") == WORKSPACE_ID
        and isinstance(health.get("build_id"), str)
        and isinstance(health.get("pid"), int)
        and health["pid"] > 0
        and isinstance(health.get("busy"), bool)
    )


def request_shutdown(health: dict, url: str = URL) -> bool:
    if not _is_owned_instance(health):
        return False
    payload = json.dumps({
        "pid": health["pid"],
        "workspace_id": health["workspace_id"],
        "build_id": health["build_id"],
    }).encode("utf-8")
    request = urllib.request.Request(
        _endpoint(url, "api/shutdown"),
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
            "X-Reel-Factory-Launcher": "1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status == 202 and body.get("stopping_pid") == health["pid"]
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _port_is_free(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def wait_for_port_release(host: str = HOST, port: int = PORT, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() <= deadline:
        if _port_is_free(host, port):
            return True
        time.sleep(0.1)
    return False


def start_server() -> subprocess.Popen:
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    with (
        (ROOT / "server.out.log").open("ab", buffering=0) as output,
        (ROOT / "server.err.log").open("ab", buffering=0) as errors,
    ):
        return subprocess.Popen(
            [sys.executable, str(ROOT / "app.py"), "--no-browser", "--port", str(PORT)],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=errors,
            close_fds=True,
            creationflags=creation_flags,
            **({} if sys.platform == "win32" else {"start_new_session": True}),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    health = fetch_health()
    if health and not is_current_instance(health):
        if not _is_owned_instance(health):
            print(
                f"Порт {PORT} занят другим или устаревшим неидентифицируемым процессом. "
                "Он не был остановлен автоматически."
            )
            return 1
        old_pid = health["pid"]
        if health.get("busy") is True:
            print(
                f"Reel Factory PID {old_pid} занят обработкой. "
                "Дождитесь завершения и снова запустите ярлык для обновления."
            )
            return 2
        print(f"Обновляю Reel Factory: останавливаю старую сборку (PID {old_pid})…")
        if not request_shutdown(health) or not wait_for_port_release():
            print(f"Не удалось корректно остановить Reel Factory PID {old_pid}.")
            return 1
        health = None

    if health is None:
        if not wait_for_port_release(timeout=0.2):
            print(f"Порт {PORT} занят чужим процессом. Он не был остановлен.")
            return 1
        process = start_server()
        for _ in range(60):
            health = fetch_health()
            if is_current_instance(health):
                break
            if process.poll() is not None and health is None:
                break
            time.sleep(0.25)
        else:
            health = None
        if not is_current_instance(health):
            print("Reel Factory не запустился с текущей сборкой. Проверьте server.err.log.")
            return 1

    if not args.no_open:
        webbrowser.open(URL)
    print(URL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
