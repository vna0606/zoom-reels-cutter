from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def codex_home() -> Path:
    """Resolve the credential directory consistently for every bundled CLI call."""
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured)
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        return Path(user_profile) / ".codex"
    return Path.home() / ".codex"


def codex_sqlite_home() -> Path:
    configured = os.environ.get("CODEX_SQLITE_HOME")
    if configured:
        return Path(configured)
    return ROOT / "workspace" / "codex-state"


def codex_runtime_home() -> Path:
    """Return the app-owned CLI home, isolated from the live Codex Desktop profile."""
    configured = os.environ.get("REEL_FACTORY_CODEX_HOME")
    if configured:
        return Path(configured)
    return ROOT / "workspace" / "codex-runtime"


def _prepare_codex_runtime_home() -> Path:
    credentials_home = codex_home()
    runtime_home = codex_runtime_home()
    try:
        if runtime_home.resolve() == credentials_home.resolve():
            raise RuntimeError(
                "REEL_FACTORY_CODEX_HOME must be separate from the Codex Desktop profile."
            )
    except OSError:
        pass
    runtime_home.mkdir(parents=True, exist_ok=True)
    (runtime_home / "tmp" / "arg0").mkdir(parents=True, exist_ok=True)
    source_auth = credentials_home / "auth.json"
    runtime_auth = runtime_home / "auth.json"
    if not source_auth.is_file() and not runtime_auth.is_file():
        raise RuntimeError(
            f"Codex authorization file was not found at {source_auth}. Run `codex login` ."
        )
    if source_auth.is_file():
        should_copy = not runtime_auth.is_file()
        if not should_copy:
            try:
                should_copy = source_auth.stat().st_mtime_ns > runtime_auth.stat().st_mtime_ns
            except OSError:
                should_copy = True
        if should_copy:
            temporary_auth = runtime_home / "auth.json.tmp"
            shutil.copyfile(source_auth, temporary_auth)
            temporary_auth.replace(runtime_auth)
    return runtime_home


def codex_environment() -> dict[str, str]:
    state_home = codex_sqlite_home()
    state_home.mkdir(parents=True, exist_ok=True)
    runtime_home = _prepare_codex_runtime_home()
    process_temp = runtime_home / "tmp" / "process"
    process_temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CODEX_HOME"] = str(runtime_home)
    env["CODEX_SQLITE_HOME"] = str(state_home)
    env["TEMP"] = str(process_temp)
    env["TMP"] = str(process_temp)
    env["TMPDIR"] = str(process_temp)
    return env
