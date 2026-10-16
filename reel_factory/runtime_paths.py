"""Cross-platform resolution of external binaries and fonts.

Order of preference everywhere: explicit env var -> project-local ``.tools`` -> ``PATH``.

Project-local layout (created by ``setup_mac.sh`` / ``setup_windows.ps1``)::

    .tools/ffmpeg/ffmpeg[.exe], ffprobe[.exe]
    .tools/node/bin/node            (macOS)   |  .tools/node/node.exe  (Windows)
    .tools/broadcast-motion-runtime-full/node_modules/hyperframes/...
    .tools/fonts/                   (Arial family with Cyrillic, for libass subtitles)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"
EXE = ".exe" if IS_WINDOWS else ""

def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _first_file(candidates) -> Path | None:
    for candidate in candidates:
        if candidate and _is_file(Path(candidate)):
            return Path(candidate)
    return None


def dotenv_value(name: str) -> str | None:
    """Value of ``NAME=...`` from ``<repo>/.env`` (empty values count as unset)."""
    try:
        for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line.startswith(name + "="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                return value or None
    except OSError:
        pass
    return None


def env_value(name: str) -> str | None:
    """Environment variable, falling back to ``<repo>/.env``."""
    return (os.environ.get(name, "").strip() or None) or dotenv_value(name)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def _which(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found) if found else None


def ffmpeg_path() -> str:
    found = _first_file([
        _env_path("REEL_FFMPEG"),
        TOOLS / "ffmpeg" / f"ffmpeg{EXE}",
        TOOLS / "media-bin" / f"ffmpeg{EXE}",
        _which("ffmpeg"),
    ])
    if found:
        return str(found)
    try:  # last resort: the wheel bundled with imageio-ffmpeg, if installed
        import imageio_ffmpeg  # type: ignore
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    raise RuntimeError(
        "FFmpeg не найден. Запустите установщик: «Установить (Mac).command» / "
        "«Установить (Windows).cmd» (ставит FFmpeg в .tools/ffmpeg)."
    )


def ffprobe_path() -> str | None:
    found = _first_file([
        _env_path("REEL_FFPROBE"),
        TOOLS / "ffmpeg" / f"ffprobe{EXE}",
        TOOLS / "media-bin" / f"ffprobe{EXE}",
        _which("ffprobe"),
    ])
    if found:
        return str(found)
    try:
        sibling = Path(ffmpeg_path()).with_name(f"ffprobe{EXE}")
    except RuntimeError:
        return None
    return str(sibling) if _is_file(sibling) else None


def node_path() -> str | None:
    unix_node = TOOLS / "node" / "bin" / "node"
    win_node = TOOLS / "node" / "node.exe"
    found = _first_file([
        _env_path("REEL_NODE"),
        unix_node,
        win_node,
        _which("node"),
    ])
    return str(found) if found else None


def codex_command() -> list[str] | None:
    """Command prefix that launches the Codex CLI, or None if it is not installed."""
    configured = _env_path("REEL_CODEX_BIN")
    if configured and _is_file(configured):
        return [str(configured)]
    local_js = TOOLS / "codex-cli" / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    node = node_path()
    if node and _is_file(local_js):
        return [node, str(local_js)]
    on_path = _which("codex")
    if on_path:
        return [str(on_path)]
    return None


def claude_path() -> str | None:
    found = _first_file([
        _env_path("REEL_CLAUDE_BIN"),
        _which("claude"),
        Path.home() / ".local" / "bin" / "claude",
        Path.home() / ".local" / "bin" / "claude.exe",
        Path(os.environ["APPDATA"]) / "npm" / "claude.cmd" if os.environ.get("APPDATA") else None,
        Path.home() / ".claude" / "local" / "claude",
        Path("/opt/homebrew/bin/claude"),
        Path("/usr/local/bin/claude"),
    ])
    return str(found) if found else None


def fonts_dir() -> Path | None:
    """Directory handed to the libass ``subtitles`` filter (must contain Arial with Cyrillic)."""
    configured = _env_path("REEL_FONTS_DIR")
    candidates = [
        configured,
        TOOLS / "fonts",  # curated dir created by the installer (Arial family only)
        ROOT / "assets" / "fonts",  # optional bundled fonts
        Path("C:/Windows/Fonts") if IS_WINDOWS else None,
        Path("/System/Library/Fonts/Supplemental") if IS_MAC else None,
        Path("/Library/Fonts") if IS_MAC else None,
        Path("/usr/share/fonts/truetype/msttcorefonts"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts"),
    ]
    for candidate in candidates:
        try:
            if candidate and candidate.is_dir():
                return candidate
        except OSError:
            continue
    return None


def ffmpeg_filter_path(path: Path | str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter option value."""
    return str(path).replace("\\", "/").replace(":", "\\:")


def subtitles_fontsdir_option() -> str:
    """Return ``:fontsdir='...'`` for the subtitles filter, or '' when no dir is known."""
    directory = fonts_dir()
    if directory is None:
        return ""
    return f":fontsdir='{ffmpeg_filter_path(directory.resolve())}'"


def subprocess_env() -> dict[str, str]:
    """Environment with project-local node/ffmpeg directories first on PATH."""
    env = os.environ.copy()
    extra = []
    node = node_path()
    if node:
        extra.append(str(Path(node).parent))
    try:
        extra.append(str(Path(ffmpeg_path()).parent))
    except RuntimeError:
        pass
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def describe() -> dict:
    def safe(fn):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - diagnostics only
            return f"ERROR: {exc}"
    return {
        "ffmpeg": safe(ffmpeg_path),
        "ffprobe": safe(ffprobe_path),
        "node": safe(node_path),
        "codex": safe(codex_command),
        "claude": safe(claude_path),
        "fonts_dir": str(safe(fonts_dir)),
    }
