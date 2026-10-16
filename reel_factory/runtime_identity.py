from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BUILD_FORMAT_VERSION = b"reel-factory-build-v1\0"


def _build_inputs(root: Path) -> tuple[Path, ...]:
    """Return only source files whose loaded contents define the app runtime."""
    paths = [root / "app.py", root / "launch_reel_factory.py"]
    package = root / "reel_factory"
    if package.is_dir():
        paths.extend(package.rglob("*.py"))
    return tuple(sorted((path for path in paths if path.is_file()), key=lambda item: item.as_posix()))


def compute_build_id(root: Path | None = None) -> str:
    """Hash runtime source contents, independent of mtimes and directory ordering."""
    resolved_root = (root or PROJECT_ROOT).resolve()
    digest = hashlib.sha256(_BUILD_FORMAT_VERSION)
    for path in _build_inputs(resolved_root):
        relative = path.relative_to(resolved_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()[:20]


def workspace_id(root: Path | None = None) -> str:
    """Identify this checkout without exposing its local absolute path over HTTP."""
    resolved = (root or PROJECT_ROOT).resolve().as_posix().casefold().encode("utf-8")
    return hashlib.sha256(b"reel-factory-workspace-v1\0" + resolved).hexdigest()[:20]


def artifact_version(path: Path) -> str:
    """Create a cheap cache key that changes whenever an artifact is replaced."""
    stat = path.stat()
    return f"{stat.st_mtime_ns:x}-{stat.st_size:x}"


def artifact_set_version(paths: Iterable[Path]) -> str:
    """Create a stable cache key for a ZIP assembled from several artifacts."""
    digest = hashlib.sha256(b"reel-factory-artifacts-v1\0")
    found = False
    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        try:
            version = artifact_version(path)
        except OSError:
            continue
        found = True
        digest.update(path.name.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
        digest.update(version.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:20] if found else BUILD_ID


def versioned_url(url: str, version: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={version}"


# These values are deliberately frozen at import time. An already-running Python
# process must keep reporting the code it actually loaded, even after files on
# disk are edited underneath it.
BUILD_ID = compute_build_id()
WORKSPACE_ID = workspace_id()

