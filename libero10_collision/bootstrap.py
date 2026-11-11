"""Fetch the exact public LIBERO source and assets used by this benchmark."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


LIBERO_URL = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"


def default_libero_root() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache.expanduser() / "libero10-collision" / "libero"


def _git(*arguments: str, cwd: Path | None = None) -> str:
    if shutil.which("git") is None:
        raise RuntimeError("Git is required. Install git, then run setup again.")
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Git command failed: git {' '.join(arguments)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def verify_libero_checkout(path: str | Path) -> dict:
    """Verify the source revision and that tracked code/assets are unmodified.

    A different checkout can still be selected explicitly with LIBERO_ROOT, but
    this setup command never overwrites it or declares it release-compatible.
    """
    root = Path(path).expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Not a Git checkout: {root}. Choose an empty setup destination.")
    revision = _git("rev-parse", "HEAD", cwd=root)
    if revision != LIBERO_REVISION:
        raise RuntimeError(f"LIBERO revision mismatch at {root}: expected {LIBERO_REVISION}, got {revision}.")
    changes = _git("status", "--porcelain", "--untracked-files=no", cwd=root)
    if changes:
        raise RuntimeError(f"LIBERO has modified tracked files at {root}. Choose a new setup destination.\n{changes}")
    required = ("libero/libero/__init__.py", "libero/libero/assets",
                "libero/libero/bddl_files/libero_10", "libero/libero/init_files/libero_10")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"Incomplete LIBERO checkout at {root}: missing {', '.join(missing)}")
    return {"root": str(root), "repository": LIBERO_URL,
            "revision": revision, "tracked_files_clean": True, "patches": []}


def setup_libero(destination: str | Path | None = None) -> Path:
    """Install the pinned checkout, or verify an existing installation.

    Fetches only the pinned commit into a temporary sibling directory, then
    renames it into place. An existing destination is never removed or reset.
    Python packages are installed separately with pip; no global config changes.
    """
    root = (Path(destination) if destination is not None else default_libero_root()).expanduser().resolve()
    if root.exists():
        verify_libero_checkout(root)
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}-setup-", dir=root.parent))
    try:
        _git("init", "--quiet", str(temporary))
        _git("remote", "add", "origin", LIBERO_URL, cwd=temporary)
        _git("fetch", "--depth", "1", "origin", LIBERO_REVISION, cwd=temporary)
        _git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=temporary)
        verify_libero_checkout(temporary)
        if root.exists():
            # Another setup process may have completed while this fetch ran.
            verify_libero_checkout(root)
        else:
            temporary.rename(root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return root
