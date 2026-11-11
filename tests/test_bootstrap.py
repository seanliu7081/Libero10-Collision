"""Setup protects existing directories and checks revision plus tracked integrity."""

from pathlib import Path
import shutil
import subprocess

import pytest

from libero10_collision import bootstrap


def test_existing_unrelated_directory_is_never_removed(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    sentinel = root / "keep-me.txt"
    sentinel.write_text("user data")
    with pytest.raises(RuntimeError, match="Not a Git checkout"):
        bootstrap.setup_libero(root)
    assert sentinel.read_text() == "user data"
    assert list(root.iterdir()) == [sentinel]


def test_xdg_cache_controls_default_location(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "custom cache"))
    assert bootstrap.default_libero_root() == tmp_path / "custom cache/libero10-collision/libero"


@pytest.fixture
def local_checkout(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("Git is needed for source checkout verification tests")
    root = tmp_path / "LIBERO"
    base = root / "libero/libero"
    for relative in ("assets", "bddl_files/libero_10", "init_files/libero_10"):
        directory = base / relative
        directory.mkdir(parents=True)
        (directory / ".keep").write_text("")
    (base / "__init__.py").write_text("# source fixture\n")
    def git(*arguments):
        return subprocess.check_output(["git", "-c", "user.name=Test",
                                        "-c", "user.email=tests@example.invalid", *arguments],
                                       cwd=root, text=True).strip()
    git("init", "--quiet")
    git("add", ".")
    git("commit", "--quiet", "-m", "Fixture source")
    revision = git("rev-parse", "HEAD")
    monkeypatch.setattr(bootstrap, "LIBERO_REVISION", revision)
    return root, base, revision


def test_setup_is_idempotent_for_verified_existing_checkout(local_checkout):
    root, _, revision = local_checkout
    assert bootstrap.setup_libero(root) == root
    report = bootstrap.verify_libero_checkout(root)
    assert report["revision"] == revision
    assert report["tracked_files_clean"] is True
    assert report["patches"] == []


def test_setup_rejects_wrong_revision_without_changing_checkout(local_checkout, monkeypatch):
    root, base, _ = local_checkout
    original = (base / "__init__.py").read_bytes()
    monkeypatch.setattr(bootstrap, "LIBERO_REVISION", "0" * 40)
    with pytest.raises(RuntimeError, match="revision mismatch"):
        bootstrap.setup_libero(root)
    assert (base / "__init__.py").read_bytes() == original


def test_setup_preserves_and_rejects_modified_tracked_files(local_checkout):
    root, base, _ = local_checkout
    source = base / "__init__.py"
    source.write_text("# user modification\n")
    with pytest.raises(RuntimeError, match="modified tracked files"):
        bootstrap.setup_libero(root)
    assert source.read_text() == "# user modification\n"
