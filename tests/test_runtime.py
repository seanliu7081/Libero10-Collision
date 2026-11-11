"""Loader validation and resource-lifetime contracts without the simulator."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from libero10_collision import runtime


@pytest.fixture
def sample_root(tmp_path):
    source = runtime.get_data_root()
    target = tmp_path / "benchmark with spaces"
    directory = target / "tasks" / "t00" / "episodes" / "ep000"
    shutil.copytree(source / "tasks" / "t00" / "episodes" / "ep000", directory)
    shutil.copy2(source / "benchmark.json", target / "benchmark.json")
    return target


def edit_manifest(root, edit):
    path = root / "tasks" / "t00" / "episodes" / "ep000" / "episode.json"
    metadata = json.loads(path.read_text())
    edit(metadata)
    path.write_text(json.dumps(metadata))


def test_data_root_override_takes_precedence_over_environment(monkeypatch, sample_root, tmp_path):
    monkeypatch.setenv("LIBERO10_COLLISION_ROOT", str(tmp_path / "missing"))
    assert runtime.get_data_root(sample_root) == sample_root.resolve()
    with pytest.raises(FileNotFoundError, match="LIBERO10_COLLISION_ROOT"):
        runtime.get_data_root()
    monkeypatch.setenv("LIBERO10_COLLISION_ROOT", str(sample_root))
    assert runtime.get_data_root() == sample_root.resolve()
    with pytest.raises(FileNotFoundError):
        runtime.get_data_root(tmp_path / "missing-explicit")


def test_wheel_data_fallback_is_independent_of_working_directory(monkeypatch, tmp_path):
    package = tmp_path / "site-packages" / "libero10_collision"
    data = package / "data"
    (data / "tasks").mkdir(parents=True)
    (data / "benchmark.json").write_text("{}")
    monkeypatch.delenv("LIBERO10_COLLISION_ROOT", raising=False)
    monkeypatch.setattr(runtime, "__file__", str(package / "runtime.py"))
    monkeypatch.chdir(tmp_path)
    assert runtime.get_data_root() == data.resolve()


@pytest.mark.parametrize("task", ["t10", "t1", "t001", "../t00", "/t00", "t00/../../..", "", 0, None])
def test_invalid_task_ids_are_rejected_before_environment_creation(task, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid IDs must not create a simulator")
    monkeypatch.setattr(runtime, "make_env", unexpected)
    with pytest.raises(ValueError, match="task ID"):
        with runtime.load_episode(task, 0):
            pass


@pytest.mark.parametrize("episode", [-1, 10, 1000, True, False, np.bool_(True), .5, "0", None])
def test_invalid_episode_ids_are_rejected_before_environment_creation(episode, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid IDs must not create a simulator")
    monkeypatch.setattr(runtime, "make_env", unexpected)
    with pytest.raises(ValueError, match="episode ID"):
        with runtime.load_episode("t00", episode):
            pass


def test_numpy_integer_episode_ids_are_accepted():
    assert runtime.episode_directory("t00", np.int64(0)).name == "ep000"


@pytest.mark.parametrize("filename", ["task.bddl", "obstacle_asset.json", "initial_state.npy"])
def test_tampered_input_is_rejected_before_environment_creation(filename, sample_root, monkeypatch):
    path = sample_root / "tasks" / "t00" / "episodes" / "ep000" / filename
    path.write_bytes(path.read_bytes() + b"changed")
    def unexpected(*args, **kwargs):
        pytest.fail("hash failure must be reported before creating a simulator")
    monkeypatch.setattr(runtime, "make_env", unexpected)
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        with runtime.load_episode("t00", 0, benchmark_root=sample_root):
            pass


def test_identity_and_validation_metadata_are_checked(sample_root):
    edit_manifest(sample_root, lambda metadata: metadata.update(episode_id=1))
    with pytest.raises(ValueError, match="identity"):
        runtime.read_episode("t00", 0, sample_root)
    edit_manifest(sample_root, lambda metadata: metadata.update(episode_id=0, checks_passed=False))
    with pytest.raises(ValueError, match="validation"):
        runtime.read_episode("t00", 0, sample_root)


def test_manifest_cannot_read_artifact_outside_episode(sample_root):
    edit_manifest(sample_root, lambda metadata: metadata["artifact_sha256"].update({"../../outside": "0" * 64}))
    with pytest.raises(ValueError, match="escapes"):
        runtime.read_episode("t00", 0, sample_root)


def test_missing_backend_has_actionable_error_and_invalid_override_does_not_fall_back(monkeypatch, tmp_path):
    monkeypatch.setenv("LIBERO_ROOT", str(tmp_path / "absent-libero"))
    with pytest.raises(FileNotFoundError, match="libero10-collision setup"):
        runtime.resolve_libero_root()


def test_backend_cache_fallback_is_not_tied_to_checkout_or_cwd(monkeypatch, tmp_path):
    cache = tmp_path / "user cache"
    root = cache / "libero10-collision" / "libero"
    base = root / "libero" / "libero"
    (base / "assets").mkdir(parents=True)
    (base / "bddl_files").mkdir()
    (base / "__init__.py").write_text("raise RuntimeError('resolver must not import LIBERO')")
    monkeypatch.delenv("LIBERO_ROOT", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)
    monkeypatch.chdir(tmp_path)
    assert runtime.resolve_libero_root() == root.resolve()


def test_installed_backend_is_resolved_without_running_interactive_import(monkeypatch, tmp_path):
    root = tmp_path / "site-packages"
    base = root / "libero" / "libero"
    (base / "assets").mkdir(parents=True)
    (base / "bddl_files").mkdir()
    (base / "__init__.py").write_text("raise RuntimeError('resolver must not import LIBERO')")
    monkeypatch.delenv("LIBERO_ROOT", raising=False)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: SimpleNamespace(
        submodule_search_locations=[str(root / "libero")], origin=None))
    assert runtime.resolve_libero_root() == root.resolve()


class FakeEnvironment:
    def __init__(self, state):
        self.closed = False
        self.actions = []
        self.initial_state = None
        self.sim = SimpleNamespace(get_state=lambda: state.copy())

    def set_init_state(self, state):
        self.initial_state = state.copy()
        return {"step": 0}

    def step(self, action):
        assert self.initial_state is not None
        self.actions.append(list(action))
        return {"step": len(self.actions)}, 0, False, {}

    def close(self):
        self.closed = True


def test_loader_returns_prescribed_start_and_closes_on_policy_exception(sample_root, monkeypatch):
    state = np.load(sample_root / "tasks/t00/episodes/ep000/initial_state.npy", allow_pickle=False)
    env = FakeEnvironment(state)
    monkeypatch.setattr(runtime, "make_env", lambda *args: env)
    with pytest.raises(RuntimeError, match="policy error"):
        with runtime.load_episode("t00", 0, benchmark_root=sample_root) as (loaded, observation, metadata):
            assert loaded is env
            assert np.array_equal(env.initial_state, state)
            assert observation == {"step": 10}
            assert env.actions == [list(runtime.HOLD)] * 10
            assert metadata["task"] == "t00"
            assert not env.closed
            raise RuntimeError("policy error")
    assert env.closed


def test_loader_closes_environment_on_model_mismatch(sample_root, monkeypatch):
    env = FakeEnvironment(np.zeros(1))
    monkeypatch.setattr(runtime, "make_env", lambda *args: env)
    with pytest.raises(ValueError, match="simulator model"):
        with runtime.load_episode("t00", 0, benchmark_root=sample_root):
            pass
    assert env.closed
