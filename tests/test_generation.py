"""Generation must not overwrite the release or silently reinterpret IDs."""

import pytest

from libero10_collision import generation, runtime


def test_generation_refuses_existing_output_and_leaves_it_untouched(tmp_path, monkeypatch):
    output = tmp_path / "existing benchmark"
    output.mkdir()
    sentinel = output / "important.npy"
    sentinel.write_bytes(b"original data")
    def unexpected(*args, **kwargs):
        pytest.fail("existing output must be rejected before importing the simulator")
    monkeypatch.setattr(runtime, "setup", unexpected)
    with pytest.raises(FileExistsError):
        generation.generate(output)
    assert sentinel.read_bytes() == b"original data"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize("tasks,episodes", [
    ([], None), (["t00", "t00"], None), (["t10"], None), (["../t00"], None),
    (None, []), (None, [-1]), (None, [10]), (None, [0, 0]), (None, [True]), (None, ["0"]),
])
def test_bad_generation_selection_is_rejected_without_creating_output(tasks, episodes, tmp_path, monkeypatch):
    output = tmp_path / "new benchmark"
    def unexpected(*args, **kwargs):
        pytest.fail("bad selection must be rejected before importing the simulator")
    monkeypatch.setattr(runtime, "setup", unexpected)
    with pytest.raises(ValueError):
        generation.generate(output, tasks=tasks, episodes=episodes)
    assert not output.exists()
