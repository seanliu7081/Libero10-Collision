"""The public inspection commands work without importing simulation packages."""

import json

import pytest

from libero10_collision import cli, runtime


def test_list_exposes_official_numbering_and_all_100_episodes(capsys):
    assert cli.main(["list", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["episode_count"] == 100
    assert result["tasks"][0]["task"] == "t00"
    assert result["tasks"][1]["language"] == "put both the cream cheese box and the butter in the basket"


def test_validate_integrity_has_no_simulation_dependency(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("static integrity validation must not configure LIBERO")
    monkeypatch.setattr(runtime, "setup", unexpected)
    monkeypatch.setattr(cli, "load_episode", unexpected)
    assert cli.main(["validate"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is True
    assert result["episode_count"] == 100
    assert result["mode"] == "artifact_integrity"
    assert result["full_task_execution_tested"] is False


def test_validate_selection_limits_the_evaluated_cases(capsys):
    assert cli.main(["validate", "--tasks", "t00", "t09", "--episodes", "0", "9"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert {(item["task"], item["episode_id"]) for item in result["records"]} == {
        ("t00", 0), ("t00", 9), ("t09", 0), ("t09", 9),
    }


@pytest.mark.parametrize("arguments", [
    ["validate", "--tasks", "t10"],
    ["validate", "--tasks", "t01", "t01"],
    ["validate", "--episodes", "-1"],
    ["validate", "--episodes", "0", "0"],
    ["validate", "--strict-rgb"],
    ["smoke", "--steps", "0"],
    ["smoke", "--render-size", "0"],
])
def test_invalid_cli_options_fail_with_clear_message(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code in (1, 2)
    assert "error:" in capsys.readouterr().err


def test_settled_pose_check_accepts_quaternion_sign_equivalence():
    from types import SimpleNamespace
    from numpy import array
    expected = [0.1, -0.2, 0.5, 1.0, 0.0, 0.0, 0.0]
    actual = array(expected)
    actual[3:] *= -1
    env = SimpleNamespace(env=SimpleNamespace(objects_dict={"target": SimpleNamespace(joints=["joint"])}),
                          sim=SimpleNamespace(data=SimpleNamespace(get_joint_qpos=lambda joint: actual)))
    result = cli._check_settled_poses(env, {"settled_object_poses": {"target": expected}})
    assert result["object_count"] == 1
    assert result["max_translation_error_m"] == 0
    assert result["max_rotation_error_rad"] == 0


@pytest.mark.parametrize("actual,name", [
    ([0.101, -0.2, 0.5, 1., 0., 0., 0.], "target"),  # shifted 1 mm
    ([0.1, -0.2, 0.5, 0.9999875000260416, 0., 0., 0.004999979166692708], "target"),  # rotated .01 rad
    ([0.1, -0.2, 0.5, 1., 0., 0., 0.], "wrong_target"),
])
def test_settled_pose_check_rejects_displacement_rotation_and_wrong_names(actual, name):
    from types import SimpleNamespace
    expected = [0.1, -0.2, 0.5, 1., 0., 0., 0.]
    env = SimpleNamespace(env=SimpleNamespace(objects_dict={name: SimpleNamespace(joints=["joint"])}),
                          sim=SimpleNamespace(data=SimpleNamespace(get_joint_qpos=lambda joint: actual)))
    with pytest.raises(ValueError, match="[Ss]ettled"):
        cli._check_settled_poses(env, {"settled_object_poses": {"target": expected}})
