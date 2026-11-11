"""Run from outside the source checkout after a non-editable wheel installation.

This deliberately fails if Python imported the source tree, so missing wheel
package data cannot be concealed by a successful checkout-based test run.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import site
import sys

import libero10_collision
from libero10_collision import runtime


def main():
    package = Path(libero10_collision.__file__).resolve().parent
    assert any(package.is_relative_to(Path(path).resolve()) for path in site.getsitepackages()), package
    root = runtime.get_data_root()
    assert root == package / "data", (package, root)
    for name in ("benchmark.json", "runtime_provenance.json", "GENERATION_PROTOCOL.md",
                 "LICENSE", "THIRD_PARTY_NOTICES.md", "generation/recipes.json"):
        assert (root / name).is_file(), name
    registry = json.loads((root / "benchmark.json").read_text())
    cases = 0
    for task in registry["tasks"]:
        for relative_path in task["episode_manifests"]:
            path = root / relative_path
            metadata = json.loads(path.read_text())
            for filename, expected in metadata["artifact_sha256"].items():
                assert hashlib.sha256((path.parent / filename).read_bytes()).hexdigest() == expected
            cases += 1
    assert cases == 100
    assert len(list((root / "tasks").glob("t*/episodes/ep*/*"))) == 400
    assert "libero.libero" not in sys.modules
    assert "mujoco" not in sys.modules
    print(json.dumps({"installed_package": str(package), "data_root": str(root),
                      "episodes_verified": cases, "simulator_imported": False}))


if __name__ == "__main__":
    main()
