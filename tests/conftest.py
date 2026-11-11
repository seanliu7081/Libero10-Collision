"""Simulation tests are opt-in; static tests need only NumPy and pytest."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--simulation", action="store_true", default=False,
        help="Run real LIBERO/MuJoCo contact tests using the configured renderer",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "simulation: requires the pinned LIBERO simulator")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--simulation"):
        skip = pytest.mark.skip(reason="use --simulation after installing the simulation environment")
        for item in items:
            if "simulation" in item.keywords:
                item.add_marker(skip)
