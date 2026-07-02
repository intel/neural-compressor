import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke_test: marks test as a smoke test")
    config.addinivalue_line(
        "markers", "smoke_test_if: marks test as a smoke test if combination of parameters matches arguments"
    )


def pytest_sessionstart(session):
    os.environ["KERAS_BACKEND"] = "jax"
    print("KERAS_BACKEND =", os.environ.get("KERAS_BACKEND"))


def pytest_collection_modifyitems(items):
    for item in items:
        marker_smoke_test_if = item.get_closest_marker("smoke_test_if")
        if marker_smoke_test_if is not None:
            test_parameters = item.nodeid[item.nodeid.rfind("[") + 1 : -1]

            for parameter_set_to_be_marked in marker_smoke_test_if.args:
                if parameter_set_to_be_marked in test_parameters:
                    item.add_marker(pytest.mark.smoke_test)
