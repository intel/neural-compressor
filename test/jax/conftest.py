import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "CI_test: marks test to be run in CI")
    config.addinivalue_line(
        "markers", "CI_test_if: marks test to be run in CI if combination of parameters matches arguments"
    )


def pytest_sessionstart(session):
    os.environ["KERAS_BACKEND"] = "jax"
    if os.getenv("XLA_FLAGS") is None:
        xla_flags = [
            "--xla_cpu_experimental_onednn_custom_call=true",
            "--xla_cpu_use_onednn=false",
            "--xla_cpu_experimental_ynn_fusion_type=invalid",
            "--xla_cpu_use_xnnpack=false",
            "--xla_backend_extra_options=xla_cpu_disable_new_fusion_emitter",
        ]
        os.environ["XLA_FLAGS"] = " ".join(xla_flags)

    print("KERAS_BACKEND =", os.environ.get("KERAS_BACKEND"))
    print("XLA_FLAGS =", os.environ.get("XLA_FLAGS"))


def pytest_collection_modifyitems(items):
    for item in items:
        marker_CI_test_if = item.get_closest_marker("CI_test_if")
        if marker_CI_test_if is not None:
            all_test_parameters = item.nodeid[item.nodeid.rfind("[") + 1 : -1]
            parameter_sets_to_be_marked = (
                marker_CI_test_if.args if isinstance(marker_CI_test_if.args[0], list) else [marker_CI_test_if.args]
            )

            # If parameters in test case match all arguments passed to CI_test_if marker, marks them with CI_test
            for parameter_set in parameter_sets_to_be_marked:
                if all(param in all_test_parameters for param in parameter_set):
                    item.add_marker(pytest.mark.CI_test)
