import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke_test: marks test as a smoke test")
    config.addinivalue_line(
        "markers", "smoke_test_if: marks test as a smoke test if combination of parameters matches arguments"
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
        marker_smoke_test_if = item.get_closest_marker("smoke_test_if")
        if marker_smoke_test_if is not None:
            test_parameters = item.nodeid[item.nodeid.rfind("[") + 1 : -1]

            for parameter_set_to_be_marked in marker_smoke_test_if.args:
                if parameter_set_to_be_marked in test_parameters:
                    item.add_marker(pytest.mark.smoke_test)
