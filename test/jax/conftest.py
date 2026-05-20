import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def setup_before_all_tests():
    if os.getenv("XLA_FLAGS") is None:
        xla_flags = [
            "--xla_cpu_experimental_onednn_custom_call=true",
            "--xla_cpu_use_onednn=false",
            "--xla_cpu_experimental_ynn_fusion_type=invalid",
            "--xla_cpu_use_xnnpack=false",
            "--xla_backend_extra_options=xla_cpu_disable_new_fusion_emitter",
        ]
        os.environ["XLA_FLAGS"] = " ".join(xla_flags)
