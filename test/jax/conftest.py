import os


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
