import os

# Must be set before ``keras`` is imported below so the backend is selected correctly.
os.environ["KERAS_BACKEND"] = "jax"

import keras
import pytest
from jax import numpy as jnp


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


@keras.saving.register_keras_serializable()
class TestModel(keras.Model):
    """Shared model for JAX quantization tests.

    Three named Dense layers (``first``, ``second``, ``third``) plus an
    unsupported ``LayerNormalization`` layer, with deterministic seeded weights
    so repeated builds produce identical models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first = keras.layers.Dense(
            6,
            activation="relu",
            use_bias=False,
            name="first",
            kernel_initializer=keras.initializers.random_normal(seed=2000),
        )
        self.second = keras.layers.Dense(
            4,
            activation="linear",
            use_bias=False,
            name="second",
            kernel_initializer=keras.initializers.random_normal(seed=3000),
        )
        self.third = keras.layers.Dense(
            2,
            activation="linear",
            use_bias=False,
            name="third",
            kernel_initializer=keras.initializers.random_normal(seed=4000),
        )
        self.norm = keras.layers.LayerNormalization(name="norm")

    def call(self, inputs):
        x = self.first(inputs)
        x = self.second(x)
        x = self.third(x)
        return self.norm(x)


@pytest.fixture
def model():
    """Return a fresh, path-populated shared test model.

    ``layer.path`` is only populated after a forward pass, so the model is
    called once on dummy input before it is returned. The fixture is
    function-scoped, so each test receives its own instance and may quantize it
    in-place without affecting other tests.
    """
    m = TestModel()
    _ = m(jnp.ones((1, 8)))
    return m


@pytest.fixture
def calibration_data():
    """Calibration data for static quantization."""
    return jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )


@pytest.fixture
def test_data():
    """Inference data for output comparisons."""
    return jnp.array(
        [
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            [7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5],
        ]
    )
