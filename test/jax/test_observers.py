#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for activation observers used during static quantization calibration.

Verifies that AbsMaxObserver tracks the maximum absolute value, and that eligible
layers select the correct observer: MinMaxObserver only for asymmetric integer
quantization, and AbsMaxObserver otherwise (fp8 and symmetric int8).
"""

import pytest
from jax import numpy as jnp
from keras.src import backend

from neural_compressor.jax.quantization.layers_static import (
    AbsMaxObserver,
    MinMaxObserver,
    StaticQDQLayer,
    get_activation_observer,
)

# Mark all tests in this file as smoke tests
pytestmark = pytest.mark.smoke_test


def test_abs_max_observer_tracks_max_abs():
    """AbsMaxObserver records the running maximum absolute value across calls."""
    observer = AbsMaxObserver(dtype="float32")

    assert not observer.is_calibrated()

    observer(jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32))
    observer(jnp.array([-5.0, 3.0], dtype=jnp.float32))
    observer(jnp.array([4.0, -1.0], dtype=jnp.float32))

    assert observer.is_calibrated()

    calibrated_range = observer.get_calibrated_range()
    # Only the maximum absolute value is returned
    assert float(calibrated_range[0]) == pytest.approx(5.0)


def test_abs_max_observer_passthrough():
    """AbsMaxObserver returns its inputs unchanged."""
    observer = AbsMaxObserver(dtype="float32")
    inputs = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
    outputs = observer(inputs)
    assert jnp.array_equal(inputs, outputs)


def test_abs_max_observer_respects_mask():
    """AbsMaxObserver ignores masked-out positions."""

    def _get_fields():
        observer = AbsMaxObserver(dtype="float32")
        inputs = jnp.array([[1.0, -9.0, 2.0]], dtype=jnp.float32)
        mask = jnp.array([[True, False, True]])
        return observer, inputs, mask

    observer, inputs, mask = _get_fields()
    observer(inputs, mask=mask)

    calibrated_range = observer.get_calibrated_range()
    assert float(calibrated_range[0]) == pytest.approx(2.0)

    observer, inputs, mask = _get_fields()
    backend.set_keras_mask(inputs, mask)
    observer(inputs)

    calibrated_range = observer.get_calibrated_range()
    assert float(calibrated_range[0]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "activation_dtype,asymmetric,expected",
    [
        (jnp.dtype("float8_e4m3fn"), False, AbsMaxObserver),
        (jnp.dtype("float8_e5m2"), False, AbsMaxObserver),
        (jnp.dtype("float8_e4m3fn"), True, AbsMaxObserver),  # fp8 is always symmetric
        (jnp.dtype("int8"), False, AbsMaxObserver),  # symmetric int8
        (jnp.dtype("int8"), True, MinMaxObserver),  # asymmetric int8
    ],
)
def test_get_activation_observer_selection(activation_dtype, asymmetric, expected):
    """The helper selects MinMaxObserver only for asymmetric integer quantization."""
    observer = get_activation_observer(activation_dtype, asymmetric, dtype_policy="float32")
    assert isinstance(observer, expected)


def test_static_qdq_layer_no_observer_with_fixed_range():
    """No observer is attached when a fixed range is provided."""
    layer = StaticQDQLayer(
        name="static_qdq_fixed",
        activation_dtype=jnp.dtype("float8_e4m3fn"),
        dtype="float32",
        fixed_range=(-3.0, 3.0),
    )
    layer.add_observers()
    assert not hasattr(layer, "input_observer")
