# Copyright (c) 2024-2026 Intel Corporation
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
"""The utility functions and classes for JAX."""
import inspect
import os
from typing import Callable, Dict, Optional

algos_mapping: Dict[str, Callable] = {}
import itertools
from functools import partial

import jax
import keras
import keras.src.utils.dtype_utils as dtype_utils
import keras_hub.src.utils.tensor_utils as tensor_utils
import ml_dtypes
from jax import numpy as jnp
from keras import ops, tree

# TODO: move to keras/src/utils/dtype_utils.py
# WA for missing float8 dtypes
from keras.src.utils.dtype_utils import DTYPE_TO_SIZE

from neural_compressor.common import logger

# TODO make it more general
WA_DTYPE_TO_SIZE = {
    **{f"float{i}": i for i in (16, 32, 64)},
    **{f"int{i}": i for i in (8, 16, 32, 64)},
    **{f"uint{i}": i for i in (8, 16, 32, 64)},
    "bfloat16": 16,
    "bool": 1,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
}


def wa_dtype_size(dtype):
    size = WA_DTYPE_TO_SIZE.get(dtype, None)
    if size is None:
        raise ValueError(f"Invalid dtype: {dtype}")
    return size


dtype_utils.dtype_size = wa_dtype_size

get_dtype_size_in_bits_orig = tensor_utils.get_dtype_size_in_bits


def wa_get_dtype_size_in_bits(dtype):
    q_dtypes = ["float8_e4m3fn", "float8_e4m3", "float8_e5m2"]
    if dtype in q_dtypes:
        return 8
    return get_dtype_size_in_bits_orig(dtype)


tensor_utils.get_dtype_size_in_bits = wa_get_dtype_size_in_bits


def register_algo(name):
    """Decorator function to register algorithms in the algos_mapping dictionary.

    Usage example:
        @register_algo(name=example_algo)
        def example_algo(model: tf.keras.Model, quant_config: StaticQuantConfig) -> tf.keras.Model:
            ...
    Args:
        name (str): The name under which the algorithm function will be registered.

    Returns:
        decorator: The decorator function to be used with algorithm functions.
    """

    def decorator(algo_func):
        algos_mapping[name] = algo_func
        return algo_func

    return decorator


def get_quantize_fun(dtype=ml_dtypes.float8_e4m3):
    @partial(jax.lax.composite, name="inc.quantize_fp8")
    def quantize_tensor_float(x, scale):
        return jax.lax.clamp(
            jnp.finfo(dtype).min.astype(x.dtype), x / scale, jnp.finfo(dtype).max.astype(x.dtype)
        ).astype(dtype)

    @partial(jax.lax.composite, name="inc.quantize_int8")
    def quantize_tensor_int(x, scale):
        return jnp.clip(jnp.round(x / scale), jnp.iinfo(dtype).min, jnp.iinfo(dtype).max).astype(dtype)

    if jnp.issubdtype(dtype, jnp.floating):
        return quantize_tensor_float
    elif jnp.issubdtype(dtype, jnp.integer):
        return quantize_tensor_int
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_dequantize_fun(dtype=jnp.float32):
    @partial(jax.lax.composite, name="inc.dequantize")
    def dequantize(x, scale):
        return x.astype(dtype) * scale

    return dequantize


def get_scale(orig_weight, dtype=ml_dtypes.float8_e4m3, compute_dtype=jnp.float32):
    # fp8 quantization
    @partial(jax.lax.composite, name="inc.get_scale_fp8")
    def float_get_scale(orig_weight):
        if 0 in orig_weight.shape:
            # For empty tensor, return scale as 1.0
            return jnp.array(1.0, dtype=compute_dtype)
        return (
            (jnp.max(jnp.abs(orig_weight), keepdims=True) / jnp.finfo(dtype).max.astype(orig_weight.dtype))
            .reshape((1,))
            .astype(compute_dtype)
        )

    @partial(jax.lax.composite, name="inc.get_scale_int")
    def integer_get_scale(orig_weight):
        return (jnp.max(jnp.abs(orig_weight), keepdims=True) / jnp.iinfo(dtype).max).reshape((1,)).astype(compute_dtype)

    if jnp.issubdtype(dtype, jnp.floating):
        return float_get_scale(orig_weight)
    elif jnp.issubdtype(dtype, jnp.integer):
        return integer_get_scale(orig_weight)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def print_model(container, max_lines=999999, internal=True, str_length=(0, 0), path=""):
    """Print the model structure.

    Args:
        container: The model or layer to be printed.
        max_lines: The maximum number of elements to print.
        internal: Whether to print layers from internal _layers (True) or public layers API (False).
        str_length: Tuple with max lengths for class name and path.
    """

    def get_str_length(container, max_long=(0, 0), path=""):
        current = (len(container.__class__.__name__), len(path))
        max_long = (max(current[0], max_long[0]), max(current[1], max_long[1]))
        if hasattr(container, "_layers"):
            for i, layer in enumerate(container._layers):
                max_long = get_str_length(layer, max_long, path + f".{layer.name}")
        return max_long

    if not hasattr(print_model, "count"):
        print_model.count = 0
    print_model.count += 1
    if str_length == (0, 0):
        print_model.count = 0
        str_length = get_str_length(container)
        logger.debug(f"{'-' * str_length[0]} {'internal' if internal else 'public'} representation:")
    else:
        path += f".{container.name}"
    if print_model.count >= max_lines:
        if print_model.count == max_lines:
            logger.debug("...")
        return

    # additional_info = f"built={'True' if container.built else 'False':<5}"
    additional_info = ""
    if hasattr(container, "min_val") and hasattr(container, "max_val"):
        additional_info += f" min,max={container.min_val.value:9.4g},{container.max_val.value:9.4g}"
    if hasattr(container, "ascale"):
        additional_info += f" a_scale={container.ascale._value}"
    if hasattr(container, "wscale"):
        additional_info += f" w_scale={container.wscale._value}"
    logger.debug(f"{container.__class__.__name__:{str_length[0]}} {path:{str_length[1]}}{additional_info}")

    if internal:
        if hasattr(container, "_layers"):
            for layer in container._layers:
                print_model(layer, max_lines, internal, str_length, path)
    else:
        if hasattr(container, "layers"):
            for layer in container.layers:
                print_model(layer, max_lines, internal, str_length, path)


dtype_mapping = {
    "int8": jnp.int8,
    "fp8": jnp.dtype(ml_dtypes.float8_e4m3),
    "fp8_e4m3": jnp.dtype(ml_dtypes.float8_e4m3fn),
    "fp8_e5m2": jnp.dtype(ml_dtypes.float8_e5m2),
}


def causal_lm_make_replace_generate_function(self, revert=False):
    """Replace generate function for the model to version suitable for calibration,
    where non-trainable are also stored.

    For revert=True, restore the original generate function.
    """

    @partial(jax.jit, static_argnames=["stop_token_ids"])
    def compiled_generate_function(inputs, stop_token_ids, state):
        (
            sampler_variables,
            trainable_variables,
            non_trainable_variables,
        ) = state
        mapping = itertools.chain(
            zip(self.sampler.variables, sampler_variables),
            zip(self.trainable_variables, trainable_variables),
            zip(self.non_trainable_variables, non_trainable_variables),
        )

        with keras.StatelessScope(state_mapping=mapping) as scope:
            outputs = self.generate_step(inputs, stop_token_ids)

        # Get updated sampler variables from the stateless scope.
        sampler_variables = []
        non_trainable_variables_new = []
        for v in self.non_trainable_variables:
            new_v = scope.get_current_value(v)
            non_trainable_variables_new.append(new_v if new_v is not None else v)
        for v in self.sampler.variables:
            new_v = scope.get_current_value(v)
            sampler_variables.append(new_v if new_v is not None else v)
        return outputs, non_trainable_variables_new, sampler_variables

    def wrapped_generate_function(
        inputs,
        stop_token_ids=None,
    ):
        if isinstance(stop_token_ids, list):
            stop_token_ids = tuple(stop_token_ids)

        # Create an explicit tuple of all variable state.
        state = (
            self.sampler.variables,
            # Use the explicit variable.value to preserve the
            # sharding spec of distribution.
            [v.value for v in self.trainable_variables],
            [v.value for v in self.non_trainable_variables],
        )
        inputs = tree.map_structure(ops.convert_to_tensor, inputs)
        outputs, non_trainable_variables, sampler_variables = compiled_generate_function(
            inputs,
            stop_token_ids,
            state,
        )
        # Only assign the sampler variables (random seeds), as other
        # model variables should never be updated in generation.
        for ref_v, v in zip(self.sampler.variables, sampler_variables):
            ref_v.assign(v)
        for ref_v, v in zip(self.non_trainable_variables, non_trainable_variables):
            ref_v.assign(v)
        return outputs

    if revert:
        self.generate_function = self.generate_function_orig
        del self.generate_function_orig
    else:
        if not hasattr(self, "generate_function_orig"):
            self.generate_function_orig = self.generate_function
        self.generate_function = wrapped_generate_function

    return self.generate_function


def iterate_over_layers(model, operations, /, *, filter_function: Optional[Callable] = lambda _: True):

    for layer in model._flatten_layers():

        if filter_function(layer.__class__):
            for operation in operations:
                operation(layer)

    return model


def verify_api(orig_cls, quant_cls, method_name):
    """Check if quantized layer method API matches original layer method API."""
    orig_method = getattr(orig_cls, method_name)
    quant_method = getattr(quant_cls, method_name)
    if inspect.signature(orig_method) != inspect.signature(quant_method):
        logger.error(
            f"Signature of {orig_cls.__name__}.{method_name} has changed, "
            f"please update {quant_cls.__name__} class to match it, "
            "or revert Keras to earlier version where the signature is not changed.\n"
        )
