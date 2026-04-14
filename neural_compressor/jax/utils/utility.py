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

from neural_compressor.common import logger


def check_backend(raise_error=True):
    """Check if the current Keras backend is JAX and log a warning or error if not."""

    if keras.config.backend() != "jax":
        message = (
            f"neural_compressor.jax only supports JAX backend, but the current Keras backend is {keras.config.backend()}. "
            'Consider setting KERAS_BACKEND env var to "jax".'
        )
        if raise_error:
            raise ValueError(message)
        else:
            logger.warning(message)


check_backend(raise_error=False)


def add_fp8_support(function):
    """Extend a dtype size function to support FP8 dtypes.

    Args:
        function (Callable): Function that returns the size of a dtype in bits.

    Returns:
        Callable: Wrapped function that handles FP8 dtypes.
    """

    def wrapper(dtype):
        """Return dtype size in bits with added FP8 support.

        Args:
            dtype (str): Dtype name to query.

        Returns:
            int: Size of the dtype in bits.
        """
        q_dtypes = ["float8_e4m3fn", "float8_e4m3", "float8_e5m2"]
        if dtype in q_dtypes:
            return 8
        return function(dtype)

    return wrapper


# Replace Keras and Keras-hub functions with the extended versions that support FP8.
tensor_utils.get_dtype_size_in_bits = add_fp8_support(tensor_utils.get_dtype_size_in_bits)
dtype_utils.dtype_size = add_fp8_support(dtype_utils.dtype_size)


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
        """Register an algorithm implementation in the global mapping.

        Args:
            algo_func (Callable): Algorithm implementation to register.

        Returns:
            Callable: The original algorithm function.
        """
        algos_mapping[name] = algo_func
        return algo_func

    return decorator


def get_quantize_fun(dtype=ml_dtypes.float8_e4m3, asymmetric=False):
    """Create a quantization function for the specified dtype.

    Args:
        dtype (jnp.dtype): Target quantization dtype.
        asymmetric (bool): Whether to use asymmetric quantization for integer dtypes.

    Returns:
        Callable: Quantization function that maps tensors to the target dtype.
    """

    @partial(jax.lax.composite, name="inc.quantize")
    def quantize_tensor_float(x, scale):
        """Quantize floating-point tensors using clamping.

        Args:
            x (jnp.ndarray): Input tensor.
            scale (jnp.ndarray): Scale factor for quantization.

        Returns:
            jnp.ndarray: Quantized tensor.
        """
        return jax.lax.clamp(
            jnp.finfo(dtype).min.astype(x.dtype), x / scale, jnp.finfo(dtype).max.astype(x.dtype)
        ).astype(dtype)

    @partial(jax.lax.composite, name="inc.quantize")
    def quantize_tensor_int(x, scale):
        """Quantize integer tensors using symmetric scaling.

        Args:
            x (jnp.ndarray): Input tensor.
            scale (jnp.ndarray): Scale factor for quantization.

        Returns:
            jnp.ndarray: Quantized tensor.
        """
        val = jnp.round(x / scale)
        val = jnp.clip(val, jnp.iinfo(dtype).min, jnp.iinfo(dtype).max)
        return val.astype(dtype)

    @partial(jax.lax.composite, name="inc.quantize")
    def quantize_tensor_int_asymmetric(x, scale, zero_point):
        """Quantize integer tensors using asymmetric scaling.

        Args:
            x (jnp.ndarray): Input tensor.
            scale (jnp.ndarray): Scale factor for quantization.
            zero_point (jnp.ndarray): Zero point offset.

        Returns:
            jnp.ndarray: Quantized tensor.
        """
        val = jnp.round(x / scale) + zero_point
        val = jnp.clip(val, jnp.iinfo(dtype).min, jnp.iinfo(dtype).max)
        return val.astype(dtype)

    if jnp.issubdtype(dtype, jnp.floating):
        return quantize_tensor_float
    elif jnp.issubdtype(dtype, jnp.integer):
        return quantize_tensor_int_asymmetric if asymmetric else quantize_tensor_int
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_dequantize_fun(dtype=jnp.float32, asymmetric=False):
    """Create a dequantization function for the specified dtype.

    Args:
        dtype (jnp.dtype): Output dtype after dequantization.
        asymmetric (bool): Whether to use asymmetric dequantization.

    Returns:
        Callable: Function that dequantizes tensors.
    """

    @partial(jax.lax.composite, name="inc.dequantize")
    def dequantize(x, scale):
        """Dequantize a tensor by applying the scale.

        Args:
            x (jnp.ndarray): Quantized tensor.
            scale (jnp.ndarray): Scale factor used for quantization.

        Returns:
            jnp.ndarray: Dequantized tensor.
        """
        return x.astype(dtype) * scale

    @partial(jax.lax.composite, name="inc.dequantize")
    def dequantize_asymmetric(x, scale, zero_point=jnp.array(0, dtype=dtype)):
        """Dequantize a tensor with asymmetric scaling.

        Args:
            x (jnp.ndarray): Quantized tensor.
            scale (jnp.ndarray): Scale factor used for quantization.
            zero_point (jnp.ndarray): Zero point offset.

        Returns:
            jnp.ndarray: Dequantized tensor.
        """
        negated_zero_point = -zero_point
        return (x.astype(dtype) + negated_zero_point) * scale

    return dequantize_asymmetric if asymmetric else dequantize


def get_scale(orig_weight, dtype=ml_dtypes.float8_e4m3, compute_dtype=jnp.float32):
    """Compute the quantization scale for a weight tensor.

    Args:
        orig_weight (jnp.ndarray): Weight tensor to analyze.
        dtype (jnp.dtype): Target quantized dtype.
        compute_dtype (jnp.dtype): dtype for scale computation.

    Returns:
        jnp.ndarray: Computed scale tensor.
    """

    # fp8 quantization
    @partial(jax.lax.composite, name="inc.get_scale_fp8")
    def float_get_scale(orig_weight):
        """Compute scale for floating-point quantization.

        Args:
            orig_weight (jnp.ndarray): Weight tensor to analyze.

        Returns:
            jnp.ndarray: Computed scale tensor.
        """
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
        """Compute scale for integer quantization.

        Args:
            orig_weight (jnp.ndarray): Weight tensor to analyze.

        Returns:
            jnp.ndarray: Computed scale tensor.
        """
        if 0 in orig_weight.shape:
            # For empty tensor, return scale as 1.0
            return jnp.array(1.0, dtype=compute_dtype)
        return (
            (jnp.max(jnp.abs(orig_weight), keepdims=True) / jnp.array(jnp.iinfo(dtype).max).astype(orig_weight.dtype))
            .reshape((1,))
            .astype(compute_dtype)
        )

    if jnp.issubdtype(dtype, jnp.floating):
        return float_get_scale(orig_weight)
    elif jnp.issubdtype(dtype, jnp.integer):
        return integer_get_scale(orig_weight)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_q_params(orig_weight, dtype=ml_dtypes.float8_e4m3, compute_dtype=jnp.float32, asymmetric=False):
    """Compute quantization scale and zero-point for a weight tensor.

    Args:
        orig_weight (jnp.ndarray): Weight tensor to analyze.
        dtype (jnp.dtype): Target quantized dtype.
        compute_dtype (jnp.dtype): dtype for scale computation.
        asymmetric (bool): Whether to compute asymmetric quantization parameters.

    Returns:
        Tuple[jnp.ndarray, Optional[jnp.ndarray]]: Scale and zero-point. Zero-point is `None` for floating-point
        dtypes or symmetric quantization.
    """

    @partial(jax.lax.composite, name="inc.get_q_params_int")
    def integer_get_q_params(orig_weight):
        """Compute scale and zero-point for integer quantization.

        Args:
            orig_weight (jnp.ndarray): Weight tensor to analyze.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Scale and zero-point tensors.
        """
        if 0 in orig_weight.shape:
            # For empty tensor, return scale as 1.0
            return jnp.array(1.0, dtype=compute_dtype), jnp.array(0.0, dtype=jnp.int32)
        orig_min = jnp.min(orig_weight).astype(compute_dtype)
        orig_max = jnp.max(orig_weight).astype(compute_dtype)
        int_min = jnp.array(jnp.iinfo(dtype).min).astype(compute_dtype)
        int_max = jnp.array(jnp.iinfo(dtype).max).astype(compute_dtype)
        scale = (orig_max - orig_min) / (int_max - int_min)
        zero_point = jnp.round(int_min - orig_min / scale)
        return scale.reshape((1,)).astype(compute_dtype), zero_point.reshape((1,)).astype(jnp.int32)

    if jnp.issubdtype(dtype, jnp.floating):
        return get_scale(orig_weight, dtype, compute_dtype), None
    elif jnp.issubdtype(dtype, jnp.integer):
        if asymmetric:
            return integer_get_q_params(orig_weight)
        else:
            return get_scale(orig_weight, dtype, compute_dtype), None
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def print_model(container, max_lines=999999, internal=True, str_length=(0, 0), path=""):
    """Print the model structure.

    Args:
        container (keras.Model): The model or layer to be printed.
        max_lines (int): The maximum number of elements to print.
        internal (bool): Whether to print layers from internal _layers (True) or public layers API (False).
        str_length (Tuple[int, int]): Tuple with max lengths for class name and path.
        path (str): Prefix path for the current layer.

    Returns:
        None: Logs model structure via the logger.
    """

    def get_str_length(container, max_long=(0, 0), path=""):
        """Compute maximum string lengths for aligned model printing.

        Args:
            container (keras.Layer): Layer or model to inspect.
            max_long (Tuple[int, int]): Current maximum lengths.
            path (str): Path prefix for this layer.

        Returns:
            Tuple[int, int]: Updated maximum lengths for class name and path.
        """
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
    if hasattr(container, "a_scale"):
        if isinstance(container.a_scale, jax.Array):
            additional_info += f" a_scale(attr)={container.a_scale}"
        else:
            additional_info += f" a_scale={container.a_scale.value}"
    if hasattr(container, "a_zero_point"):
        if isinstance(container.a_zero_point, jax.Array):
            additional_info += f" a_zero_point(attr)={container.a_zero_point}"
        else:
            additional_info += f" a_zero_point={container.a_zero_point.value}"
    if hasattr(container, "w_scale"):
        if isinstance(container.w_scale, jax.Array):
            additional_info += f" w_scale(attr)={container.w_scale}"
        else:
            additional_info += f" w_scale={container.w_scale.value}"
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
    """Replace generate function for calibration and restore on demand.

    Args:
        self (keras.Model): Causal language model instance to modify.
        revert (bool): When True, restore the original generate function.

    Returns:
        Callable: Updated generate function.
    """

    @partial(jax.jit, static_argnames=["stop_token_ids"])
    def compiled_generate_function(inputs, stop_token_ids, state):
        """JIT-compiled generate function for calibration-friendly state handling.

        Args:
            inputs (jnp.ndarray): Input tokens for generation.
            stop_token_ids (Tuple[int, ...]): Token IDs used to stop generation.
            state (Tuple[Any, Any, Any]): Tuple of sampler, trainable, and non-trainable variables.

        Returns:
            Tuple[Any, List[Any], List[Any]]: Outputs, updated non-trainable variables, and sampler variables.
        """
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
        """Wrapper around generate_step to preserve variable state.

        Args:
            inputs (jnp.ndarray): Input tokens for generation.
            stop_token_ids (Optional[Tuple[int, ...]]): Token IDs used to stop generation.

        Returns:
            Any: Model outputs from generate_step.
        """
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
    """Apply operations to model layers matching the filter function.

    Args:
        model (keras.Model): Keras model with a _flatten_layers iterator.
        operations (Iterable[Callable]): Operations to apply to each layer.
        filter_function (Callable, optional): Predicate to select layers. Defaults to always True.

    Returns:
        keras.Model: The original model after operations have been applied.
    """
    for layer in model._flatten_layers():

        if filter_function(layer.__class__):
            for operation in operations:
                operation(layer)

    return model


def verify_api(orig_cls, quant_cls, method_name):
    """Check if quantized layer method API matches original layer method API.

    Args:
        orig_cls (type): Original layer class.
        quant_cls (type): Quantized layer class.
        method_name (str): Method name to compare.

    Returns:
        None: Logs an error if the method signatures differ.
    """
    orig_method = getattr(orig_cls, method_name)
    quant_method = getattr(quant_cls, method_name)
    if inspect.signature(orig_method) != inspect.signature(quant_method):
        logger.error(
            f"Signature of {orig_cls.__name__}.{method_name} has changed, "
            f"please update {quant_cls.__name__} class to match it, "
            "or revert Keras to earlier version where the signature is not changed.\n"
        )
