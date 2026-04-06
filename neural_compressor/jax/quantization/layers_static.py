"""Static quantized layer implementations for JAX-backed Keras models."""

# Copyright (c) 2025-2026 Intel Corporation
#
# Portions of this code are derived from:
# - Keras (https://github.com/keras-team/keras)
# - Keras-hub, Copyright 2024, KerasHub authors (https://github.com/keras-team/keras-hub)
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


import keras
import numpy as np
from jax import numpy as jnp
from keras import ops
from keras.layers import Dense, EinsumDense, MultiHeadAttention
from keras_hub.layers import ReversibleEmbedding, RotaryEmbedding
from keras_hub.src.models.gemma3.gemma3_attention import CachedGemma3Attention
from keras_hub.src.models.gemma3.gemma3_vision_encoder import Gemma3VisionAttention

from neural_compressor.common import logger
from neural_compressor.jax.quantization.saving import SaveableLayerMixin
from neural_compressor.jax.utils.utility import (
    get_dequantize_fun,
    get_q_params,
    get_quantize_fun,
    verify_api,
)

static_quant_mapping = {}


def register_static_quantized_layer(clso):
    """Register quantized layer class for an original layer class.

    Args:
        clso (type): Original layer class to map to a quantized implementation.

    Returns:
        Callable: Decorator that registers the quantized class.
    """

    def decorator(cls):
        """Attach the quantized class to the static mapping.

        Args:
            cls (type): Quantized layer class to register.

        Returns:
            type: The same class, for decorator chaining.
        """
        static_quant_mapping[clso] = cls
        return cls

    return decorator


class MinMaxObserver(keras.layers.Layer):
    """Observer that tracks running min/max values for calibration."""

    def __init__(self, *args, **kwargs):
        """Initialize the min/max observer layer.

        Args:
            *args: Positional arguments for the base layer.
            **kwargs: Keyword arguments for the base layer.

        Returns:
            None: Initializes the observer layer.
        """
        super().__init__(*args, **kwargs, name="min_max")
        # Track running min/max as non-trainable weights
        self.min_val = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(np.inf),
            trainable=False,
            name="min_val",
            dtype=self.compute_dtype,
        )
        self.max_val = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(-np.inf),
            trainable=False,
            name="max_val",
            dtype=self.compute_dtype,
        )
        self.supports_masking = True

    def call(self, inputs, mask=None):
        """Update min/max statistics during calibration.

        Args:
            inputs (jnp.ndarray): Input tensor to observe.
            mask (Optional[jnp.ndarray]): Optional mask to ignore padded elements.

        Returns:
            jnp.ndarray: The original inputs for passthrough.
        """
        if 0 not in inputs.shape:
            if mask is not None:
                # Expand mask to match input dimensions if needed
                if len(mask.shape) < len(inputs.shape):
                    for _ in range(len(inputs.shape) - len(mask.shape)):
                        mask = ops.expand_dims(mask, axis=-1)
                # Apply mask to exclude masked positions
                masked_inputs_min = ops.where(mask, inputs, jnp.array(float("inf"), dtype=inputs.dtype))
                masked_inputs_max = ops.where(mask, inputs, jnp.array(float("-inf"), dtype=inputs.dtype))
                batch_min = keras.ops.min(masked_inputs_min)
                batch_max = keras.ops.max(masked_inputs_max)
            else:
                batch_min = keras.ops.min(inputs)
                batch_max = keras.ops.max(inputs)

            self.min_val.assign(keras.ops.minimum(self.min_val, batch_min))
            self.max_val.assign(keras.ops.maximum(self.max_val, batch_max))
        return inputs

    def build(self, input_shape):
        """Override build with no additional variables.

        Args:
            input_shape (Tuple[int, ...]): Input shape for the layer.

        Returns:
            None: No additional variables are created.
        """
        pass

    def get_calibrated_range(self):
        """Return the calibrated min/max range as a tensor.

        Returns:
            jnp.ndarray: Tensor containing min and max values.
        """
        return ops.array((self.min_val, self.max_val))


class StaticQDQLayer(SaveableLayerMixin, keras.layers.Layer):
    """Layer that applies static quantize-dequantize to activations."""

    def __init__(self, name, activation_dtype, dtype="float32", asymmetric=False, const_scale=False):
        """Initialize the static QDQ helper layer.

        Args:
            name (str): Layer name.
            activation_dtype (jnp.dtype): Activation dtype used for quantization.
            dtype (str | keras.DTypePolicy): dtype for the layer - see keras.layers.Layer API for details.
            asymmetric (bool): Whether to use asymmetric quantization.
            const_scale (bool): Whether to use constant scales.

        Returns:
            None: Initializes the layer instance.
        """
        super().__init__(name=name, dtype=dtype)
        self.activation_dtype = activation_dtype
        self._is_asymmetric = asymmetric
        self.supports_masking = True
        self._is_quantized = False
        self.const_scale = const_scale
        if const_scale:
            self._const_variables = ["a_scale"]
            if asymmetric:
                self._const_variables.append("a_zero_point")
        else:
            self._const_variables = []

    def add_observers(self):
        """Attach observer layers for calibration.

        Returns:
            None: Adds observer layers.
        """
        self._tracker.unlock()
        self.input_observer = MinMaxObserver(dtype=self.dtype_policy)
        self._tracker.lock()

    def add_variables(self):
        """Create quantization variables for activations.

        Returns:
            None: Initializes quantization variables.
        """
        self._tracker.unlock()
        if self._is_asymmetric:
            self.a_zero_point = self.add_weight(
                name="a_zero_point",
                shape=(1,),
                initializer="zeros",
                trainable=False,
                autocast=False,
                dtype=jnp.int32,
            )
        self.a_scale = self.add_weight(
            name="a_scale",
            shape=(1,),
            initializer="zeros",
            trainable=False,
            autocast=False,
            dtype=self.compute_dtype,
        )
        self.aquantfun = get_quantize_fun(dtype=self.activation_dtype, asymmetric=self._is_asymmetric)
        self.adequantfun = get_dequantize_fun(dtype=self.compute_dtype, asymmetric=self._is_asymmetric)
        self._tracker.lock()

    def convert(self):
        """Compute activation scale and finalize static quantization.

        Returns:
            None: Updates activation scale variables.
        """
        self._tracker.unlock()
        arange = self.input_observer.get_calibrated_range()
        a_scale, a_zero_point = get_q_params(
            arange, self.activation_dtype, self.compute_dtype, asymmetric=self._is_asymmetric
        )
        if jnp.isinf(a_scale).any().item():
            logger.warning(
                f"Activation scale is inf for layer {self._path}. This may be caused by missing calibration data. "
                "Please make sure to run calibration with representative dataset."
            )
            self._is_quantized = False
            self._tracker.lock()
            return
        self.a_scale.assign(a_scale)
        if self._is_asymmetric:
            self.a_zero_point.assign(a_zero_point)
        self._tracker.lock()

    def post_quantization_cleanup(self):
        """Remove observers and finalize quantized call path.

        Returns:
            None: Cleans up observers and sets quantized call.
        """
        self._tracker.unlock()
        if hasattr(self, "_layers") and hasattr(self, "input_observer"):
            if self.input_observer in self._layers:
                self._layers.remove(self.input_observer)
                del self.input_observer

        # convert some variables to const if needed
        for name in self._const_variables:
            var = getattr(self, name)
            value = jnp.array(var.value)
            self._non_trainable_variables[:] = [v for v in self._non_trainable_variables if v is not var]
            setattr(self, name, value)

        self.call = self.call_asymmetric if self._is_asymmetric else self.call_symmetric
        self._is_quantized = True
        self._tracker.lock()

    def call(self, inputs, mask=None):
        """Run calibration observer on inputs.

        Args:
            inputs (jnp.ndarray): Input tensor.
            mask (Optional[jnp.ndarray]): Optional mask tensor.

        Returns:
            jnp.ndarray: Observed inputs.
        """
        x = self.input_observer(inputs, mask=mask)
        return x

    def call_symmetric(self, inputs, mask=None):
        """Apply symmetric quantize-dequantize to inputs.

        Args:
            inputs (jnp.ndarray): Input tensor.
            mask (Optional[jnp.ndarray]): Optional mask tensor.

        Returns:
            jnp.ndarray: Quantized-dequantized tensor.
        """

        if self.const_scale:
            a_scale = self.a_scale
        else:
            a_scale = self.a_scale.value

        x = self.aquantfun(inputs, a_scale)
        x = self.adequantfun(x, a_scale)
        return x

    def call_asymmetric(self, inputs, mask=None):
        """Apply asymmetric quantize-dequantize to inputs.

        Args:
            inputs (jnp.ndarray): Input tensor.
            mask (Optional[jnp.ndarray]): Optional mask tensor.

        Returns:
            jnp.ndarray: Quantized-dequantized tensor.
        """
        if self.const_scale:
            a_scale = self.a_scale
            a_zero_point = self.a_zero_point
        else:
            a_scale = self.a_scale.value
            a_zero_point = self.a_zero_point.value
        x = self.aquantfun(inputs, a_scale, a_zero_point)
        x = self.adequantfun(x, a_scale, a_zero_point)
        return x


class QStaticDenseMixin(SaveableLayerMixin):
    """Mixin that adds static quantization to dense-like layers."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a dense-like layer instance for static quantization.

        Args:
            orig (keras.layers.Layer): Original layer instance.
            weight_dtype (jnp.dtype): Dtype for quantized weights.
            activation_dtype (jnp.dtype): Dtype for quantized activations.
            const_scale (bool): Whether to use constant scales.
            const_weight (bool): Whether to use constant weight.

        Returns:
            keras.layers.Layer: The updated layer instance.
        """
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.weight_dtype = weight_dtype
        orig.activation_dtype = activation_dtype
        orig.const_scale = const_scale
        orig.const_weight = const_weight
        orig._is_quantized = False
        orig._is_int8 = jnp.issubdtype(activation_dtype, jnp.integer)
        orig.kernel_shape = orig.kernel.shape
        if const_scale:
            orig._const_variables = ["a_scale", "w_scale"]
            if orig._is_int8:
                orig._const_variables.append("a_zero_point")
        else:
            orig._const_variables = []
        if const_weight:
            orig._const_variables.append("_kernel_quant")
        orig._tracker.lock()
        return orig

    def add_observers(self):
        """Attach observer layers for calibration.

        Returns:
            None: Adds observer layers.
        """
        self._tracker.unlock()
        self.input_observer = MinMaxObserver(dtype=self.dtype_policy)
        self._tracker.lock()

    def add_variables(self):
        """Create quantization variables for activations and weights.

        Returns:
            None: Initializes quantization variables.
        """
        self._tracker.unlock()
        if self._is_int8:
            self.a_zero_point = self.add_weight(
                name="a_zero_point",
                shape=(1,),
                initializer="zeros",
                trainable=False,
                autocast=False,
                dtype=jnp.int32,
            )
        self.a_scale = self.add_weight(
            name="a_scale",
            shape=(1,),
            initializer="zeros",
            trainable=False,
            autocast=False,
            dtype=self.compute_dtype,
        )
        self.w_scale = self.add_weight(
            name="w_scale",
            shape=(1,),
            initializer="zeros",
            trainable=False,
            autocast=False,
            dtype=self.compute_dtype,
        )
        self._kernel_quant = self.add_weight(
            name="_kernel_quant",
            shape=self.kernel_shape,
            initializer="zeros",
            trainable=False,
            dtype=self.weight_dtype,
            autocast=False,
        )

        self.aquantfun = get_quantize_fun(dtype=self.activation_dtype, asymmetric=self._is_int8)
        self.adequantfun = get_dequantize_fun(dtype=self.compute_dtype, asymmetric=self._is_int8)
        self.wquantfun = get_quantize_fun(dtype=self.weight_dtype, asymmetric=False)
        self.wdequantfun = get_dequantize_fun(dtype=self.compute_dtype, asymmetric=False)

        self._tracker.lock()

    def convert(self):
        """Compute activation/weight scales and quantize weights.

        Returns:
            None: Updates quantization variables with calibrated values.
        """
        self._tracker.unlock()

        arange = self.input_observer.get_calibrated_range()
        a_scale, a_zero_point = get_q_params(
            arange, self.activation_dtype, self.compute_dtype, asymmetric=self._is_int8
        )
        if jnp.isinf(a_scale).any().item():
            logger.warning(
                f"Activation scale is inf for layer {self._path}. This may be caused by missing calibration data. "
                "Please make sure to run calibration with representative dataset."
            )
            self.call = super().call
            self._is_quantized = False
            self._tracker.lock()
            return
        self.a_scale.assign(a_scale)
        if self._is_int8:
            self.a_zero_point.assign(a_zero_point)

        w_scale, _ = get_q_params(self.kernel, self.weight_dtype, self.compute_dtype, asymmetric=False)
        self.w_scale.assign(w_scale)
        if self._is_int8:
            _kernel_quant = self.wquantfun(self.kernel, self.w_scale.value)
        else:
            _kernel_quant = self.wquantfun(self.kernel, self.w_scale.value)
        self._kernel_quant.assign(_kernel_quant)
        self._tracker.lock()

    def post_quantization_cleanup(self):
        """Finalize static quantization and drop unused weights.

        Returns:
            None: Cleans up observers and original weights.
        """
        self._tracker.unlock()
        if hasattr(self, "_kernel") and self._kernel in self._trainable_variables:
            self._trainable_variables.remove(self._kernel)
            del self._kernel

        # convert variables to attributes (const) if needed
        for name in self._const_variables:
            var = getattr(self, name)
            value = jnp.array(var.value)
            self._non_trainable_variables[:] = [v for v in self._non_trainable_variables if v is not var]
            delattr(self, name)
            setattr(self, name, value)

        if hasattr(self, "_layers") and hasattr(self, "input_observer"):
            if self.input_observer in self._layers:
                self._layers.remove(self.input_observer)
                del self.input_observer

        self.call = self.call_int8 if self._is_int8 else self.call_fp8
        self._is_quantized = True
        self._tracker.lock()

    @property
    def kernel(self):
        """Return the dequantized kernel tensor.

        Returns:
            jnp.ndarray: Dequantized kernel tensor.
        """
        if self._is_quantized:
            if self.const_weight:
                _kernel_quant = self._kernel_quant
            else:
                _kernel_quant = self._kernel_quant.value
            if self.const_scale:
                w_scale = self.w_scale
            else:
                w_scale = self.w_scale.value
            _kernel_quant = self.wdequantfun(_kernel_quant, w_scale)
            return _kernel_quant
        ret = super().kernel
        return ret.value

    def call(self, inputs, training=None):
        """Run calibration observer before the dense computation.

        Args:
            inputs (jnp.ndarray): Input tensor.
            training (Optional[bool]): Training mode flag.

        Returns:
            jnp.ndarray: Layer output tensor.
        """
        x = self.input_observer(inputs)
        x = super().call(x, training=training)
        return x

    def call_fp8(self, inputs, training=None):
        """Apply FP8 quantize-dequantize before dense computation.

        Args:
            inputs (jnp.ndarray): Input tensor.
            training (Optional[bool]): Training mode flag.

        Returns:
            jnp.ndarray: Layer output tensor.
        """
        if self.const_scale:
            a_scale = self.a_scale
        else:
            a_scale = self.a_scale.value
        x = self.aquantfun(inputs, a_scale)
        x = self.adequantfun(x, a_scale)
        x = super().call(x, training=training)
        return x

    def call_int8(self, inputs, training=None):
        """Apply int8 quantize-dequantize before dense computation.

        Args:
            inputs (jnp.ndarray): Input tensor.
            training (Optional[bool]): Training mode flag.

        Returns:
            jnp.ndarray: Layer output tensor.
        """
        if self.const_scale:
            a_scale = self.a_scale
            a_zero_point = self.a_zero_point
        else:
            a_scale = self.a_scale.value
            a_zero_point = self.a_zero_point.value
        x = self.aquantfun(inputs, a_scale, a_zero_point)
        x = self.adequantfun(x, a_scale, a_zero_point)
        x = super().call(x, training=training)
        return x


@register_static_quantized_layer(Dense)
class QStaticDense(QStaticDenseMixin, Dense):
    """Statically quantized Dense layer."""

    pass


verify_api(Dense, QStaticDense, "call")


@register_static_quantized_layer(EinsumDense)
class QStaticEinsumDense(QStaticDenseMixin, EinsumDense):
    """Statically quantized EinsumDense layer."""

    pass


verify_api(EinsumDense, QStaticEinsumDense, "call")


@register_static_quantized_layer(MultiHeadAttention)
class QStaticMultiHeadAttention(SaveableLayerMixin, MultiHeadAttention):
    """Statically quantized MultiHeadAttention layer."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a MultiHeadAttention instance for static quantization.

        Args:
            orig (keras.layers.MultiHeadAttention): Original layer instance.
            weight_dtype (jnp.dtype): Dtype for quantized weights.
            activation_dtype (jnp.dtype): Dtype for quantized activations.
            const_scale (bool): Whether to use constant scales.
            const_weight (bool): ignored, included for API consistency.

        Returns:
            keras.layers.MultiHeadAttention: Updated layer instance.
        """
        orig._tracker.unlock()
        orig.__class__ = cls
        orig._is_int8 = jnp.issubdtype(activation_dtype, jnp.integer)
        orig.q_qdq = StaticQDQLayer("q_qdq", activation_dtype, orig.dtype_policy, orig._is_int8, const_scale)
        orig.k_qdq = StaticQDQLayer("k_qdq", activation_dtype, orig.dtype_policy, orig._is_int8, const_scale)
        orig.a_qdq = StaticQDQLayer("a_qdq", activation_dtype, orig.dtype_policy, orig._is_int8, const_scale)
        orig.v_qdq = StaticQDQLayer("v_qdq", activation_dtype, orig.dtype_policy, orig._is_int8, const_scale)
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def add_observers(self):
        """Attach observer layers for calibration.

        Returns:
            None: Adds observer layers.
        """
        self.q_qdq.add_observers()
        self.k_qdq.add_observers()
        self.a_qdq.add_observers()
        self.v_qdq.add_observers()

    def add_variables(self):
        """Create quantization variables for activation QDQ.

        Returns:
            None: Initializes QDQ helper variables.
        """
        self.q_qdq.add_variables()
        self.k_qdq.add_variables()
        self.a_qdq.add_variables()
        self.v_qdq.add_variables()

    def convert(self):
        """Compute activation calibration values for QDQ helpers.

        Returns:
            None: Updates QDQ helpers with calibrated values.
        """
        self.q_qdq.convert()
        self.k_qdq.convert()
        self.a_qdq.convert()
        self.v_qdq.convert()

    def post_quantization_cleanup(self):
        """Finalize static quantization and mark the layer as quantized.

        Returns:
            None: Cleans up observers and marks quantized state.
        """
        self._tracker.unlock()
        self.q_qdq.post_quantization_cleanup()
        self.k_qdq.post_quantization_cleanup()
        self.a_qdq.post_quantization_cleanup()
        self.v_qdq.post_quantization_cleanup()
        self._is_quantized = True
        self._tracker.lock()

    # fmt: off
    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        training=None,
        return_attention_scores=False,
    ):
        """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Args:
            query: Projected query tensor of shape `(B, T, N, key_dim)`.
            key: Projected key tensor of shape `(B, S, N, key_dim)`.
            value: Projected value tensor of shape `(B, S, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions. It is generally not needed if
                the `query` and `value` (and/or `key`) are masked.
            training: Python boolean indicating whether the layer should behave
                in training mode (adding dropout) or in inference mode (doing
                nothing).

        Returns:
          Tuple[jnp.ndarray, Optional[jnp.ndarray]]: Attention outputs and attention scores.
        """
        # Check for flash attention constraints
        if self._flash_attention and return_attention_scores:
            raise ValueError(
                "Returning attention scores is not supported when flash "
                "attention is enabled. Please disable flash attention to access"
                " attention scores."
            )

        # Determine whether to use dot-product attention
        # use_dot_product_attention = not (
        #     self._dropout > 0.0
        #     or return_attention_scores
        #     or (len(query.shape) != 4)
        # )
        use_dot_product_attention = False  # TODO Add dot_product_attention support

        if use_dot_product_attention:
            if attention_mask is not None:
                # Ensure attention_mask has the correct shape for broadcasting
                # Expected shape: [batch_size, num_heads, query_seq_len,
                # key_seq_len].
                mask_expansion_axis = -len(self._attention_axes) * 2 - 1
                len_attention_scores_shape = 4  # Only accepts 4D inputs
                for _ in range(
                    len_attention_scores_shape - len(attention_mask.shape)
                ):
                    attention_mask = ops.expand_dims(
                        attention_mask, axis=mask_expansion_axis
                    )
                attention_mask = ops.cast(attention_mask, dtype="bool")
            # Directly compute the attention output using dot-product attention
            attention_output = ops.dot_product_attention(
                query=query,
                key=key,
                value=value,
                bias=None,
                mask=attention_mask,
                scale=self._inverse_sqrt_key_dim,
                is_causal=False,
                flash_attention=self._flash_attention,
            )
            return attention_output, None

        # Default behavior without flash attention, with explicit attention
        # scores
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        )

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        key = self.k_qdq(key)
        query = self.q_qdq(query)
        attention_scores = ops.einsum(self._dot_product_equation, key, query)

        # Apply the mask using the custom masked softmax
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )

        # Apply dropout to the attention scores if needed
        if self._dropout > 0.0:
            final_attn_scores = self._dropout_layer(
                attention_scores, training=training
            )
        else:
            final_attn_scores = attention_scores

        # `context_layer` = [B, T, N, H]
        final_attn_scores = self.a_qdq(final_attn_scores)
        value = self.v_qdq(value)
        attention_output = ops.einsum(
            self._combine_equation, final_attn_scores, value
        )
        return attention_output, attention_scores
    # fmt on


verify_api(MultiHeadAttention, QStaticMultiHeadAttention, "_compute_attention")


@register_static_quantized_layer(CachedGemma3Attention)
class QStaticCachedGemma3Attention(SaveableLayerMixin, CachedGemma3Attention):
    """Statically quantized CachedGemma3Attention layer."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a CachedGemma3Attention instance for static quantization.

        Args:
            orig (CachedGemma3Attention): Original layer instance.
            weight_dtype (jnp.dtype): Dtype for quantized weights.
            activation_dtype (jnp.dtype): Dtype for quantized activations.

        Returns:
            CachedGemma3Attention: Updated layer instance.
        """
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.q_qdq = StaticQDQLayer("q_qdq", activation_dtype, orig.dtype_policy, False, const_scale)
        orig.k_qdq = StaticQDQLayer("k_qdq", activation_dtype, orig.dtype_policy, False, const_scale)
        orig.attention_softmax_qdq = StaticQDQLayer(
            "attention_softmax_qdq", activation_dtype, orig.dtype_policy, False, const_scale
        )
        orig.v_qdq = StaticQDQLayer("v_qdq", activation_dtype, orig.dtype_policy, False, const_scale)
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def add_observers(self):
        """Attach observer layers for calibration.

        Returns:
            None: Adds observer layers.
        """
        self.q_qdq.add_observers()
        self.k_qdq.add_observers()
        self.attention_softmax_qdq.add_observers()
        self.v_qdq.add_observers()

    def add_variables(self):
        """Create quantization variables for activation QDQ.

        Returns:
            None: Initializes QDQ helper variables.
        """
        self.q_qdq.add_variables()
        self.k_qdq.add_variables()
        self.attention_softmax_qdq.add_variables()
        self.v_qdq.add_variables()

    def convert(self):
        """Compute activation calibration values for QDQ helpers.

        Returns:
            None: Updates QDQ helpers with calibrated values.
        """
        self.q_qdq.convert()
        self.k_qdq.convert()
        self.attention_softmax_qdq.convert()
        self.v_qdq.convert()

    def post_quantization_cleanup(self):
        """Finalize static quantization and mark the layer as quantized.

        Returns:
            None: Cleans up observers and marks quantized state.
        """
        self._tracker.unlock()
        self.q_qdq.post_quantization_cleanup()
        self.k_qdq.post_quantization_cleanup()
        self.attention_softmax_qdq.post_quantization_cleanup()
        self.v_qdq.post_quantization_cleanup()
        self._is_quantized = True
        self._tracker.lock()

    def _compute_attention(
        self,
        q,
        k,
        v,
        attention_mask,
        training=False,
        cache_update_index=0,
    ):
        """Compute attention with static activation quantization.

        Args:
            q (jnp.ndarray): Query tensor.
            k (jnp.ndarray): Key tensor.
            v (jnp.ndarray): Value tensor.
            attention_mask (Optional[jnp.ndarray]): Optional attention mask.
            training (bool): Training mode flag.
            cache_update_index (int): Cache update index for generation.

        Returns:
            jnp.ndarray: Attention output tensor.
        """
        if self.query_head_dim_normalize:
            query_normalization = 1 / np.sqrt(self.head_dim)
        else:
            query_normalization = 1 / np.sqrt(self.hidden_dim // self.num_query_heads)

        if self.use_sliding_window_attention and attention_mask is not None:
            attention_mask = self._mask_sliding_window(
                attention_mask,
                cache_update_index=cache_update_index,
            )

        if self._use_fused_attention_op():
            logger.warning(
                "Flash attention is not supported in static quantization yet. "
                "Falling back to standard attention computation."
            )

        q *= ops.cast(query_normalization, dtype=q.dtype)
        q_shape = ops.shape(q)
        q = ops.reshape(
            q,
            (
                *q_shape[:-2],
                self.num_key_value_heads,
                self.num_query_heads // self.num_key_value_heads,
                q_shape[-1],
            ),
        )
        b, q_len, _, _, h = ops.shape(q)

        # Fallback to standard attention if flash attention is disabled
        q = self.q_qdq(q)
        k = self.k_qdq(k)
        attention_logits = ops.einsum("btkgh,bskh->bkgts", q, k)
        if self.logit_soft_cap is not None:
            attention_logits = ops.divide(attention_logits, self.logit_soft_cap)
            attention_logits = ops.multiply(ops.tanh(attention_logits), self.logit_soft_cap)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :, :]

        attention_softmax = self.softmax(attention_logits, mask=attention_mask)

        if self.dropout:
            attention_softmax = self.dropout_layer(attention_softmax, training=training)

        attention_softmax = self.attention_softmax_qdq(attention_softmax)
        v = self.v_qdq(v)
        results = ops.einsum("bkgts,bskh->btkgh", attention_softmax, v)
        return ops.reshape(results, (b, q_len, self.num_query_heads, h))


verify_api(CachedGemma3Attention, QStaticCachedGemma3Attention, "_compute_attention")


@register_static_quantized_layer(Gemma3VisionAttention)
class QStaticGemma3VisionAttention(SaveableLayerMixin, Gemma3VisionAttention):
    """Statically quantized Gemma3VisionAttention layer."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a Gemma3VisionAttention instance for static quantization.

        Args:
            orig (Gemma3VisionAttention): Original layer instance.
            weight_dtype (jnp.dtype): Dtype for quantized weights.
            activation_dtype (jnp.dtype): Dtype for quantized activations.
            const_scale (bool): Whether to use constant scales.
            const_weight (bool): ignored, included for API consistency.

        Returns:
            Gemma3VisionAttention: Updated layer instance.
        """
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.query_qdq = StaticQDQLayer("query_qdq", activation_dtype, orig.dtype_policy, False, const_scale)
        orig.key_qdq = StaticQDQLayer("key_qdq", activation_dtype, orig.dtype_policy, False, const_scale)
        orig.dropout_attention_probs_qdq = StaticQDQLayer(
            "dropout_attention_probs_qdq", activation_dtype, orig.dtype_policy, False, const_scale
        )
        orig.value_qdq = StaticQDQLayer("value_qdq", activation_dtype, orig.dtype_policy, False, const_scale)
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def add_observers(self):
        """Attach observer layers for calibration.

        Returns:
            None: Adds observer layers.
        """
        self.query_qdq.add_observers()
        self.key_qdq.add_observers()
        self.dropout_attention_probs_qdq.add_observers()
        self.value_qdq.add_observers()

    def add_variables(self):
        """Create quantization variables for activation QDQ.

        Returns:
            None: Initializes QDQ helper variables.
        """
        self.query_qdq.add_variables()
        self.key_qdq.add_variables()
        self.dropout_attention_probs_qdq.add_variables()
        self.value_qdq.add_variables()

    def convert(self):
        """Compute activation calibration values for QDQ helpers.

        Returns:
            None: Updates QDQ helpers with calibrated values.
        """
        self.query_qdq.convert()
        self.key_qdq.convert()
        self.dropout_attention_probs_qdq.convert()
        self.value_qdq.convert()

    def post_quantization_cleanup(self):
        """Finalize static quantization and mark the layer as quantized.

        Returns:
            None: Cleans up observers and marks quantized state.
        """
        self._tracker.unlock()
        self.query_qdq.post_quantization_cleanup()
        self.key_qdq.post_quantization_cleanup()
        self.dropout_attention_probs_qdq.post_quantization_cleanup()
        self.value_qdq.post_quantization_cleanup()
        self._is_quantized = True
        self._tracker.lock()

    def call(
        self,
        x,
        attention_mask=None,
        return_attention_scores=None,
        training=False,
    ):
        """Compute vision attention with static activation quantization.

        Args:
            x (jnp.ndarray): Input tensor.
            attention_mask (Optional[jnp.ndarray]): Optional attention mask.
            return_attention_scores (Optional[bool]): Whether to return attention scores.
            training (bool): Training mode flag.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Attention output and attention probabilities.
        """
        batch_size = ops.shape(x)[0]
        mixed_query_layer = self.query_proj(inputs=x)
        mixed_key_layer = self.key_proj(inputs=x)
        mixed_value_layer = self.value_proj(inputs=x)
        query_layer = self._transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self._transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self._transpose_for_scores(mixed_value_layer, batch_size)

        # Scaled dot product between key and query = raw attention scores.
        key_layer_transposed = ops.transpose(key_layer, axes=[0, 1, 3, 2])
        query_layer = self.query_qdq(query_layer)
        key_layer_transposed = self.key_qdq(key_layer_transposed)
        attention_scores = ops.matmul(query_layer, key_layer_transposed)
        dk = ops.cast(ops.sqrt(self.head_dim), dtype=attention_scores.dtype)
        attention_scores = ops.divide(attention_scores, dk)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in the
            # call() function)
            attention_scores = ops.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        dropout_attention_probs = self.dropout_layer(inputs=attention_probs, training=training)

        dropout_attention_probs = self.dropout_attention_probs_qdq(dropout_attention_probs)
        value_layer = self.value_qdq(value_layer)
        attn_output = ops.matmul(dropout_attention_probs, value_layer)
        attn_output = ops.transpose(attn_output, axes=[0, 2, 1, 3])

        # (batch_size, seq_len_q, hidden_dim)
        seq_len_q = ops.shape(attn_output)[1]
        attn_output = ops.reshape(attn_output, (batch_size, seq_len_q, self.hidden_dim))

        attn_output = self.out_proj(attn_output, training=training)
        return (attn_output, attention_probs)


verify_api(Gemma3VisionAttention, QStaticGemma3VisionAttention, "call")


# @register_static_quantized_layer(RotaryEmbedding)
class QStaticRotaryEmbedding(SaveableLayerMixin, RotaryEmbedding):
    """Statically quantized RotaryEmbedding layer."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a RotaryEmbedding instance for static quantization.

        Args:
            orig (RotaryEmbedding): Original layer instance.
            weight_dtype (jnp.dtype): Dtype for quantized weights.
            activation_dtype (jnp.dtype): Dtype for quantized activations.
            const_scale (bool): Whether to use constant scales.
            const_weight (bool): ignored, included for API consistency.

        Returns:
            RotaryEmbedding: Updated layer instance.
        """
        orig._tracker.unlock()
        orig.__class__ = cls
        orig._is_int8 = jnp.issubdtype(activation_dtype, jnp.integer)
        orig.positions_qdq = StaticQDQLayer(
            "positions_qdq", activation_dtype, orig.dtype_policy, orig._is_int8, const_scale
        )
        orig.inverse_freq_qdq = StaticQDQLayer(
            "inverse_freq_qdq", activation_dtype, orig.dtype_policy, orig._is_int8, const_scale
        )
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def add_observers(self):
        """Attach observer layers for calibration.

        Returns:
            None: Adds observer layers.
        """
        self.positions_qdq.add_observers()
        self.inverse_freq_qdq.add_observers()

    def add_variables(self):
        """Create quantization variables for activation QDQ.

        Returns:
            None: Initializes QDQ helper variables.
        """
        self.positions_qdq.add_variables()
        self.inverse_freq_qdq.add_variables()

    def convert(self):
        """Compute activation calibration values for QDQ helpers.

        Returns:
            None: Updates QDQ helpers with calibrated values.
        """
        self.positions_qdq.convert()
        self.inverse_freq_qdq.convert()

    def post_quantization_cleanup(self):
        """Finalize static quantization and mark the layer as quantized.

        Returns:
            None: Cleans up observers and marks quantized state.
        """
        self._tracker.unlock()
        self.positions_qdq.post_quantization_cleanup()
        self.inverse_freq_qdq.post_quantization_cleanup()
        self._is_quantized = True
        self._tracker.lock()

    def _compute_cos_sin_embedding(self, inputs, start_index=0, positions=None):
        """Compute cosine/sine embeddings with quantized inputs.

        Args:
            inputs (jnp.ndarray): Input tensor.
            start_index (int): Starting index for positions.
            positions (Optional[jnp.ndarray]): Optional explicit positions tensor.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Cosine and sine embeddings.
        """
        feature_axis = len(inputs.shape) - 1
        sequence_axis = 1

        rotary_dim = ops.shape(inputs)[feature_axis]
        inverse_freq = self._get_inverse_freq(rotary_dim)

        if positions is None:
            positions = self._compute_positions(inputs, start_index)
        else:
            positions = ops.cast(positions, "float32")

        positions = positions / ops.cast(self.scaling_factor, "float32")
        positions = self.positions_qdq(positions)
        inverse_freq = self.inverse_freq_qdq(inverse_freq)
        freq = ops.einsum("i,j->ij", positions, inverse_freq)
        embedding = ops.stack((freq, freq), axis=-2)
        embedding = ops.reshape(embedding, (*ops.shape(freq)[:-1], ops.shape(freq)[-1] * 2))

        # Reshape the embedding to be broadcastable with input shape.
        if feature_axis < sequence_axis:
            embedding = ops.transpose(embedding)
        for axis in range(len(inputs.shape)):
            if axis != sequence_axis and axis != feature_axis:
                embedding = ops.expand_dims(embedding, axis)

        cos_emb = ops.cast(ops.cos(embedding), self.compute_dtype)
        sin_emb = ops.cast(ops.sin(embedding), self.compute_dtype)
        return cos_emb, sin_emb


# verify_api(RotaryEmbedding, QStaticRotaryEmbedding, "_compute_cos_sin_embedding")


@register_static_quantized_layer(ReversibleEmbedding)
class QStaticReversibleEmbedding(SaveableLayerMixin, ReversibleEmbedding):
    """Statically quantized ReversibleEmbedding layer."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a ReversibleEmbedding instance for static quantization.

        Args:
            orig (ReversibleEmbedding): Original layer instance.
            weight_dtype (jnp.dtype): Dtype for quantized weights.
            activation_dtype (jnp.dtype): Dtype for quantized activations.

        Returns:
            ReversibleEmbedding: Updated layer instance.
        """
        orig._tracker.unlock()
        orig.__class__ = cls
        orig._is_int8 = jnp.issubdtype(activation_dtype, jnp.integer)
        orig.inputs_qdq = StaticQDQLayer("inputs_qdq", activation_dtype, orig.dtype_policy, orig._is_int8, const_scale)
        orig.kernel_qdq = StaticQDQLayer("kernel_qdq", weight_dtype, orig.dtype_policy, orig._is_int8, const_scale)
        orig.const_scale = const_scale
        orig.const_weight = const_weight
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def add_observers(self):
        """Attach observer layers for calibration.

        Returns:
            None: Adds observer layers.
        """
        self.inputs_qdq.add_observers()
        self.kernel_qdq.add_observers()

    def add_variables(self):
        """Create quantization variables for activation QDQ.

        Returns:
            None: Initializes QDQ helper variables.
        """
        self.inputs_qdq.add_variables()
        self.kernel_qdq.add_variables()

    def convert(self):
        """Compute activation calibration values for QDQ helpers.

        Returns:
            None: Updates QDQ helpers with calibrated values.
        """
        # TODO maybe make kernel (offline) quantization for reversible embedding (self.embeddings in our path) ?
        self.inputs_qdq.convert()
        self.kernel_qdq.convert()

    def post_quantization_cleanup(self):
        """Finalize static quantization and mark the layer as quantized.

        Returns:
            None: Cleans up observers and marks quantized state.
        """
        self._tracker.unlock()
        self.inputs_qdq.post_quantization_cleanup()
        self.kernel_qdq.post_quantization_cleanup()
        self._is_quantized = True
        self._tracker.lock()

    def call(self, inputs, reverse=False):
        """Compute forward or reverse embedding with static quantization.

        Args:
            inputs (jnp.ndarray): Input tensor.
            reverse (bool): Whether to compute the reverse embedding.

        Returns:
            jnp.ndarray: Embedded outputs or logits.
        """
        if reverse:
            if self.tie_weights:
                kernel = ops.transpose(ops.convert_to_tensor(self.embeddings))
            else:
                kernel = self.reverse_embeddings
            if self.reverse_dtype is not None:
                inputs = ops.cast(inputs, self.reverse_dtype)
                kernel = ops.cast(kernel, self.reverse_dtype)
            inputs = self.inputs_qdq(inputs)
            kernel = self.kernel_qdq(kernel)
            logits = ops.matmul(inputs, kernel)
            # Optionally soft-cap logits.
            if self.logit_soft_cap is not None:
                soft_cap = self.logit_soft_cap
                logits = ops.tanh(logits / soft_cap) * soft_cap
            return logits

        return super(ReversibleEmbedding, self).call(inputs)


verify_api(ReversibleEmbedding, QStaticReversibleEmbedding, "call")
