"""Dynamic quantized layer implementations for JAX-backed Keras models."""

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
from keras_hub.layers import ReversibleEmbedding
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

dynamic_quant_mapping = {}


def register_dynamic_quantized_layer(clso):
    """Register quantized layer class for an original layer class.

    Args:
        clso (type): Original layer class to map to a quantized implementation.

    Returns:
        Callable: Decorator that registers the quantized class.
    """

    def decorator(cls):
        """Attach the quantized class to the dynamic mapping.

        Args:
            cls (type): Quantized layer class to register.

        Returns:
            type: The same class, for decorator chaining.
        """
        dynamic_quant_mapping[clso] = cls
        return cls

    return decorator


class DynamicQDQLayer(SaveableLayerMixin, keras.layers.Layer):
    """Layer that applies dynamic quantize-dequantize to activations."""

    def __init__(self, name, activation_dtype, asymmetric=False):
        """Initialize the dynamic QDQ helper layer.

        Args:
            name (str): Layer name.
            activation_dtype (jnp.dtype): Activation dtype used for quantization.
            asymmetric (bool): Whether to use asymmetric quantization.

        Returns:
            None: Initializes the layer instance.
        """
        super().__init__(name=name)
        self.activation_dtype = activation_dtype
        self._is_asymmetric = asymmetric
        self.supports_masking = True

    def add_variables(self):
        """Create quantization helper functions for activations.

        Returns:
            None: Initializes quantization functions.
        """
        self._tracker.unlock()
        self.aquantfun = get_quantize_fun(dtype=self.activation_dtype, asymmetric=self._is_asymmetric)
        self.adequantfun = get_dequantize_fun(dtype=self.compute_dtype, asymmetric=self._is_asymmetric)
        self._tracker.lock()

    def call_symmetric(self, inputs, batch_min_max, mask=None):
        """Apply symmetric quantization to inputs.

        Args:
            inputs (jnp.ndarray): Input tensor.
            batch_min_max (jnp.ndarray): Min/max tensor for the batch.
            mask (Optional[jnp.ndarray]): Optional mask tensor.

        Returns:
            jnp.ndarray: Quantized-dequantized tensor.
        """
        ascale, _ = get_q_params(batch_min_max, self.activation_dtype, asymmetric=False)
        x = self.aquantfun(inputs, ascale)
        x = self.adequantfun(x, ascale)
        return x

    def call_asymmetric(self, inputs, batch_min_max, mask=None):
        """Apply asymmetric quantization to inputs.

        Args:
            inputs (jnp.ndarray): Input tensor.
            batch_min_max (jnp.ndarray): Min/max tensor for the batch.
            mask (Optional[jnp.ndarray]): Optional mask tensor.

        Returns:
            jnp.ndarray: Quantized-dequantized tensor.
        """
        ascale, azero_point = get_q_params(batch_min_max, self.activation_dtype, asymmetric=True)
        x = self.aquantfun(inputs, ascale, azero_point)
        x = self.adequantfun(x, ascale, azero_point)
        return x

    def call(self, inputs, mask=None):
        """Apply dynamic activation quantize-dequantize.

        Args:
            inputs (jnp.ndarray): Input tensor.
            mask (Optional[jnp.ndarray]): Optional mask tensor.

        Returns:
            jnp.ndarray: Tensor with quantize-dequantize applied.
        """
        if any([dim == 0 for dim in inputs.shape]):
            # Skip quantization for zero-size inputs
            return inputs
        if mask is not None:
            # Expand mask to match input dimensions if needed
            if len(mask.shape) < len(inputs.shape):
                for _ in range(len(inputs.shape) - len(mask.shape)):
                    mask = ops.expand_dims(mask, axis=-1)
            # Apply mask to exclude masked positions
            masked_inputs_min = ops.where(mask, inputs, float("inf"))
            masked_inputs_max = ops.where(mask, inputs, float("-inf"))
            batch_min = keras.ops.min(masked_inputs_min)
            batch_max = keras.ops.max(masked_inputs_max)
        else:
            batch_min = keras.ops.min(inputs)
            batch_max = keras.ops.max(inputs)

        batch_min_max = keras.ops.array((batch_min, batch_max))

        if self._is_asymmetric:
            return self.call_asymmetric(inputs, batch_min_max, mask)
        return self.call_symmetric(inputs, batch_min_max, mask)


class QDynamicDenseMixin(SaveableLayerMixin):
    """Mixin that adds dynamic quantization to dense-like layers."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a dense-like layer instance for dynamic quantization.

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
        orig.const_scale = const_scale
        orig.const_weight = const_weight
        orig._is_int8 = jnp.issubdtype(activation_dtype, jnp.integer)
        orig.input_qdq = DynamicQDQLayer("input_qdq", activation_dtype, orig._is_int8)
        if const_scale:
            orig._const_variables = ["wscale"]
        else:
            orig._const_variables = []
        if const_weight:
            orig._const_variables.append("w")
        orig._tracker.lock()
        return orig

    def add_variables(self):
        """Create quantization variables and cached weight tensor.

        Returns:
            None: Initializes quantization variables.
        """
        self._tracker.unlock()
        self.input_qdq.add_variables()
        wscale, _ = get_q_params(self._kernel.value, self.weight_dtype, asymmetric=False)
        self.wscale = self.add_weight(
            name="weight_scale",
            shape=wscale.shape,
            initializer=keras.initializers.Constant(wscale),
            trainable=False,
            dtype=self.compute_dtype,
        )
        wquantfun = get_quantize_fun(dtype=self.weight_dtype, asymmetric=False)
        self.wdequantfun = get_dequantize_fun(dtype=self.compute_dtype, asymmetric=False)
        self.w = self.add_weight(
            name="kernel_quant",
            shape=self._kernel.shape,
            initializer="zeros",
            trainable=False,
            dtype=self.weight_dtype,
            autocast=False,
        )

        self.w.assign(wquantfun(self._kernel.value, scale=self.wscale.value))
        self._tracker.lock()

    def post_quantization_cleanup(self):
        """Remove original weights after quantization is complete.

        Returns:
            None: Cleans up original weights.
        """
        self._tracker.unlock()
        self._trainable_variables.remove(self._kernel)
        del self._kernel

        # convert variables to attributes (const) if needed
        for name in self._const_variables:
            var = getattr(self, name)
            value = jnp.array(var.value)
            self._non_trainable_variables[:] = [v for v in self._non_trainable_variables if v is not var]
            setattr(self, name, value)

        self._tracker.lock()

    @property
    def kernel(self):
        """Return the dequantized kernel tensor.

        Returns:
            jnp.ndarray: Dequantized kernel tensor.
        """

        if self.const_weight:
            w = self.w
        else:
            w = self.w.value
        if self.const_scale:
            wscale = self.wscale
        else:
            wscale = self.wscale.value

        w = self.wdequantfun(w, wscale)
        return w

    def call(self, inputs, training=None):
        """Apply quantized input processing before the dense computation.

        Args:
            inputs (jnp.ndarray): Input tensor.
            training (Optional[bool]): Training mode flag.

        Returns:
            jnp.ndarray: Layer output tensor.
        """
        x = self.input_qdq(inputs)
        x = super().call(x, training=training)
        return x


@register_dynamic_quantized_layer(Dense)
class QDynamicDense(QDynamicDenseMixin, Dense):
    """Dynamically quantized Dense layer."""

    pass


verify_api(Dense, QDynamicDense, "call")


@register_dynamic_quantized_layer(EinsumDense)
class QDynamicEinsumDense(QDynamicDenseMixin, EinsumDense):
    """Dynamically quantized EinsumDense layer."""

    pass


verify_api(EinsumDense, QDynamicEinsumDense, "call")


@register_dynamic_quantized_layer(MultiHeadAttention)
class QDynamicMultiHeadAttention(SaveableLayerMixin, MultiHeadAttention):
    """Dynamically quantized MultiHeadAttention layer."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a MultiHeadAttention instance for dynamic quantization.

        Args:
            orig (keras.layers.MultiHeadAttention): Original layer instance.
            weight_dtype (jnp.dtype): Dtype for quantized weights.
            activation_dtype (jnp.dtype): Dtype for quantized activations.
            const_scale (bool): ignored, included for API consistency.
            const_weight (bool): ignored, included for API consistency.

        Returns:
            keras.layers.MultiHeadAttention: Updated layer instance.
        """
        orig._tracker.unlock()
        orig.__class__ = cls
        orig._is_int8 = jnp.issubdtype(activation_dtype, jnp.integer)
        orig.q_qdq = DynamicQDQLayer("q_qdq", activation_dtype, False)
        orig.k_qdq = DynamicQDQLayer("k_qdq", activation_dtype, orig._is_int8)
        orig.a_qdq = DynamicQDQLayer("a_qdq", activation_dtype, orig._is_int8)
        orig.v_qdq = DynamicQDQLayer("v_qdq", activation_dtype, False)
        orig._tracker.lock()
        return orig

    def add_variables(self):
        """Create quantization helper layers for activations.

        Returns:
            None: Initializes quantization helper layers.
        """
        self.q_qdq.add_variables()
        self.k_qdq.add_variables()
        self.a_qdq.add_variables()
        self.v_qdq.add_variables()

    def post_quantization_cleanup(self):
        """Finalize dynamic quantization with no extra cleanup.

        Returns:
            None: Keeps the layer ready for inference.
        """
        pass

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


verify_api(MultiHeadAttention, QDynamicMultiHeadAttention, "_compute_attention")


@register_dynamic_quantized_layer(CachedGemma3Attention)
class QDynamicCachedGemma3Attention(SaveableLayerMixin, CachedGemma3Attention):
    """Dynamically quantized CachedGemma3Attention layer."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a CachedGemma3Attention instance for dynamic quantization.

        Args:
            orig (CachedGemma3Attention): Original layer instance.
            weight_dtype (jnp.dtype): Dtype for quantized weights.
            activation_dtype (jnp.dtype): Dtype for quantized activations.
            const_scale (bool): ignored, included for API consistency.
            const_weight (bool): ignored, included for API consistency.

        Returns:
            CachedGemma3Attention: Updated layer instance.
        """
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.qdq = DynamicQDQLayer("qdq", activation_dtype, False)
        orig._tracker.lock()
        return orig

    def add_variables(self):
        """Create activation QDQ helper layer.

        Returns:
            None: Initializes activation helper layer.
        """
        self.qdq.add_variables()

    def post_quantization_cleanup(self):
        """Finalize dynamic quantization with no extra cleanup.

        Returns:
            None: Keeps the layer ready for inference.
        """
        pass

    def _compute_attention(
        self,
        q,
        k,
        v,
        attention_mask,
        training=False,
        cache_update_index=0,
    ):
        """Compute attention with dynamic activation quantization.

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
                "Flash attention is not supported in dynamic quantization yet. "
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
        q = self.qdq(q)
        k = self.qdq(k)
        attention_logits = ops.einsum("btkgh,bskh->bkgts", q, k)
        if self.logit_soft_cap is not None:
            attention_logits = ops.divide(attention_logits, self.logit_soft_cap)
            attention_logits = ops.multiply(ops.tanh(attention_logits), self.logit_soft_cap)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :, :]

        attention_softmax = self.softmax(attention_logits, mask=attention_mask)

        if self.dropout:
            attention_softmax = self.dropout_layer(attention_softmax, training=training)

        attention_softmax = self.qdq(attention_softmax)
        v = self.qdq(v)
        results = ops.einsum("bkgts,bskh->btkgh", attention_softmax, v)
        return ops.reshape(results, (b, q_len, self.num_query_heads, h))


verify_api(CachedGemma3Attention, QDynamicCachedGemma3Attention, "_compute_attention")


@register_dynamic_quantized_layer(Gemma3VisionAttention)
class QDynamicGemma3VisionAttention(SaveableLayerMixin, Gemma3VisionAttention):
    """Dynamically quantized Gemma3VisionAttention layer."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a Gemma3VisionAttention instance for dynamic quantization.

        Args:
            orig (Gemma3VisionAttention): Original layer instance.
            weight_dtype (jnp.dtype): Dtype for quantized weights.
            activation_dtype (jnp.dtype): Dtype for quantized activations.
            const_scale (bool): ignored, included for API consistency.
            const_weight (bool): ignored, included for API consistency.

        Returns:
            Gemma3VisionAttention: Updated layer instance.
        """
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.qdq = DynamicQDQLayer("qdq", activation_dtype, False)
        orig._tracker.lock()
        return orig

    def add_variables(self):
        """Create activation QDQ helper layer.

        Returns:
            None: Initializes activation helper layer.
        """
        self.qdq.add_variables()

    def post_quantization_cleanup(self):
        """Finalize dynamic quantization with no extra cleanup.

        Returns:
            None: Keeps the layer ready for inference.
        """
        pass

    def call(
        self,
        x,
        attention_mask=None,
        return_attention_scores=None,
        training=False,
    ):
        """Compute vision attention with quantized activations.

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
        query_layer = self.qdq(query_layer)
        key_layer_transposed = self.qdq(key_layer_transposed)
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

        dropout_attention_probs = self.qdq(dropout_attention_probs)
        value_layer = self.qdq(value_layer)
        attn_output = ops.matmul(dropout_attention_probs, value_layer)
        attn_output = ops.transpose(attn_output, axes=[0, 2, 1, 3])

        # (batch_size, seq_len_q, hidden_dim)
        seq_len_q = ops.shape(attn_output)[1]
        attn_output = ops.reshape(attn_output, (batch_size, seq_len_q, self.hidden_dim))

        attn_output = self.out_proj(attn_output, training=training)
        return (attn_output, attention_probs)


verify_api(Gemma3VisionAttention, QDynamicGemma3VisionAttention, "call")


@register_dynamic_quantized_layer(ReversibleEmbedding)
class QDynamicReversibleEmbedding(SaveableLayerMixin, ReversibleEmbedding):
    """Dynamically quantized ReversibleEmbedding layer."""

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype, const_scale=False, const_weight=False):
        """Convert a ReversibleEmbedding instance for dynamic quantization.

        Args:
            orig (ReversibleEmbedding): Original layer instance.
            weight_dtype (jnp.dtype): Dtype for quantized weights.
            activation_dtype (jnp.dtype): Dtype for quantized activations.
            const_scale (bool): ignored, included for API consistency.
            const_weight (bool): ignored, included for API consistency.

        Returns:
            ReversibleEmbedding: Updated layer instance.
        """
        orig._tracker.unlock()
        orig.__class__ = cls
        orig._is_int8 = jnp.issubdtype(activation_dtype, jnp.integer)
        orig.inputs_qdq = DynamicQDQLayer("inputs_qdq", activation_dtype, orig._is_int8)
        orig.kernel_qdq = DynamicQDQLayer("kernel_qdq", activation_dtype, False)
        orig._tracker.lock()
        return orig

    def add_variables(self):
        """Create activation QDQ helper layers.

        Returns:
            None: Initializes activation helper layers.
        """
        self.inputs_qdq.add_variables()
        self.kernel_qdq.add_variables()

    def post_quantization_cleanup(self):
        """Finalize dynamic quantization with no extra cleanup.

        Returns:
            None: Keeps the layer ready for inference.
        """
        pass

    # TODO maybe make kernel (offline) quantization for reversible embedding (self.embeddings in our path) ?

    def call(self, inputs, reverse=False):
        """Compute forward or reverse embedding with activation quantization.

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


verify_api(ReversibleEmbedding, QDynamicReversibleEmbedding, "call")
