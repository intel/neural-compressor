# Copyright (c) 2025-2026 Intel Corporation
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
from neural_compressor.jax.utils.utility import get_dequantize_fun, get_quantize_fun, get_scale, verify_api

if keras.config.backend() != "jax":
    raise ValueError(
        f"{__name__} only supports JAX backend, but the current backend is {keras.config.backend()}.\n"
        'Consider setting KERAS_BACKEND env var to "jax".'
    )

static_quant_mapping = {}


def register_static_quantized_layer(clso):
    """Register quantized layer class for original layer class."""

    def decorator(cls):
        static_quant_mapping[clso] = cls
        return cls

    return decorator


class MinMaxObserver(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, name="min_max")
        # Track running min/max as non-trainable weights
        self.min_val = self.add_weight(
            shape=(), initializer=keras.initializers.Constant(float("inf")), trainable=False, name="min_val"
        )
        self.max_val = self.add_weight(
            shape=(), initializer=keras.initializers.Constant(float("-inf")), trainable=False, name="max_val"
        )
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if 0 not in inputs.shape:
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

            self.min_val.assign(keras.ops.minimum(self.min_val, batch_min))
            self.max_val.assign(keras.ops.maximum(self.max_val, batch_max))
        return inputs

    def build(self, input_shape):
        pass

    def get_calibrated_range(self):
        return ops.array((self.min_val, self.max_val))


class StaticQDQLayer(keras.layers.Layer, SaveableLayerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if not self._is_quantized:
            x = self.input_observer(inputs, mask=mask)
            return x
        ascale = self.ascale.value
        x = self.aquantfun(inputs, ascale)
        x = self.adequantfun(x, ascale)
        return x

    @classmethod
    def prepare(cls, orig, activation_dtype):
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.activation_dtype = activation_dtype
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def post_quantization_cleanup(self):
        self._tracker.unlock()
        if hasattr(self, "_layers") and hasattr(self, "input_observer"):
            if self.input_observer in self._layers:
                self._layers.remove(self.input_observer)
                del self.input_observer
        self._is_quantized = True
        self._tracker.lock()

    def convert(self):
        self._tracker.unlock()
        arange = self.input_observer.get_calibrated_range()
        ascale = get_scale(arange, self.activation_dtype)
        if ascale == jnp.inf:
            logger.warning(
                f"Activation scale is inf for layer {self._path}. This may be caused by missing calibration data. "
                "Please make sure to run calibration with representative dataset."
            )
            self._is_quantized = False
            self._tracker.lock()
            return
        self.ascale.assign(ascale)
        self._tracker.lock()

    def add_variables(self):
        self._tracker.unlock()
        self.ascale = self.add_weight(
            name="activation_scale",
            shape=(1,),
            initializer="zeros",
            trainable=False,
            autocast=False,
            dtype=self.compute_dtype,
        )
        self.aquantfun = get_quantize_fun(dtype=self.activation_dtype)
        self.adequantfun = get_dequantize_fun(dtype=self.compute_dtype)
        self._tracker.lock()

    def add_observers(self):
        self._tracker.unlock()
        self.input_observer = MinMaxObserver()
        self._tracker.lock()


class QStaticDenseMixin(SaveableLayerMixin):

    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype):
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.weight_dtype = weight_dtype
        orig.activation_dtype = activation_dtype
        orig._is_quantized = False
        orig.kernel_shape = orig.kernel.shape
        orig._tracker.lock()
        return orig

    def add_variables(self):
        self._tracker.unlock()

        self.ascale = self.add_weight(
            name="activation_scale",
            shape=(1,),
            initializer="zeros",
            trainable=False,
            autocast=False,
            dtype=self.compute_dtype,
        )
        self.wscale = self.add_weight(
            name="weight_scale",
            shape=(1,),
            initializer="zeros",
            trainable=False,
            autocast=False,
            dtype=self.compute_dtype,
        )
        self.w = self.add_weight(
            name="quantized_kernel",
            shape=self.kernel_shape,
            initializer="zeros",
            trainable=False,
            dtype=self.weight_dtype,
            autocast=False,
        )

        self.aquantfun = get_quantize_fun(dtype=self.activation_dtype)
        self.adequantfun = get_dequantize_fun(dtype=self.compute_dtype)
        self.wquantfun = get_quantize_fun(dtype=self.weight_dtype)
        self.wdequantfun = get_dequantize_fun(dtype=self.compute_dtype)

        self._tracker.lock()

    def post_quantization_cleanup(self):
        self._tracker.unlock()
        if hasattr(self, "_kernel") and self._kernel in self._trainable_variables:
            self._trainable_variables.remove(self._kernel)
            del self._kernel

        if hasattr(self, "_layers") and hasattr(self, "input_observer"):
            if self.input_observer in self._layers:
                self._layers.remove(self.input_observer)
                del self.input_observer

        self.call = self.call_quantized
        self._is_quantized = True
        self._tracker.lock()

    def add_observers(self):
        self._tracker.unlock()
        self.input_observer = MinMaxObserver()
        self._tracker.lock()

    def convert(self):
        self._tracker.unlock()
        arange = self.input_observer.get_calibrated_range()
        ascale = get_scale(arange, self.activation_dtype)
        if ascale == jnp.inf:
            logger.warning(
                f"Activation scale is inf for layer {self._path}. This may be caused by missing calibration data. "
                "Please make sure to run calibration with representative dataset."
            )
            self.call = super().call
            self._is_quantized = False
            self._tracker.lock()
            return
        self.ascale.assign(ascale)
        wscale = get_scale(self.kernel, self.weight_dtype)
        self.wscale.assign(wscale)
        w = self.wquantfun(self.kernel, scale=self.wscale)
        self.w.assign(w)
        self._tracker.lock()

    @property
    def kernel(self):
        if self._is_quantized:
            w = self.wdequantfun(self.w.value, self.wscale.value)
            return w
        ret = super().kernel
        return ret.value

    def call(self, inputs, training=None):
        x = self.input_observer(inputs)
        x = super().call(x, training=training)
        return x

    def call_quantized(self, inputs, training=None):
        ascale = self.ascale.value
        x = self.aquantfun(inputs, ascale)
        x = self.adequantfun(x, ascale)
        x = super().call(x, training=training)
        return x


@register_static_quantized_layer(keras.layers.Dense)
class QStaticDense(QStaticDenseMixin, keras.layers.Dense):
    pass


verify_api(Dense, QStaticDense, "call")


@register_static_quantized_layer(EinsumDense)
class QStaticEinsumDense(QStaticDenseMixin, EinsumDense):
    pass


verify_api(EinsumDense, QStaticEinsumDense, "call")


@register_static_quantized_layer(MultiHeadAttention)
class QStaticMultiHeadAttention(MultiHeadAttention, SaveableLayerMixin):
    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype):
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.q_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="q_qdq"), activation_dtype)
        orig.k_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="k_qdq"), activation_dtype)
        orig.a_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="a_qdq"), activation_dtype)
        orig.v_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="v_qdq"), activation_dtype)
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def add_variables(self):
        self.q_qdq.add_variables()
        self.k_qdq.add_variables()
        self.a_qdq.add_variables()
        self.v_qdq.add_variables()

    def add_observers(self):
        self.q_qdq.add_observers()
        self.k_qdq.add_observers()
        self.a_qdq.add_observers()
        self.v_qdq.add_observers()

    def post_quantization_cleanup(self):
        self._tracker.unlock()
        self.q_qdq.post_quantization_cleanup()
        self.k_qdq.post_quantization_cleanup()
        self.a_qdq.post_quantization_cleanup()
        self.v_qdq.post_quantization_cleanup()
        self._is_quantized = True
        self._tracker.lock()

    def convert(self):
        self.q_qdq.convert()
        self.k_qdq.convert()
        self.a_qdq.convert()
        self.v_qdq.convert()

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
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """

        query = ops.multiply(query, ops.cast(self._inverse_sqrt_key_dim, query.dtype))
        key = self.k_qdq(key)
        query = self.q_qdq(query)
        attention_scores = ops.einsum(self._dot_product_equation, key, query)

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        attention_scores = self.a_qdq(attention_scores)
        value = self.v_qdq(value)
        attention_output = ops.einsum(self._combine_equation, attention_scores, value)
        return attention_output, attention_scores


verify_api(MultiHeadAttention, QStaticMultiHeadAttention, "_compute_attention")


@register_static_quantized_layer(CachedGemma3Attention)
class QStaticCachedGemma3Attention(CachedGemma3Attention, SaveableLayerMixin):
    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype):
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.q_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="q_qdq"), activation_dtype)
        orig.k_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="k_qdq"), activation_dtype)
        orig.attention_softmax_qdq = StaticQDQLayer.prepare(
            StaticQDQLayer(name="attention_softmax_qdq"), activation_dtype
        )
        orig.v_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="v_qdq"), activation_dtype)
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def add_variables(self):
        self.q_qdq.add_variables()
        self.k_qdq.add_variables()
        self.attention_softmax_qdq.add_variables()
        self.v_qdq.add_variables()

    def add_observers(self):
        self.q_qdq.add_observers()
        self.k_qdq.add_observers()
        self.attention_softmax_qdq.add_observers()
        self.v_qdq.add_observers()

    def post_quantization_cleanup(self):
        self._tracker.unlock()
        self.q_qdq.post_quantization_cleanup()
        self.k_qdq.post_quantization_cleanup()
        self.attention_softmax_qdq.post_quantization_cleanup()
        self.v_qdq.post_quantization_cleanup()
        self._is_quantized = True
        self._tracker.lock()

    def convert(self):
        self.q_qdq.convert()
        self.k_qdq.convert()
        self.attention_softmax_qdq.convert()
        self.v_qdq.convert()

    def _compute_attention(
        self,
        q,
        k,
        v,
        attention_mask,
        training=False,
        cache_update_index=0,
    ):
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
        orig_dtype = attention_logits.dtype
        attention_softmax = self.softmax(attention_logits, mask=attention_mask)
        # attention_softmax = ops.cast(attention_softmax, orig_dtype)

        if self.dropout:
            attention_softmax = self.dropout_layer(attention_softmax, training=training)

        attention_softmax = self.attention_softmax_qdq(attention_softmax)
        v = self.v_qdq(v)
        results = ops.einsum("bkgts,bskh->btkgh", attention_softmax, v)
        return ops.reshape(results, (b, q_len, self.num_query_heads, h))


verify_api(CachedGemma3Attention, QStaticCachedGemma3Attention, "_compute_attention")


@register_static_quantized_layer(Gemma3VisionAttention)
class QStaticGemma3VisionAttention(Gemma3VisionAttention, SaveableLayerMixin):
    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype):
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.query_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="query_qdq"), activation_dtype)
        orig.key_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="key_qdq"), activation_dtype)
        orig.dropout_attention_probs_qdq = StaticQDQLayer.prepare(
            StaticQDQLayer(name="dropout_attention_probs_qdq"), activation_dtype
        )
        orig.value_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="value_qdq"), activation_dtype)
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def add_variables(self):
        self.query_qdq.add_variables()
        self.key_qdq.add_variables()
        self.dropout_attention_probs_qdq.add_variables()
        self.value_qdq.add_variables()

    def add_observers(self):
        self.query_qdq.add_observers()
        self.key_qdq.add_observers()
        self.dropout_attention_probs_qdq.add_observers()
        self.value_qdq.add_observers()

    def post_quantization_cleanup(self):
        self._tracker.unlock()
        self.query_qdq.post_quantization_cleanup()
        self.key_qdq.post_quantization_cleanup()
        self.dropout_attention_probs_qdq.post_quantization_cleanup()
        self.value_qdq.post_quantization_cleanup()
        self._is_quantized = True
        self._tracker.lock()

    def convert(self):
        self.query_qdq.convert()
        self.key_qdq.convert()
        self.dropout_attention_probs_qdq.convert()
        self.value_qdq.convert()

    def call(
        self,
        x,
        attention_mask=None,
        return_attention_scores=None,
        training=False,
    ):
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
class QStaticRotaryEmbedding(RotaryEmbedding, SaveableLayerMixin):
    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype):
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.positions_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="positions_qdq"), activation_dtype)
        orig.inverse_freq_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="inverse_freq_qdq"), activation_dtype)
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def add_variables(self):
        self.positions_qdq.add_variables()
        self.inverse_freq_qdq.add_variables()

    def add_observers(self):
        self.positions_qdq.add_observers()
        self.inverse_freq_qdq.add_observers()

    def post_quantization_cleanup(self):
        self._tracker.unlock()
        self.positions_qdq.post_quantization_cleanup()
        self.inverse_freq_qdq.post_quantization_cleanup()
        self._is_quantized = True
        self._tracker.lock()

    def convert(self):
        self.positions_qdq.convert()
        self.inverse_freq_qdq.convert()

    def _compute_cos_sin_embedding(self, inputs, start_index=0, positions=None):
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
class QStaticReversibleEmbedding(ReversibleEmbedding, SaveableLayerMixin):
    @classmethod
    def prepare(cls, orig, weight_dtype, activation_dtype):
        orig._tracker.unlock()
        orig.__class__ = cls
        orig.inputs_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="inputs_qdq"), activation_dtype)
        orig.kernel_qdq = StaticQDQLayer.prepare(StaticQDQLayer(name="kernel_qdq"), activation_dtype)
        orig._is_quantized = False
        orig._tracker.lock()
        return orig

    def add_variables(self):
        self.inputs_qdq.add_variables()
        self.kernel_qdq.add_variables()

    def add_observers(self):
        self.inputs_qdq.add_observers()
        self.kernel_qdq.add_observers()

    def post_quantization_cleanup(self):
        self._tracker.unlock()
        self.inputs_qdq.post_quantization_cleanup()
        self.kernel_qdq.post_quantization_cleanup()
        self._is_quantized = True
        self._tracker.lock()

    def convert(self):
        # TODO maybe make kernel (offline) quantization for reversible embedding (self.embeddings in our path) ?
        self.inputs_qdq.convert()
        self.kernel_qdq.convert()

    def call(self, inputs, reverse=False):
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
