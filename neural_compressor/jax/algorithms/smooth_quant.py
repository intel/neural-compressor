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
"""SmoothQuant for JAX/Keras models, applied by folding (no extra ops at inference).

SmoothQuant (Xiao et al., 2023) migrates activation outliers into the weights. For a matmul ``y = x @ W`` and a
per-input-channel factor ``s``::

    y = (x / s) @ (diag(s) W),    s_j = max|x_j| ** alpha / max|W_j| ** (1 - alpha)

The division ``x / s`` is folded into the layer that produces ``x`` (a normalization gain or the output channels of
a previous linear layer), so the smoothed model computes the same function with the same graph, and quantizing it
gives the same kernels and speed as quantizing the original model.

Which layers can be folded depends on the model's data flow, so the foldable groups are described per decoder-block
class in ``SMOOTH_RULES``. Gemma3 decoder blocks are supported; other models are left unchanged (with a warning).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import keras
import numpy as np
from keras import ops

from neural_compressor.common import logger

SCALE_MIN = 1e-2  # clip range of the smoothing factors (avoids extreme rescaling of nearly-dead channels)
SCALE_MAX = 1e2


class ChannelAbsMaxObserver(keras.layers.Layer):
    """Calibration observer: running per-channel maximum of ``|x|`` over all but the last ``num_channel_dims`` axes.

    The statistic is a non-trainable variable, so it is updated inside a jitted, stateless calibration function
    (e.g. the CausalLM generate function used during quantization) like the other INC observers. Input autocasting is
    disabled, so the observed layer receives its input unchanged (same dtype) during calibration.
    """

    def __init__(self, channel_shape: Tuple[int, ...], **kwargs):
        """Initialize the observer.

        Args:
            channel_shape (Tuple[int, ...]): Shape of the trailing (channel) axes of the observed input.
            **kwargs: Keyword arguments for the base layer.
        """
        kwargs.setdefault("autocast", False)
        super().__init__(name="smooth_quant_observer", **kwargs)
        self.channel_shape = tuple(int(d) for d in channel_shape)
        self.channel_abs_max = self.add_weight(
            shape=self.channel_shape,
            initializer="zeros",
            trainable=False,
            name="channel_abs_max",
            dtype="float32",
        )

    def call(self, inputs):
        """Update the per-channel abs-max statistic and pass the inputs through unchanged.

        Args:
            inputs (jnp.ndarray): Observed tensor, shape ``(..., *channel_shape)``.

        Returns:
            jnp.ndarray: The unchanged inputs.
        """
        if 0 not in inputs.shape:
            axes = tuple(range(len(inputs.shape) - len(self.channel_shape)))
            batch_max = ops.max(ops.abs(ops.cast(inputs, "float32")), axis=axes)
            self.channel_abs_max.assign(ops.maximum(self.channel_abs_max, batch_max))
        return inputs

    def build(self, input_shape):
        """No additional variables."""
        pass


# --- kernel layout helpers -------------------------------------------------------------------------------------------
def _np(x) -> np.ndarray:
    """Variable or array (any float dtype, including bfloat16) as a float32 numpy array."""
    return np.asarray(x).astype(np.float32)


def _equation(layer) -> Tuple[str, str, str]:
    """Return (input, kernel, output) subscripts of a Dense or EinsumDense layer."""
    if isinstance(layer, keras.layers.EinsumDense):
        lhs, out = layer.equation.split("->")
        inp, ker = lhs.split(",")
        return inp, ker, out
    if isinstance(layer, keras.layers.Dense):
        return "...i", "io", "...o"
    raise TypeError(f"SmoothQuant supports Dense and EinsumDense layers, got {type(layer).__name__}")


def input_axes(layer) -> Tuple[int, ...]:
    """Kernel axes contracted with the layer input (the input channels), in input order."""
    inp, ker, out = _equation(layer)
    contracted = [c for c in inp if c in ker and c not in out]
    if inp[-len(contracted) :] != "".join(contracted):
        raise ValueError(f"{layer.name}: contracted axes must be the trailing input axes ({layer.equation})")
    return tuple(ker.index(c) for c in contracted)


def output_axes(layer) -> Tuple[int, ...]:
    """Kernel axes that become output channels, in kernel order."""
    inp, ker, out = _equation(layer)
    return tuple(i for i, c in enumerate(ker) if c in out and c not in inp)


def _move_to_front(kernel: np.ndarray, axes: Tuple[int, ...]) -> np.ndarray:
    rest = [i for i in range(kernel.ndim) if i not in axes]
    return np.transpose(kernel, list(axes) + rest)


def kernel_input_abs_max(layer) -> np.ndarray:
    """Per-input-channel max |W| of a layer's kernel, shape = input channel shape."""
    axes = input_axes(layer)
    k = _move_to_front(np.abs(_np(layer.kernel)), axes)
    return k.reshape(k.shape[: len(axes)] + (-1,)).max(-1)


def _broadcast(s: np.ndarray, ndim: int, axes: Tuple[int, ...]) -> np.ndarray:
    shape = [1] * ndim
    for a, d in zip(axes, s.shape):
        shape[a] = d
    return s.reshape(shape)


def scale_kernel(layer, s: np.ndarray, axes: Tuple[int, ...]):
    """Multiply the kernel by ``s`` along ``axes``."""
    k = _np(layer.kernel)
    layer.kernel.assign((k * _broadcast(s, k.ndim, axes)).astype(np.float32))


# --- groups and rules -------------------------------------------------------------------------------------------------
@dataclass
class SmoothGroup:
    """Consumers sharing one input, the layer whose input is observed, and how ``1 / s`` is folded.

    ``head_repeat`` > 1 means the consumer input channels are ``(n_q, h)`` while the absorbing layer produces
    ``(n_kv, h)`` with ``n_q = n_kv * head_repeat`` (grouped-query attention); one factor per kv channel is shared
    by the query heads of its group.

    ``absorb(s)`` folds ``1 / s`` into the producing layer and returns the factors that the stored (possibly
    bfloat16-rounded) weights actually implement; the consumers are scaled by those, so the fold stays exact.
    """

    name: str
    consumers: List[keras.layers.Layer]
    absorb: Callable[[np.ndarray], np.ndarray]
    head_repeat: int = 1
    alpha: float = 0.5
    observer: Optional[ChannelAbsMaxObserver] = field(default=None, repr=False)

    @property
    def observed(self):
        return self.consumers[0]


def _assign(variable, value: np.ndarray) -> np.ndarray:
    """Assign and return the stored value (rounded to the variable dtype, e.g. bfloat16)."""
    variable.assign(value.astype(np.float32))
    return _np(variable)


def _fold_into_gemma_rms_norm(norm) -> Callable[[np.ndarray], np.ndarray]:
    """Fold 1/s into a Gemma RMSNorm.

    It computes ``x_hat * (1 + scale)``, so dividing its output by s gives ``1 + scale' = (1 + scale) / s``. In
    bfloat16, ``scale'`` close to -1 is coarse (e.g. s = 100 stores a gain 17% too large), so the returned factors are
    the ones the stored gain implements, ``(1 + scale) / (1 + scale'_stored)``. Channels whose gain would round to
    zero are left unsmoothed.
    """

    def absorb(s):
        gain = 1.0 + _np(norm.scale)
        stored = 1.0 + _assign(norm.scale, gain / s - 1.0)
        lost = (stored == 0) & (gain != 0)
        if lost.any():
            s = np.where(lost, 1.0, s)
            stored = 1.0 + _assign(norm.scale, gain / s - 1.0)
        return np.where(stored != 0, gain / np.where(stored != 0, stored, 1.0), 1.0).astype(np.float32)

    return absorb


def _fold_into_output_channels(layer) -> Callable[[np.ndarray], np.ndarray]:
    """Divide the output channels (and bias) of a linear producer by s."""

    def absorb(s):
        axes = output_axes(layer)
        k = _np(layer.kernel)
        layer.kernel.assign((k / _broadcast(s, k.ndim, axes)).astype(np.float32))
        if getattr(layer, "bias", None) is not None:
            b = _np(layer.bias)
            layer.bias.assign((b / s.reshape(b.shape)).astype(np.float32))
        return s  # per-element rounding of the weights only, no systematic per-channel error

    return absorb


def gemma3_decoder_block_groups(block) -> List[SmoothGroup]:
    """Foldable groups of a keras-hub ``Gemma3DecoderBlock``.

    - q/k/v projections, input = ``pre_attention_norm`` output -> fold into that norm
    - gate/up projections, input = ``pre_ffw_norm`` output -> fold into that norm
    - down projection, input = ``gelu(gate) * up`` -> fold into the up projection output channels
    - attention output projection, input = ``softmax(qk) @ v`` -> fold into the value projection output channels
      (shared by the query heads of each kv head)
    """
    att = block.attention
    n_q, n_kv = att.num_query_heads, att.num_key_value_heads
    return [
        SmoothGroup(
            "qkv",
            [att.query_dense, att.key_dense, att.value_dense],
            _fold_into_gemma_rms_norm(block.pre_attention_norm),
        ),
        SmoothGroup("gate_up", [block.gating_ffw, block.gating_ffw_2], _fold_into_gemma_rms_norm(block.pre_ffw_norm)),
        SmoothGroup("down", [block.ffw_linear], _fold_into_output_channels(block.gating_ffw_2)),
        SmoothGroup(
            "attention_output", [att.output_dense], _fold_into_output_channels(att.value_dense), head_repeat=n_q // n_kv
        ),
    ]


SMOOTH_RULES: Dict[str, Callable[[keras.layers.Layer], List[SmoothGroup]]] = {
    "Gemma3DecoderBlock": gemma3_decoder_block_groups,
}


# Per-layer SmoothQuant settings: ``None`` if the layer is not quantized, else ``(enabled, alpha)``.
LayerSettings = Callable[[keras.layers.Layer], Optional[Tuple[bool, float]]]


def _resolve(group: SmoothGroup, layer_settings: Optional[LayerSettings], alpha: float) -> Optional[float]:
    """Alpha for a group, or None to skip it.

    A group is smoothed only if all of its quantized consumers enable SmoothQuant with the same alpha: the consumers
    share one input and one factor, so a consumer configured without SmoothQuant (or with another alpha) would
    otherwise be changed against its configuration. Consumers that are not quantized don't matter (the fold is exact).
    """
    if layer_settings is None:
        return alpha
    settings = [st for st in (layer_settings(c) for c in group.consumers) if st is not None]
    if not settings or not any(enabled for enabled, _ in settings):
        return None
    if not all(enabled for enabled, _ in settings) or len({a for _, a in settings}) > 1:
        logger.warning(
            "SmoothQuant: %s skipped, its consumers (%s) share an input but have different SmoothQuant settings %s.",
            group.name,
            ", ".join(c.path or c.name for c in group.consumers),
            settings,
        )
        return None
    return settings[0][1]


def find_smooth_groups(model, layer_settings: Optional[LayerSettings] = None, alpha: float = 0.5) -> List[SmoothGroup]:
    """Collect the foldable groups of all supported blocks and their alpha.

    Args:
        model (keras.Model): Model to inspect.
        layer_settings (Callable, optional): Per-layer settings, see ``LayerSettings``. Default: every group with
            ``alpha``.
        alpha (float): Alpha used when ``layer_settings`` is not given.

    Returns:
        List[SmoothGroup]: Groups to smooth.
    """
    groups = []
    for layer in model._flatten_layers(include_self=False, recursive=True):
        rule = SMOOTH_RULES.get(type(layer).__name__)
        if rule is None:
            continue
        for g in rule(layer):
            g.name = f"{layer.path or layer.name}/{g.name}"
            group_alpha = _resolve(g, layer_settings, alpha)
            if group_alpha is None:
                continue
            if not 0.0 <= group_alpha <= 1.0:
                raise ValueError(f"smooth_quant_alpha must be in [0, 1], got {group_alpha} for {g.name}")
            g.alpha = group_alpha
            groups.append(g)
    return groups


# --- observers --------------------------------------------------------------------------------------------------------
def _attach_observer(group: SmoothGroup):
    layer = group.observed
    channel_shape = tuple(layer.kernel.shape[a] for a in input_axes(layer))
    obs = ChannelAbsMaxObserver(channel_shape, dtype=layer.dtype_policy)
    original_call = layer.call

    def call(inputs, *args, **kwargs):
        return original_call(obs(inputs), *args, **kwargs)

    layer._tracker.unlock()
    layer._smooth_quant_observer = obs
    # remember whether `call` was an instance attribute: after calibration the instance attribute must be removed,
    # not re-assigned, or it would shadow the class method of a later class swap (static quantization prepare)
    layer._smooth_quant_original_call = layer.__dict__.get("call")
    layer._tracker.lock()
    layer.call = call
    group.observer = obs


def _detach_observer(group: SmoothGroup):
    layer = group.observed
    layer._tracker.unlock()
    if layer._smooth_quant_original_call is None:
        del layer.call
    else:
        layer.call = layer._smooth_quant_original_call
    if group.observer in layer._layers:
        layer._layers.remove(group.observer)
    del layer._smooth_quant_observer
    del layer._smooth_quant_original_call
    layer._tracker.lock()
    group.observer = None


# --- main entry -------------------------------------------------------------------------------------------------------
def smoothing_factors(act_abs_max: np.ndarray, weight_abs_max: np.ndarray, alpha: float) -> np.ndarray:
    """``s = act^alpha / w^(1 - alpha)``, clipped to [SCALE_MIN, SCALE_MAX]; 1 where a statistic is zero."""
    a = np.asarray(act_abs_max, dtype=np.float64)
    w = np.asarray(weight_abs_max, dtype=np.float64)
    valid = (a > 0) & (w > 0)
    s = np.ones_like(a)
    s[valid] = a[valid] ** alpha / w[valid] ** (1.0 - alpha)
    return np.clip(s, SCALE_MIN, SCALE_MAX).astype(np.float32)


def apply_smooth_quant(
    model: keras.Model,
    calib_function: Callable,
    alpha: float = 0.5,
    layer_settings: Optional[LayerSettings] = None,
) -> None:
    """Smooth the model in place: calibrate per-channel activation ranges, then fold the factors into the weights.

    Only blocks with a rule in ``SMOOTH_RULES`` are smoothed (Gemma3 decoder blocks); for other models a warning is
    logged and nothing is run (no calibration pass).

    Args:
        model (keras.Model): FP32/BF16 model (not yet quantized), modified in place.
        calib_function (Callable): Runs calibration data through the model (the static-quantization calib function).
        alpha (float): Migration strength in [0, 1] (0.5 balances activation and weight ranges), used for all
            groups when ``layer_settings`` is not given.
        layer_settings (Callable, optional): Per-layer settings from the quantization configs, see ``LayerSettings``.
    """
    groups = find_smooth_groups(model, layer_settings, alpha)
    if not groups:
        logger.warning(
            "SmoothQuant: no layers to smooth (supported blocks: %s, with SmoothQuant enabled for their projections); "
            "model left unchanged.",
            ", ".join(SMOOTH_RULES),
        )
        return
    for g in groups:
        _attach_observer(g)
    try:
        calib_function(model)
        act = {g.name: _np(g.observer.channel_abs_max) for g in groups}
    finally:
        for g in groups:
            _detach_observer(g)

    for g in groups:
        a = act[g.name]
        w = np.max([kernel_input_abs_max(c) for c in g.consumers], axis=0)
        if not np.any(a > 0):
            logger.warning("SmoothQuant: %s saw no calibration data; skipped.", g.name)
            continue
        if g.head_repeat > 1:  # (n_q, h) -> (n_kv, h): one factor per kv channel
            a = a.reshape((-1, g.head_repeat) + a.shape[1:]).max(1)
            w = w.reshape((-1, g.head_repeat) + w.shape[1:]).max(1)
        s = g.absorb(smoothing_factors(a, w, g.alpha))  # the factors the stored producer weights implement
        s_consumer = np.repeat(s, g.head_repeat, axis=0) if g.head_repeat > 1 else s
        for c in g.consumers:
            scale_kernel(c, s_consumer, input_axes(c))
        logger.debug("SmoothQuant: %s factors in [%.3g, %.3g]", g.name, float(s.min()), float(s.max()))
    logger.info(
        "SmoothQuant: smoothed %d groups (alpha %s).", len(groups), ", ".join(sorted({str(g.alpha) for g in groups}))
    )
