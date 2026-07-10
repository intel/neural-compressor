"""
Layer-adaptive and step-adaptive attention routing.

Routes each transformer layer and/or denoising step to a different
attention configuration based on call counting. Designed for
CogVideoX-2B (30 layers per step) and modeled on the SelectiveAttention
pattern from tasks/triton_mx/layer_skip.py.

Usage:
    # Threshold-based (deep layers get higher precision)
    router = LayerAdaptiveAttention(early_fn, deep_fn, deep_start=21)

    # Set-based (specific layers get higher precision)
    router = LayerAdaptiveAttention(early_fn, deep_fn, high_precision_layers={12,13,14,15,16,17,18,19})

    # Step+layer combined routing
    router = StepLayerAdaptiveAttention(
        fp4_fn, fp8_fn, sdpa_fn,
        critical_steps={1,2,3,4,5,6,7,9},
        moderate_steps={8,10,11,12,13,14,15,16,17,18,19,49},
        deep_start=21,
    )
"""

from typing import Callable, Optional, Set


class LayerAdaptiveAttention:
    """
    Routes each transformer layer to one of two attention functions
    based on call_count % num_layers.

    Supports two modes:
    - Threshold: Layers [0, deep_start) → early_fn, [deep_start, num_layers) → deep_fn
    - Set-based: Layers in high_precision_layers → deep_fn, others → early_fn

    If high_precision_layers is provided, it takes precedence over deep_start.

    Args:
        early_fn: Attention function for early/default layers.
        deep_fn: Attention function for deep/high-precision layers.
        num_layers: Number of transformer layers per denoising step (30 for CogVideoX-2B).
        deep_start: First layer index that uses deep_fn (threshold mode).
        high_precision_layers: Set of layer indices that use deep_fn (set mode).
    """

    def __init__(
        self,
        early_fn: Callable,
        deep_fn: Callable,
        num_layers: int = 30,
        deep_start: int = 21,
        high_precision_layers: Optional[Set[int]] = None,
    ):
        self.early_fn = early_fn
        self.deep_fn = deep_fn
        self.num_layers = num_layers
        self.deep_start = deep_start
        self.high_precision_layers = high_precision_layers
        self._call_count = 0

    def _is_high_precision(self, layer_idx: int) -> bool:
        if self.high_precision_layers is not None:
            return layer_idx in self.high_precision_layers
        return layer_idx >= self.deep_start

    def __call__(self, *args, **kwargs):
        layer_idx = self._call_count % self.num_layers
        self._call_count += 1
        if self._is_high_precision(layer_idx):
            return self.deep_fn(*args, **kwargs)
        return self.early_fn(*args, **kwargs)

    def reset(self):
        """Reset call counter (e.g., between inference runs)."""
        self._call_count = 0

    @property
    def stats(self):
        """Return routing statistics."""
        total = self._call_count
        steps = total // self.num_layers
        if self.high_precision_layers is not None:
            deep_per_step = len(self.high_precision_layers)
        else:
            deep_per_step = self.num_layers - self.deep_start
        early_per_step = self.num_layers - deep_per_step
        return {
            "total_calls": total,
            "steps_completed": steps,
            "early_layers_per_step": early_per_step,
            "deep_layers_per_step": deep_per_step,
            "early_fraction": early_per_step / self.num_layers,
        }


class StepLayerAdaptiveAttention:
    """
    Combined step-level and layer-level routing.

    Three-tier step routing:
    - critical_steps → sdpa_fn (full precision)
    - moderate_steps → fp8_fn (FP8 everywhere)
    - remaining steps → layer-adaptive (early_fn for most layers, deep_fn for sensitive layers)

    Within the layer-adaptive tier, routing follows the same logic as
    LayerAdaptiveAttention (threshold or set-based).

    Args:
        early_fn: Low-precision attention (e.g., FP4-QK/FP8-PV).
        deep_fn: Medium-precision attention (e.g., full MXFP8) for sensitive layers.
        sdpa_fn: Full-precision SDPA for critical steps.
        fp8_fn: FP8 attention for moderate steps (all layers).
        critical_steps: Step indices → SDPA (highest precision).
        moderate_steps: Step indices → FP8 (medium precision).
        num_layers: Layers per denoising step.
        deep_start: Layer threshold for deep_fn (within layer-adaptive steps).
        high_precision_layers: Optional set-based layer routing.
    """

    def __init__(
        self,
        early_fn: Callable,
        deep_fn: Callable,
        sdpa_fn: Callable,
        fp8_fn: Optional[Callable] = None,
        critical_steps: Optional[Set[int]] = None,
        moderate_steps: Optional[Set[int]] = None,
        num_layers: int = 30,
        deep_start: int = 21,
        high_precision_layers: Optional[Set[int]] = None,
    ):
        self.early_fn = early_fn
        self.deep_fn = deep_fn
        self.sdpa_fn = sdpa_fn
        self.fp8_fn = fp8_fn or deep_fn  # fallback to deep_fn if no separate FP8
        self.critical_steps = critical_steps or set()
        self.moderate_steps = moderate_steps or set()
        self.num_layers = num_layers
        self.deep_start = deep_start
        self.high_precision_layers = high_precision_layers
        self._call_count = 0

    def _is_high_precision_layer(self, layer_idx: int) -> bool:
        if self.high_precision_layers is not None:
            return layer_idx in self.high_precision_layers
        return layer_idx >= self.deep_start

    def __call__(self, *args, **kwargs):
        step_idx = self._call_count // self.num_layers
        layer_idx = self._call_count % self.num_layers
        self._call_count += 1

        if step_idx in self.critical_steps:
            return self.sdpa_fn(*args, **kwargs)
        elif step_idx in self.moderate_steps:
            return self.fp8_fn(*args, **kwargs)
        elif self._is_high_precision_layer(layer_idx):
            return self.deep_fn(*args, **kwargs)
        else:
            return self.early_fn(*args, **kwargs)

    def reset(self):
        """Reset call counter."""
        self._call_count = 0

    @property
    def stats(self):
        """Return routing statistics."""
        total = self._call_count
        steps = total // self.num_layers
        return {
            "total_calls": total,
            "steps_completed": steps,
            "critical_steps": len(self.critical_steps),
            "moderate_steps": len(self.moderate_steps),
            "layer_adaptive_steps": max(0, steps - len(self.critical_steps | self.moderate_steps)),
        }
