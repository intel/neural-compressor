# Copyright (c) 2025 Intel Corporation
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
"""Stateful ``TensorQuantizer`` module for QAT, QAD, and PTQ simulation."""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from .config import QuantizerConfig
from .formats import QuantFormat
from .ops.qdq import (
    QuantBackendName,
    fake_quant_dynamic,
    fake_quant_fp8_static_amax,
    ste_fake_quant,
)


class TensorQuantizer(nn.Module):
    """Per-slot (weight / input / output) fake-quantizer module.

    Guarantees:
    * ``amax`` buffer is strictly maintained in ``torch.float32`` even if ``model.to(torch.bfloat16)``
      or ``model.half()`` is called on an ancestor module.
    * When ``x.requires_grad`` is ``False`` (e.g. frozen base weights under LoRA or teacher/eval runs),
      QDQ executes directly under ``torch.no_grad()`` without building autograd nodes.
    * When ``x.requires_grad`` is ``True``, :func:`ste_fake_quant` applies an identity STE backward
      with zero saved tensors.
    """

    amax: Optional[torch.Tensor]

    def __init__(self, config: Optional[QuantizerConfig] = None) -> None:
        super().__init__()
        self.config: QuantizerConfig = config if config is not None else QuantizerConfig(enabled=False)
        self.format: QuantFormat = self.config.format
        self._dynamic: bool = self.config.dynamic
        self._enabled: bool = self.config.enabled and self.format.name != "bf16"
        self._calibrating: bool = False
        self.observer: str = self.config.observer
        self.backend: QuantBackendName = self.config.backend

        init_amax = (
            torch.tensor(float(self.config.amax), dtype=torch.float32) if self.config.amax is not None else None
        )
        self.register_buffer("amax", init_amax, persistent=True)

    # ------------------------------------------------------------------
    # State properties & controls
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        self._enabled = bool(value) and self.format.name != "bf16"

    @property
    def enabled(self) -> bool:
        """Alias for :attr:`is_enabled`."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.is_enabled = value

    @property
    def is_dynamic(self) -> bool:
        return self._dynamic

    @property
    def scheme(self) -> str:
        return "dynamic" if self._dynamic else "static"

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @is_calibrating.setter
    def is_calibrating(self, value: bool) -> None:
        self._calibrating = bool(value)

    @property
    def calibrating(self) -> bool:
        """Alias for :attr:`is_calibrating`."""
        return self._calibrating

    @calibrating.setter
    def calibrating(self, value: bool) -> None:
        self.is_calibrating = value

    def enable_quant(self) -> None:
        self.is_enabled = True

    def disable_quant(self) -> None:
        self.is_enabled = False

    def enable_calib(self) -> None:
        if not self._dynamic and self._enabled:
            self._calibrating = True

    def disable_calib(self) -> None:
        self._calibrating = False

    def reset_amax(self) -> None:
        self.amax = None

    def set_amax(self, value: Union[float, torch.Tensor]) -> None:
        if isinstance(value, torch.Tensor):
            val_t = value.detach().to(dtype=torch.float32).reshape(())
        else:
            device = self.amax.device if self.amax is not None else torch.device("cpu")
            val_t = torch.tensor(float(value), dtype=torch.float32, device=device)
        if (val_t < 0).any():
            raise ValueError(f"amax must be non-negative, got {float(val_t)}")
        self.amax = val_t

    def finalize(self) -> None:
        """Exit calibration mode and verify static quantizers hold a valid ``amax``."""
        self._calibrating = False
        if self._enabled and not self._dynamic and self.amax is None:
            raise RuntimeError(
                f"Static TensorQuantizer ({self.format.name}) was finalized without a calibrated amax."
            )

    # ------------------------------------------------------------------
    # Precision & checkpoint invariants
    # ------------------------------------------------------------------

    def _apply(self, fn):
        orig_amax = self.amax
        super()._apply(fn)
        if orig_amax is not None and self.amax is not None:
            self.amax = orig_amax.to(device=self.amax.device, dtype=torch.float32)
        return self

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        amax_key = f"{prefix}amax"
        if amax_key in state_dict and self.amax is None and state_dict[amax_key] is not None:
            src = state_dict[amax_key]
            self.amax = torch.empty_like(src, dtype=torch.float32)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if self.amax is not None and self.amax.dtype != torch.float32:
            self.amax = self.amax.to(torch.float32)

    # ------------------------------------------------------------------
    # Observation & Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        """Update running ``amax`` from ``x`` without affecting autograd."""
        local_x = x.to_local() if hasattr(x, "to_local") else x
        if local_x.numel() == 0:
            return
        cur = local_x.detach().abs().amax().to(dtype=torch.float32)
        if self.amax is None:
            self.amax = cur
        else:
            if self.amax.device != cur.device:
                self.amax = self.amax.to(device=cur.device, dtype=torch.float32)
            torch.maximum(self.amax, cur, out=self.amax)

    @torch.no_grad()
    def sync_amax_across_distributed(self, group: Optional[dist.ProcessGroup] = None) -> None:
        """All-reduce ``amax`` with ``ReduceOp.MAX`` across ranks when distributed is active."""
        if self.amax is None or not dist.is_available() or not dist.is_initialized():
            return
        backend = dist.get_backend(group)
        orig_device = self.amax.device
        work_tensor = self.amax
        if backend == "nccl" and orig_device.type != "cuda" and torch.cuda.is_available():
            work_tensor = self.amax.cuda()
        dist.all_reduce(work_tensor, op=dist.ReduceOp.MAX, group=group)
        if work_tensor is not self.amax:
            self.amax = work_tensor.to(device=orig_device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._enabled or self.format.name == "bf16":
            return x
        if self._calibrating:
            if not self._dynamic:
                self.observe(x)
            return x
        if not self._dynamic and self.amax is None:
            raise RuntimeError(
                f"Static TensorQuantizer ({self.format.name}) has no calibrated amax; "
                "run calibrate_static_amax() or provide quantizer_state_path."
            )
        if not torch.is_grad_enabled() or not x.requires_grad:
            if self._dynamic:
                return fake_quant_dynamic(x, self.format, backend=self.backend)
            assert self.amax is not None
            return fake_quant_fp8_static_amax(x, self.amax, backend=self.backend)
        return ste_fake_quant(
            x,
            self.format,
            self.amax,
            dynamic=self._dynamic,
            backend=self.backend,
        )

    def extra_repr(self) -> str:
        if not self._enabled or self.format.name == "bf16":
            return "disabled"
        parts = [f"format={self.format.name}", f"scheme={self.scheme}"]
        if self.format.block_size is not None:
            parts.append(f"block_size={self.format.block_size}")
        if not self._dynamic:
            amax_str = f"{float(self.amax.detach().cpu()):.6g}" if self.amax is not None else "uncalibrated"
            parts.append(f"amax={amax_str}")
        if self._calibrating:
            parts.append("calibrating=True")
        return ", ".join(parts)


@contextlib.contextmanager
def disable_fake_quant(model: nn.Module) -> Iterator[None]:
    """Temporarily disable all :class:`TensorQuantizer` modules inside ``model``."""
    saved: list[tuple[TensorQuantizer, bool]] = []
    for mod in model.modules():
        if isinstance(mod, TensorQuantizer):
            saved.append((mod, mod.is_enabled))
            mod.is_enabled = False
    try:
        yield
    finally:
        for mod, prev in saved:
            mod.is_enabled = prev


__all__ = ["TensorQuantizer", "disable_fake_quant"]
