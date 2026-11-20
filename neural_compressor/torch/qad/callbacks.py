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
"""Optional diagnostics that stay out of the normal training path."""

from __future__ import annotations

import torch
from transformers import TrainerCallback


class GradDebugCallback(TrainerCallback):
    """Report non-finite gradients by parameter name before the optimizer step."""

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        """Scan gradients and print the offending parameters."""
        if model is None:
            return
        bad = 0
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            local = grad.to_local() if hasattr(grad, "to_local") else grad
            if not torch.isfinite(local).all():
                bad += 1
                print(
                    f"[grad-nan] step={state.global_step} {name} "
                    f"shape={tuple(local.shape)} dtype={local.dtype} "
                    f"finite={torch.isfinite(local).float().mean().item():.3f}",
                    flush=True,
                )
        if bad == 0:
            print(f"[grad-nan] step={state.global_step} all grads finite", flush=True)


class StopAfterSteps(TrainerCallback):
    """Stop after N optimizer steps while keeping a longer LR schedule."""

    def __init__(self, step: int) -> None:
        """Store the stopping step."""
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        """Request a stop once the target step is reached."""
        if self.step and state.global_step >= self.step:
            control.should_training_stop = True


__all__ = ["GradDebugCallback", "StopAfterSteps"]
