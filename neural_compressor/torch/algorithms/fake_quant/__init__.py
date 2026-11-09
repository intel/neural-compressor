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
"""Algorithm entry for fake quantization (QAT/QAD building block)."""

from __future__ import annotations

from typing import Any, Dict

import torch

from neural_compressor.common.utils import FAKE_QUANT, Mode
from neural_compressor.torch.quantization.fake_quant.calibrator import calibrate
from neural_compressor.torch.quantization.fake_quant.options import ModuleQuantConfig
from neural_compressor.torch.quantization.fake_quant.plan import PlanReport, apply_plan
from neural_compressor.torch.quantization.fake_quant.quantizer import TensorQuantizer
from neural_compressor.torch.utils import logger, register_algo

_QUANTIZER_FIELDS = ("weight", "input", "output")


def _pending_static(plan: Dict[str, ModuleQuantConfig]) -> list[str]:
    pending = []
    for name, config in plan.items():
        for field_name in _QUANTIZER_FIELDS:
            quantizer_cfg = getattr(config, field_name)
            if quantizer_cfg is None or quantizer_cfg.scheme != "static":
                continue
            if quantizer_cfg.amax is None:
                pending.append(f"{name}.{field_name}")
    return pending


def _set_calibrating(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, TensorQuantizer) and module.scheme == "static" and module.enabled:
            module.calibrating = enabled


@register_algo(name=FAKE_QUANT)
def fake_quant_entry(
    model: torch.nn.Module,
    configs_mapping,
    mode: Mode = Mode.QUANTIZE,
    run_fn=None,
    run_args: Any = None,
    **kwargs,
) -> torch.nn.Module:
    """Replace modules with fake-quant subclasses and optionally calibrate.

    Args:
        model: Float model, modified in place.
        configs_mapping: ``{(module_name, op_type): ModuleQuantConfig}`` from
            :meth:`FakeQuantConfig.to_config_mapping`.
        mode: ``PREPARE`` inserts quantizers only; ``QUANTIZE`` also calibrates
            static quantizers when ``run_fn`` is given; ``CONVERT`` is a no-op
            because fake-quant has no real-quant conversion step.
        run_fn: Calibration forward loop, called as ``run_fn(model, *run_args)``.
        run_args: Positional arguments for ``run_fn``.
    """
    if mode == Mode.CONVERT:
        logger.info("[fake_quant] convert is a no-op; export() produces deployment artifacts")
        return model

    plan = {name: config for (name, _op_type), config in configs_mapping.items()}
    report = PlanReport(
        plan=plan,
        op_types={name: op_type for (name, op_type) in configs_mapping},
    )
    pending = _pending_static(plan)
    if mode != Mode.PREPARE and pending and run_fn is None:
        head = ", ".join(pending[:5])
        raise ValueError(
            f"{len(pending)} static quantizers have no amax and no run_fn was given: {head} ...; "
            "pass run_fn/run_args, an amax summary, or use prepare()+calibrate()"
        )
    result = apply_plan(model, report)
    logger.info(f"[fake_quant] replaced {result['n_replaced']} modules with quantized subclasses")

    if mode == Mode.PREPARE:
        _set_calibrating(model, True)
        if pending:
            logger.info(f"[fake_quant] {len(pending)} static quantizers await calibration")
        return model

    if pending:
        logger.info(f"[fake_quant] calibrating {len(pending)} static quantizers with run_fn")
        calibrate(model, run_fn, run_args or ())
    for module in model.modules():
        if isinstance(module, TensorQuantizer):
            module.finalize()
    return model


__all__ = ["fake_quant_entry"]
