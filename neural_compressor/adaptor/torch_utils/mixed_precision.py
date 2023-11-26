#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from neural_compressor.utils.utility import LazyImport

ipex = LazyImport("intel_extension_for_pytorch")
torch = LazyImport("torch")


def ipex_mixed_precision(model, example_inputs=None, device="cpu"):
    """The function is used for ipex mixed precision quantization."""
    model.eval()
    if device == "xpu":
        autocast = torch.xpu.amp.autocast
    else:
        autocast = torch.cpu.amp.autocast
    mp_model = model._model
    if example_inputs is None:
        assert False, "please provide the correct example_inputs for torchscript mode"
    mp_model = ipex.optimize(mp_model, dtype=torch.bfloat16)

    with torch.no_grad(), autocast():
        if isinstance(example_inputs, dict):
            try:
                mp_model = torch.jit.trace(mp_model, example_kwarg_inputs=example_inputs)
            except:
                mp_model = torch.jit.trace(
                    mp_model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False
                )
        else:
            try:
                mp_model = torch.jit.trace(mp_model, example_inputs)
            except:
                mp_model = torch.jit.trace(mp_model, example_inputs, strict=False)
        mp_model = torch.jit.freeze(mp_model.eval())
    model._model = mp_model
    return model
