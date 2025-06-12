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

import torch
from auto_round.data_type.nvfp import full_quant
from neural_compressor.torch.utils import set_module
from neural_compressor.torch.experimental.fp4.modules import NVFP4Linear

NVFP4_MODULE_MAPPING = {
    torch.nn.Linear: NVFP4Linear,
}


def qdq_model(model, dtype="nvfp4"):
    """
    Quantize and dequantize the weights of a model using NVFP4.

    Args:
        model (torch.nn.Module): The model whose weights are to be quantized and dequantized.
    """
    assert dtype=="nvfp4", "NVFP4 quantization is only supported for nvfp4 dtype."
    for name, module in model.named_modules():
        if isinstance(module, tuple(NVFP4_MODULE_MAPPING.keys())):
            qdq_module = NVFP4_MODULE_MAPPING[module.__class__](module)
            set_module(model, name, qdq_module)
    return model

def qdq_module_weights(module):
    """
    Quantize and dequantize the weights of a module using NVFP4.

    Args:
        module (torch.nn.Module): The module whose weights are to be quantized and dequantized.
    """
    if hasattr(module, 'weight'):
        # Quantize the weight
        qdq_weight = full_quant(module.weight)
        module.weight.copy_(qdq_weight)
        # Dequantize the weight
        module.weight.data = module.weight_qdq.dequantize()


def qdq_module_forward(module):



