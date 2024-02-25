# Copyright (c) 2024 Intel Corporation
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

# pylint:disable=import-error

import json
import os

import habana_frameworks.torch.core as htcore
import torch

from neural_compressor.torch.utils import logger, WEIGHT_NAME, QCONFIG_NAME
from neural_compressor.common.utils import save_qconfig, load_qconfig

from .fp8_quant import FP8_DTYPE, dtype_mapping
from .modules import (  # fp32; dynamic modules
    Autocast,
    BatchMatmul,
    FP8Cast,
    FP8DynamicBatchMatmul,
    FP8DynamicLinear,
    FP8DynamicMatmul,
    Matmul,
)


def save(model, output_dir="./saved_results"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), QCONFIG_NAME)
    # saving process
    save_qconfig(model.qconfig, qconfig_file_path)

    import fp8_convert

    stat_dict = {}
    for k, v in model.state_dict().items():
        if v.dtype in FP8_DTYPE:
            v = fp8_convert.to_u8(v.to("cpu"))
        stat_dict[k] = v.to("cpu")
    torch.save(stat_dict, qmodel_file_path)

    logger.info("Save state_dict of quantized model to {}.".format(qmodel_file_path))
    logger.info("Save configuration of quantized model to {}.".format(qconfig_file_path))


def load(model, output_dir="./saved_results"):
    from neural_compressor.torch.utils import fetch_module, set_module

    from .fp8_quant import quantization_mapping, white_list

    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    stat_dict = torch.load(qmodel_file_path)
    import fp8_convert
    
    for (op_name, op_type), op_qconfig in model.qconfig.items():
        dtype = op_qconfig.w_dtype
        choice = 1 if dtype == "fp8_e4m3" else 0
        if op_name + ".weight" in stat_dict:
            stat_dict[op_name + ".weight"] = fp8_convert.from_u8(stat_dict[op_name + ".weight"], choice)
        if dtype not in FP8_DTYPE:
            continue
        module = fetch_module(model, op_name)
        dtype = dtype_mapping[dtype]
        # replace module
        if op_qconfig.approach == "static":
            if isinstance(module, white_list):
                QModule = quantization_mapping[type(module)]
                module = QModule(module, dtype)
        else:
            if isinstance(module, torch.nn.Linear):
                # need module for initialization
                module = FP8DynamicLinear(module, dtype)
            elif isinstance(module, Matmul):
                module = FP8DynamicMatmul(dtype)
            elif isinstance(module, BatchMatmul):
                module = FP8DynamicBatchMatmul(dtype)
            elif isinstance(module, Autocast):
                module = FP8Cast(dtype=dtype)
        set_module(model, op_name, module)
        htcore.mark_step()
    model.load_state_dict(stat_dict, assign=True)
    model.to('hpu')
    htcore.mark_step()
    logger.info("Quantized model loading successful.")
    return model
