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

import copy
import json
from typing import Union

try:
    from neural_compressor.utils.utility import LazyImport

    torch = LazyImport("torch")
    from neural_compressor.utils import logger
except:  # pragma: no cover
    import logging

    import torch

    logger = logging.getLogger()


def export_compressed_model(
    model,
    weight_config: Union[str, dict],
    enable_full_range=False,
    compression_dtype=torch.int32,
    compression_dim=1,
    scale_dtype=torch.float32,
    device="cpu",
    use_optimum_format=True,
):
    """Convert Linear to WeightOnlyLinear for low memory inference.

    Args:
        weight_config (str|dict): qconfig dict or Path of qconfig.json.
        enable_full_range (bool, optional): Whether to leverage the full compression range
                                            under symmetric quantization. Defaults to False.
        compression_dtype (torch.Tensor, optional): The target dtype after comoression.
                                                    Defaults to torch.int32.
        compression_dim (int, optional): Select from [0, 1], 0 is output channel,
                                            1 is input channel. Defaults to 1.
        scale_dtype (torch.Tensor, optional): Use float32 or float16.
                                                Defaults to torch.float32.
        device (str, optional): choose device for compression. Defaults to cpu.
        use_optimum_format (bool, optional): use the popular huggingface compression format.
            1: compression_dim: weight = 1, zeros = 0 and both are transposed.
            2: zeros -= 1 before compression. Why we need it?
            3: g_idx: use same number for one group instead of recording the channel order.
            4. parameter name changed, such as 'packed_weight' -> 'qweight'.
            5. zeros is always needed even for sym.
    """
    from .autoround import get_module, quant_weight_w_scale, set_module
    from .model_wrapper import WeightOnlyLinear

    compressed_model = copy.deepcopy(model)
    if isinstance(weight_config, str):
        with open(weight_config, "r") as f:
            q_config = json.load(f)
    else:
        q_config = weight_config
    for k, v in q_config.items():
        logger.info(f"Compressing {k} on device {device}")
        if v["data_type"] == "float":
            continue
        else:
            dtype = v["data_type"]
            num_bits = v["bits"]
            group_size = v["group_size"]
            scheme = v["scheme"]
        m = get_module(compressed_model, k)
        fp_weight = m.weight.data
        scale = torch.tensor(v["scale"], dtype=torch.float32)  # may exist dtype dismatch problem
        zp = None if scheme == "sym" else torch.tensor(v["zp"], dtype=torch.int32)
        int_weight = quant_weight_w_scale(fp_weight, scale, zp, group_size)
        int_weight = int_weight.type(torch.int32)
        new_module = WeightOnlyLinear(
            m.in_features,
            m.out_features,
            num_bits,
            group_size,
            dtype=dtype,
            zp=zp is not None,
            bias=m.bias is not None,
            device=device,
            use_optimum_format=True,
        )
        new_module.pack(int_weight, scale, zp, m.bias)
        set_module(compressed_model, k, new_module)
    return compressed_model
