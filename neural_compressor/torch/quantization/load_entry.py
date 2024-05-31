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

import json
import os

from neural_compressor.common.utils import FP8_QUANT  # unified namespace
from neural_compressor.common.utils import load_config_mapping  # unified namespace
from neural_compressor.torch.quantization.config import (
    AutoRoundConfig,
    AWQConfig,
    FP8Config,
    GPTQConfig,
    RTNConfig,
    TEQConfig,
)

config_name_mapping = {
    FP8_QUANT: FP8Config,
}


def load(model_name_or_path="./saved_results", model=None, format="default", *hf_model_args, **hf_model_kwargs):
    """Load quantized model.

    Args:
        model_name_or_path (str, optional): local path where quantized weights or model are saved
            or huggingface model id. Defaults to "./saved_results".
        model (torch.nn.Module, optional): original model. Require to pass when loading INC WOQ quantized model
            or loading FP8 model. Defaults to None.
        format (str, optional): 'defult' for loading INC quantized model.
            'huggingface' now only for loading huggingface WOQ causal language model. Defaults to "default".

    Returns:
        torch.nn.Module: quantized model
    """
    if format == "default":
        from neural_compressor.common.base_config import ConfigRegistry
        from neural_compressor.torch.algorithms.habana_fp8 import load as habana_fp8_load
        from neural_compressor.torch.algorithms.static_quant import load as static_quant_load
        from neural_compressor.torch.algorithms.weight_only.save_load import load as woq_load

        qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(model_name_or_path)), "qconfig.json")
        with open(qconfig_file_path, "r") as f:
            per_op_qconfig = json.load(f)

        if " " in per_op_qconfig.keys():  # ipex qconfig format: {' ': {'q_op_infos': {'0': {'op_type': ...
            return static_quant_load(model_name_or_path)
        else:
            config_mapping = load_config_mapping(qconfig_file_path, ConfigRegistry.get_all_configs()["torch"])
            # select load function
            config_object = config_mapping[next(iter(config_mapping))]

            if isinstance(config_object, (RTNConfig, GPTQConfig, AWQConfig, TEQConfig, AutoRoundConfig)):  # WOQ
                return woq_load(model_name_or_path, model=model, format=format)

            model.qconfig = config_mapping
            if isinstance(config_object, FP8Config):  # FP8
                return habana_fp8_load(model, model_name_or_path)
    elif format == "huggingface":
        # now only support load huggingface WOQ causal language model
        from neural_compressor.torch.algorithms.weight_only.save_load import load as woq_load

        return woq_load(model_name_or_path, format=format, *hf_model_args, **hf_model_kwargs)
    else:
        raise ValueError("`format` in load function can only be 'huggingface' or 'default', but get {}".format(format))
