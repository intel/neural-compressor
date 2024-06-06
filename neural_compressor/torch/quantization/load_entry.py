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
from neural_compressor.torch.utils import LoadFormat

config_name_mapping = {
    FP8_QUANT: FP8Config,
}


def load(model=None, checkpoint_dir="./saved_results", format="default", *hf_model_args, **hf_model_kwargs):
    """Load quantized model.

    1. Load INC quantized model in local.
    2. Load HuggingFace quantized model, including GPTQ/AWQ models and upstreamed INC quantized models in HF model hub.

    Args:
        model (Union[torch.nn.Module], str): torch model or hugginface model_name_or_path.
            if 'format' is set to 'huggingface', it means the huggingface model_name_or_path.
            if 'format' is set to 'default', it means the fp32 model and the 'checkpoint_dir'
            parameter should not be None. it coworks with 'checkpoint_dir' parameter to load INC
            quantized model in local.
        checkpoint_dir (str, optional): local path where quantized weights or model are saved.
            Only needed if 'format' is set to 'default'.
        format (str, optional): 'defult' for loading INC quantized model.
            'huggingface' for loading huggingface WOQ causal language model. Defaults to "default".

    Returns:
        torch.nn.Module: quantized model
    """
    if format == LoadFormat.DEFAULT.value:
        from neural_compressor.common.base_config import ConfigRegistry

        qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(checkpoint_dir)), "qconfig.json")
        with open(qconfig_file_path, "r") as f:
            per_op_qconfig = json.load(f)

        if " " in per_op_qconfig.keys():  # ipex qconfig format: {' ': {'q_op_infos': {'0': {'op_type': ...
            from neural_compressor.torch.algorithms.static_quant import load

            return load(checkpoint_dir)
        else:
            config_mapping = load_config_mapping(qconfig_file_path, ConfigRegistry.get_all_configs()["torch"])
            # select load function
            config_object = config_mapping[next(iter(config_mapping))]

            if isinstance(config_object, (RTNConfig, GPTQConfig, AWQConfig, TEQConfig, AutoRoundConfig)):  # WOQ
                from neural_compressor.torch.algorithms.weight_only.save_load import load

                return load(model=model, checkpoint_dir=checkpoint_dir, format=LoadFormat.DEFAULT)

            model.qconfig = config_mapping
            if isinstance(config_object, FP8Config):  # FP8
                from neural_compressor.torch.algorithms.habana_fp8 import load

                return load(model, checkpoint_dir)
    elif format == LoadFormat.HUGGINGFACE.value:
        # now only support load huggingface WOQ causal language model
        from neural_compressor.torch.algorithms.weight_only.save_load import load

        return load(model=model, format=LoadFormat.HUGGINGFACE, *hf_model_args, **hf_model_kwargs)
    else:
        raise ValueError("`format` in load function can only be 'huggingface' or 'default', but get {}".format(format))
