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
"""Intel Neural Compressor PyTorch load entry for all algorithms."""

import json
import os

from neural_compressor.common.utils import FP8_QUANT, Mode, load_config_mapping, log_process
from neural_compressor.torch.quantization.config import (
    AutoRoundConfig,
    AWQConfig,
    FP8Config,
    GPTQConfig,
    HQQConfig,
    RTNConfig,
    TEQConfig,
)
from neural_compressor.torch.utils import SaveLoadFormat

config_name_mapping = {
    FP8_QUANT: FP8Config,
}


def save(model, checkpoint_dir="saved_results", format="default"):
    """Save quantized model.

    Args:
        model (torch.nn.module or TorchScript model with IPEX or fx graph with pt2e, optional): Quantized model.
        checkpoint_dir (str, optional): checkpoint directory. Defaults to "saved_results".
        format (str, optional): 'defult' for loading INC quantized model.
            'huggingface' for loading huggingface WOQ causal language model. Defaults to "default".
    """
    config_mapping = model.qconfig
    config_object = config_mapping[next(iter(config_mapping))]
    # fp8_quant
    if isinstance(config_object, FP8Config):
        from neural_compressor.torch.algorithms import fp8_quant

        format = SaveLoadFormat.HUGGINGFACE.value  # TODO: support default format for FP8 algorithm
        fp8_quant.save(model, checkpoint_dir, format)
    else:
        assert format == SaveLoadFormat.DEFAULT.value, "Currently, only default format is supported."
        model.save(checkpoint_dir)  # TODO: support huggingface format for WOQ algorithms.


@log_process(mode=Mode.LOAD)
def load(model_name_or_path, original_model=None, format="default", device="cpu", **kwargs):
    """Load quantized model.

    1. Load INC quantized model in local.
        case 1: WOQ
            from neural_compressor.torch.quantization import load
            load(model_name_or_path="saved_results", original_model=fp32_model)

        case 2: INT8/FP8
            from neural_compressor.torch.quantization import load
            load(model_name_or_path='saved_result', original_model=fp32_model)

        case 3: TorchScript (IPEX)
            from neural_compressor.torch.quantization import load
            load(model_name_or_path='saved_result')

    2. Load HuggingFace quantized model, including GPTQ models and upstreamed INC quantized models in HF model hub.
        case 1: WOQ
            from neural_compressor.torch.quantization import load
            load(model_name_or_path=model_name_or_path, format="huggingface")

    Args:
        model_name_or_path (str):  torch checkpoint directory or hugginface model_name_or_path.
            If 'format' is set to 'huggingface', it means the huggingface model_name_or_path.
            If 'format' is set to 'default', it means the 'checkpoint_dir'.
            Parameter should not be None. it coworks with 'original_model' parameter to load INC
            quantized model in local.
        original_model (torch.nn.module or TorchScript model with IPEX or fx graph with pt2e, optional):
            original model before quantization. Needed if 'format' is set to 'default' and not TorchScript model.
            Defaults to None.
        format (str, optional): 'defult' for loading INC quantized model.
            'huggingface' for loading huggingface WOQ causal language model. Defaults to "default".
        device (str, optional): 'cpu', 'hpu'. specify the device the model will be loaded to.
            currently only used for weight-only quantization.
        kwargs (remaining dictionary of keyword arguments, optional):
            remaining dictionary of keyword arguments for loading huggingface models.
            Will be passed to the huggingface model's `__init__` method, such as 'trust_remote_code', 'revision'.

    Returns:
        The quantized model
    """
    if format == SaveLoadFormat.DEFAULT.value:
        from neural_compressor.common.base_config import ConfigRegistry

        qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(model_name_or_path)), "qconfig.json")
        with open(qconfig_file_path, "r") as f:
            per_op_qconfig = json.load(f)

        if " " in per_op_qconfig.keys():  # ipex qconfig format: {' ': {'q_op_infos': {'0': {'op_type': ...
            from neural_compressor.torch.algorithms import static_quant

            return static_quant.load(model_name_or_path)
        elif "static_quant" in per_op_qconfig.keys() or "pt2e_dynamic_quant" in per_op_qconfig.keys():  # PT2E
            from neural_compressor.torch.algorithms import pt2e_quant

            return pt2e_quant.load(model_name_or_path)
        else:
            config_mapping = load_config_mapping(qconfig_file_path, ConfigRegistry.get_all_configs()["torch"])
            # select load function
            config_object = config_mapping[next(iter(config_mapping))]

            if isinstance(
                config_object, (RTNConfig, GPTQConfig, AWQConfig, TEQConfig, AutoRoundConfig, HQQConfig)
            ):  # WOQ
                from neural_compressor.torch.algorithms import weight_only

                qmodel = weight_only.load(
                    model_name_or_path, original_model, format=SaveLoadFormat.DEFAULT, device=device
                )
                return qmodel.to(device)
    elif format == SaveLoadFormat.HUGGINGFACE.value:
        import transformers

        config = transformers.AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        # use config to check which algorithm is used.
        if (
            "fp8_config" in config.quantization_config
            or
            # for FP8 LLMs for vLLM (https://huggingface.co/neuralmagic).
            (
                "quant_method" in config.quantization_config
                and config.quantization_config["quant_method"] in ["fp8", "compressed-tensors"]
            )
        ):
            from neural_compressor.torch.algorithms import fp8_quant

            return fp8_quant.load(model_name_or_path, format=format, device=device, **kwargs)
        else:
            from neural_compressor.torch.algorithms import weight_only

            qmodel = weight_only.load(model_name_or_path, format=SaveLoadFormat.HUGGINGFACE, device=device, **kwargs)
            return qmodel.to(device)
    else:
        raise ValueError("`format` in load function can only be 'huggingface' or 'default', but get {}".format(format))
