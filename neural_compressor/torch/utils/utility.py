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
"""Intel Neural Compressor PyTorch utilities."""


import enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import psutil
import torch
from typing_extensions import TypeAlias

from neural_compressor.common.utils import (
    Mode,
    ProcessorType,
    Statistics,
    cpu_info,
    detect_processor_type_based_on_hw,
    logger,
)

OP_NAME_AND_TYPE_TUPLE_TYPE: TypeAlias = Tuple[str, Union[torch.nn.Module, Callable]]

# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}


# All constants for torch
WHITE_MODULE_LIST = [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]

HPU_SAFE_WEIGHTS_NAME = "hpu_model.safetensors"
WEIGHT_NAME = "quantized_weight.pt"
HPU_WEIGHT_NAME = "quantized_hpu_weight.pt"
QCONFIG_NAME = "qconfig.json"


def register_algo(name):
    """Decorator function to register algorithms in the algos_mapping dictionary.

    Usage example:
        @register_algo(name=example_algo)
        def example_algo(model: torch.nn.Module, quant_config: RTNConfig) -> torch.nn.Module:
            ...

    Args:
        name (str): The name under which the algorithm function will be registered.

    Returns:
        decorator: The decorator function to be used with algorithm functions.
    """

    def decorator(algo_func):
        algos_mapping[name] = algo_func
        return algo_func

    return decorator


def fetch_module(model, op_name):
    """Get module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.

    Returns:
        module (object).
    """
    module = model
    name_list = op_name.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            logger.warning(f"The {op_name} is not present in the model.")
            return None
    return module


def set_module(model, op_name, new_module):
    """Set module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.
        new_module (object): the input model.

    Returns:
        module (object).
    """
    name_list = op_name.split(".")
    if len(name_list) == 1:
        setattr(model, name_list[-1], new_module)
        return
    else:
        second_last_module = fetch_module(model, ".".join(name_list[:-1]))
        if second_last_module is None:
            logger.warning(f"Setting skipped as the {op_name} is not present in the model.")
            return None
        else:
            setattr(second_last_module, name_list[-1], new_module)


get_attr = fetch_module
set_attr = set_module


def get_model_info(model: torch.nn.Module, white_module_list: List[Callable]) -> List[Tuple[str, str]]:
    """Get model info according to white_module_list."""
    module_dict = dict(model.named_modules())
    filter_result = []
    filter_result_set = set()
    for op_name, module in module_dict.items():
        if isinstance(module, tuple(white_module_list)):
            pair = (op_name, type(module).__name__)
            if pair not in filter_result_set:
                filter_result_set.add(pair)
                filter_result.append(pair)
    logger.debug(f"Get model info: {filter_result}")
    return filter_result


def get_double_quant_config_dict(double_quant_type="BNB_NF4"):
    """Query config dict of double_quant according to double_quant_type.

    Args:
        double_quant_type (str, optional): double_quant type. Defaults to "BNB_NF4".
    """
    from neural_compressor.torch.utils.constants import DOUBLE_QUANT_CONFIGS

    assert double_quant_type in DOUBLE_QUANT_CONFIGS, "Supported double quant configs: {}".format(
        list(DOUBLE_QUANT_CONFIGS.keys())
    )
    return DOUBLE_QUANT_CONFIGS[double_quant_type]


def get_quantizer(model, quantizer_cls, quant_config=None, *args, **kwargs):
    """Get the quantizer.

    Initialize a quantizer or get `quantizer` attribute from model.

    Args:
        model (torch.nn.Module): pytorch model.
        quantizer_cls (Quantizer): quantizer class of a specific algorithm.
        quant_config (dict, optional): Specifies how to apply the algorithm on the given model.
            Defaults to None.

    Returns:
        quantizer object.
    """
    if not hasattr(model, "quantizer"):
        quantizer = quantizer_cls(quant_config=quant_config, *args, **kwargs)
        return quantizer
    else:
        return model.quantizer


def postprocess_model(model, mode, quantizer):
    """Process `quantizer` attribute of model according to current phase.

    In `prepare` phase, the `quantizer` is set as an attribute of the model
    to avoid redundant initialization during `convert` phase.

    In 'convert' or 'quantize' phase, the unused `quantizer` attribute is removed.

    Args:
        model (torch.nn.Module): pytorch model.
        mode (Mode): The mode of current phase, including 'prepare', 'convert' and 'quantize'.
        quantizer (Quantizer): quantizer object.
    """
    if mode == Mode.PREPARE:
        model.quantizer = quantizer
    elif mode == Mode.CONVERT or mode == Mode.QUANTIZE:
        if getattr(model, "quantizer", False):
            del model.quantizer


def dump_model_op_stats(mode, tune_cfg):
    """Dump quantizable ops stats of model to user.

    Args:
        mode (object): quantization mode.
        tune_cfg (dict): quantization config
    """
    if mode == Mode.PREPARE:
        return
    res = {}
    # collect all dtype info and build empty results with existing op_type
    dtype_set = set()
    for op, config in tune_cfg.items():
        op_type = op[1]
        config = config.to_dict()
        if not config["dtype"] == "fp32":
            num_bits = config["bits"]
            group_size = config["group_size"]
            dtype_str = "A32W{}G{}".format(num_bits, group_size)
            dtype_set.add(dtype_str)
    dtype_set.add("FP32")
    dtype_list = list(dtype_set)
    dtype_list.sort()

    for op, config in tune_cfg.items():
        config = config.to_dict()
        op_type = op[1]
        if op_type not in res.keys():
            res[op_type] = {dtype: 0 for dtype in dtype_list}

    # fill in results with op_type and dtype
    for op, config in tune_cfg.items():
        config = config.to_dict()
        if config["dtype"] == "fp32":
            res[op_type]["FP32"] += 1
        else:
            num_bits = config["bits"]
            group_size = config["group_size"]
            dtype_str = "A32W{}G{}".format(num_bits, group_size)
            res[op_type][dtype_str] += 1

    # update stats format for dump.
    field_names = ["Op Type", "Total"]
    field_names.extend(dtype_list)
    output_data = []
    for op_type in res.keys():
        field_results = [op_type, sum(res[op_type].values())]
        field_results.extend([res[op_type][dtype] for dtype in dtype_list])
        output_data.append(field_results)

    Statistics(output_data, header="Mixed Precision Statistics", field_names=field_names).print_stat()


def get_model_device(model: torch.nn.Module):
    """Get the device.

    Args:
        model (torch.nn.Module): the input model.

    Returns:
        device (str): a string.
    """
    for n, p in model.named_parameters():
        return p.data.device.type  # p.data.device == device(type='cpu')


def get_processor_type_from_user_config(user_processor_type: Optional[Union[str, ProcessorType]] = None):
    """Get the processor type.

    Get the processor type based on the user configuration or automatically detect it based on the hardware.

    Args:
        user_processor_type (Optional[Union[str, ProcessorType]]): The user-specified processor type. Defaults to None.

    Returns:
        ProcessorType: The detected or user-specified processor type.

    Raises:
        AssertionError: If the user-specified processor type is not supported.
        NotImplementedError: If the processor type is not recognized.
    """
    if user_processor_type is None:
        processor_type = detect_processor_type_based_on_hw()
    elif isinstance(user_processor_type, ProcessorType):
        processor_type = user_processor_type
    elif isinstance(user_processor_type, str):
        user_processor_type = user_processor_type.lower().capitalize()
        assert user_processor_type in ProcessorType.__members__, f"Unsupported processor type: {user_processor_type}"
        processor_type = ProcessorType(user_processor_type)
    else:
        raise NotImplementedError(f"Unsupported processor type: {user_processor_type}")
    return processor_type


def dowload_hf_model(repo_id, cache_dir=None, repo_type=None, revision=None):
    """Download hugging face model from hf hub."""
    import os

    from huggingface_hub.constants import DEFAULT_REVISION, HUGGINGFACE_HUB_CACHE
    from huggingface_hub.file_download import REGEX_COMMIT_HASH, repo_folder_name
    from huggingface_hub.utils import EntryNotFoundError

    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if revision is None:
        revision = DEFAULT_REVISION
    if repo_type is None:
        repo_type = "model"
    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    commit_hash = None
    if REGEX_COMMIT_HASH.match(revision):
        commit_hash = revision
    else:
        ref_path = os.path.join(storage_folder, "refs", revision)
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                commit_hash = f.read()
    if storage_folder and commit_hash:
        pointer_path = os.path.join(storage_folder, "snapshots", commit_hash)
        if os.path.isdir(pointer_path):
            return pointer_path
    else:  # pragma: no cover
        from huggingface_hub import snapshot_download

        file_path = snapshot_download(repo_id)
        return file_path


def load_empty_model(pretrained_model_name_or_path, cls=None, **kwargs):
    """Load a empty model."""
    import os

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    cls = AutoModelForCausalLM if cls is None else cls
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:  # pragma: no cover
        path = pretrained_model_name_or_path
    else:
        path = dowload_hf_model(pretrained_model_name_or_path)
    if cls.__base__ == _BaseAutoModelClass:
        config = AutoConfig.from_pretrained(path, **kwargs)
        with init_empty_weights():
            model = cls.from_config(config)
    else:  # pragma: no cover
        config = cls.config_class.from_pretrained(path, **kwargs)
        with init_empty_weights():
            model = cls(config)
    model.tie_weights()
    model.eval()
    model.path = pretrained_model_name_or_path
    return model
