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
import importlib
from collections import UserDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import psutil
import torch
import torch.nn as nn
from typing_extensions import TypeAlias

from neural_compressor.common.utils import (
    Mode,
    ProcessorType,
    Statistics,
    cpu_info,
    detect_processor_type_based_on_hw,
    logger,
)
from neural_compressor.torch.utils import is_optimum_habana_available, is_transformers_imported

if is_transformers_imported():
    import transformers

    SUPPORTED_LAYERS = [nn.Linear, transformers.modeling_utils.Conv1D]
else:
    SUPPORTED_LAYERS = [nn.Conv1d, nn.Linear]

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
        if os.path.isdir(pointer_path) and any(
            file.endswith(".bin") or file.endswith(".safetensors") for file in os.listdir(pointer_path)
        ):
            return pointer_path
    from huggingface_hub import list_repo_files, snapshot_download

    files_info = list_repo_files(repo_id)
    ignore_patterns = (
        ["*.bin", "*.bin.index.json"]
        if (
            any(file for file in files_info if file.endswith(".bin"))
            and any(file for file in files_info if file.endswith(".safetensors"))
        )
        else None
    )

    file_path = snapshot_download(repo_id, ignore_patterns=ignore_patterns)
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
        with init_empty_weights():
            model = cls.from_pretrained(path, **kwargs)
    else:  # pragma: no cover
        config = cls.config_class.from_pretrained(path, **kwargs)
        with init_empty_weights():
            model = cls(config, **kwargs)
    model.tie_weights()
    model.eval()
    model.path = pretrained_model_name_or_path
    return model


def get_module(module, key):
    """Get module from model by key name.

    Args:
        module (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    name_list = key.split(".")
    for name in name_list:
        module = getattr(module, name, None)
    return module


def get_layer_names_in_block(model, supported_types=SUPPORTED_LAYERS, to_quant_block_names=None):
    """Retrieves the names of layers within each block of the model.

    Returns:
        list: A list of strings, where each string is the name of a layer
              within a block of the model.
    """
    for n, m in model.named_modules():
        if isinstance(m, tuple(supported_types)):
            m.tmp_name = n
    layers_in_block = []
    if bool(to_quant_block_names):
        all_blocks = to_quant_block_names
    else:
        all_blocks = get_block_names(model)
    for block_names in all_blocks:
        for block_name in block_names:
            block = get_module(model, block_name)
            for n, m in block.named_modules():
                if hasattr(m, "tmp_name"):
                    layers_in_block.append(m.tmp_name)
    for n, m in model.named_modules():
        if hasattr(m, "tmp_name"):
            delattr(m, "tmp_name")
    return layers_in_block


def to_dtype(input, dtype=torch.float32):  # pragma: no cover
    """Moves input data to the specified data type.

    Args:
    input: The input data to be moved.
    dtype: The target data type.

    Returns:
    The input data on the specified data type.
    """
    if input is None:
        return None
    if isinstance(input, torch.Tensor):
        return input.to(dtype)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = to_dtype(input[inp], dtype)

    elif isinstance(input, list) or isinstance(input, tuple):
        if len(input) == 0:
            return input
        input_res = []
        for inp in input:
            input_res.append(to_dtype(inp, dtype))
        if isinstance(input, tuple):
            input_res = tuple(input_res)
        input = input_res

    return input


# for VLM usage
def to_device(input, device=torch.device("cpu")):  # pragma: no cover
    """Moves input data to the specified device.

    Args:
    input: The input data to be moved.
    device: The target device.

    Returns:
    The input data on the specified device.
    """
    if input is None:
        return None
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = to_device(input[inp], device)
    elif isinstance(input, list) or isinstance(input, tuple):
        if len(input) == 0:
            return input
        input_res = []
        for inp in input:
            input_res.append(to_device(inp, device))
        if isinstance(input, tuple):
            input_res = tuple(input_res)
        input = input_res

    return input


def get_block_names(model):
    """Get the block names for transformers-like networks.

    Args:
    model: The model.

    Returns:
    block_names: A list whose elements are list of block's layer names
    """
    block_names = []
    target_modules = []
    for n, m in model.named_modules():
        if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
            target_modules.append((n, m))
            break  ## only find the first modulelist, may be not robust
    for i, target_m in enumerate(target_modules):
        block_names.append([])
        for n, m in target_m[1].named_children():
            block_names[i].append(target_m[0] + "." + n)
    return block_names


def validate_modules(module_names):  # pragma: no cover
    """Test a list of modules' validity.

    Args:
    modules (list of str): List of strings to be validated.

    Returns:
    bool: True if all modules have equal length or not dependent, otherwise False.
    """
    if not bool(module_names):
        raise ValueError("Empty modules")
    if len(module_names) < 2:
        return True
    split_modules = [s.split(".") for s, _ in module_names]
    lengths = [len(parts) for parts in split_modules]
    if len(set(lengths)) == 1:
        return True
    max_length = max(lengths)
    min_length = min(lengths)
    longest_module = next(s for s in split_modules if len(s) == max_length)
    shortest_module = next(s for s in split_modules if len(s) == min_length)
    shortest_module = ".".join(shortest_module)
    longest_module = ".".join(longest_module)
    # Check if the shortest name is a substring of the longest name
    if shortest_module in longest_module:
        raise ValueError(
            "Invalid modules, at least two modules detected" " as dependent, {shortest_module} and {longest_module}"
        )
    return True


def get_multimodal_block_names(model, quant_vision=False):
    """Get the multimodal model block names for transformers-like networks.

    Args:
    model: The model.

    Returns:
    block_names: A list whose elements are list of block's layer names
    """
    block_names = []
    target_modules = []
    Vison_blocks_tuple = (
        "vision",
        "visual",
    )
    for n, m in model.named_modules():
        if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
            if quant_vision or all(key not in n.lower() for key in (Vison_blocks_tuple)):
                target_modules.append((n, m))
    validate_modules(target_modules)
    for i, target_m in enumerate(target_modules):
        block_names.append([])
        for n, m in target_m[1].named_children():
            block_names[i].append(target_m[0] + "." + n)
    return block_names


def detect_device(device=None):  # pragma: no cover
    """Detects the device to use for model execution (GPU, HPU, or CPU).

    Args:
        device (str, int, torch.device, optional):
            - If a string ('cuda', 'cpu', or 'hpu') or torch.device is provided, that device is selected.
            - If an integer is provided, it treats it as a GPU device index.
            - If None or 'auto', it automatically selects 'cuda' if available, 'hpu' if Habana is available,
              or falls back to 'cpu'.

    Returns:
        str: The selected device in string format ('cuda:X', 'hpu', or 'cpu').
    """

    def is_valid_digit(s):
        try:
            num = int(s)
            return 0 <= num
        except:
            return False

    dev_idx = None
    if is_valid_digit(device):
        dev_idx = int(device)
        device = "auto"
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU device")
        elif is_optimum_habana_available():
            device = torch.device("hpu")
            print("Using HPU device")
        # Use CPU as a fallback
        else:
            device = torch.device("cpu")
            print("Using CPU device")
        if dev_idx is not None and str(device) != "cpu":
            device = str(device) + f":{dev_idx}"
        return str(device)
    elif isinstance(device, torch.device):
        device = str(device)
    return device


def find_matching_blocks(model, all_blocks, to_quant_block_names=None):
    """Find and return matching blocks in the model based on to_quant_block_names.

    Args:
        model: The model (not used in this specific function but kept for completeness).
        all_blocks: List of lists, where each inner list contains full block names in the model.
        to_quant_block_names: Comma-separated string of target block names to match.

    Returns:
        target_blocks: List of lists containing full paths of matching blocks in the model.
    """
    import re

    if not to_quant_block_names:
        return all_blocks
    to_quant_block_list = to_quant_block_names
    if isinstance(to_quant_block_names, list) or isinstance(to_quant_block_names, tuple):
        return to_quant_block_names
    if isinstance(to_quant_block_names, str):
        to_quant_block_list = [name.strip() for name in to_quant_block_names.split(",")]
    target_blocks = []
    for block_list in all_blocks:
        matched_sublist = []
        for name in to_quant_block_list:
            matches = [block for block in block_list if re.search(name, block)]
            if matches:
                matched_sublist.extend(matches)
        if matched_sublist:
            target_blocks.append(matched_sublist)
        if not target_blocks:
            raise ValueError(
                "No block names matched. Please check the input for to_quant_block_name,"
                "or set to_quant_block_name to None to automatically match quantizable blocks."
            )
    return target_blocks


def get_non_persistent_buffers(model):
    """Get all non-persistent buffers in the model.

    Args:
        model (torch.nn.Module): PyTorch model

    Returns:
        dict: A dictionary containing all non-persistent buffers, {buffer_names: buffer_tensors}
    """
    non_persistent_buffers = {}
    for name, module in model.named_modules():
        # Check the module's non-persistent buffer
        for buffer_name, buffer in module._buffers.items():
            if buffer_name in module._non_persistent_buffers_set:
                non_persistent_buffers[(name, buffer_name)] = buffer
    return non_persistent_buffers


def load_non_persistent_buffers(model, non_persistent_buffers):
    """Load all non-persistent buffers into the model.

    Args:
        model (torch.nn.Module): PyTorch model
        non_persistent_buffers (dict): A dictionary containing all non-persistent buffers, {buffer_names: buffer_tensors}
    """
    for full_name, buffer in non_persistent_buffers.items():
        module_name, buffer_name = full_name
        module = model.get_submodule(module_name) if module_name else model
        setattr(module, buffer_name, buffer)


# copied from neural_compressor/adaptor/torch_utils/util.py
def move_input_device(input, device="cpu"):
    """Auto mapping input to device for all kinds of format.

    Args:
        input (torch.tensor): input data
        device (str, optional): target device. Defaults to "cpu".

    Returns:
        input (torch.tensor): input data on target device
    """
    if isinstance(input, dict) or isinstance(input, UserDict):
        tmp_input = {}
        for k, inp in input.items():
            tmp_input[k] = move_input_device(inp, device)
        input = tmp_input
    elif isinstance(input, list) or isinstance(input, tuple):
        tmp_input = []
        for inp in input:
            tmp_input.append(move_input_device(inp, device))
        input = tmp_input
    elif isinstance(input, torch.Tensor):
        input = input.to(device)  # pylint: disable=no-member
    return input


# copied from neural_compressor/adaptor/torch_utils/util.py
def forward_wrapper(model, input):
    """Model forward with device auto mapping.

    Args:
        model (torch.nn.Module): input model
        input (torch.tensor): input data

    Returns:
        output: output data
    """
    try:
        device = next(model.parameters()).device
    except:
        # for RecursiveScriptModule
        device = "cpu"
    input = move_input_device(input, device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        output = model(**input)
    elif isinstance(input, list) or isinstance(input, tuple):
        try:
            output = model(*input)
        except:
            output = model(input)
    else:
        output = model(input)
    return output
