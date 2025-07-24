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

import os
import shutil

import torch

from ._quant_common.quant_config import HpDtype
from ._core.quant_dequant import QuantDequantBase
from ._core.scale_handler import update_state_dict_method, ScaleFormat
from ._core.quantized_func_wrappers import (
    init_quantized_func_wrapper_factory,
    clear_quantized_func_wrapper_factory,
    get_quantized_func_wrapper,
    OP_TYPE,
)
from .prepare_quant.prepare_model import get_world_size, get_local_rank
from .utils.logger import logger
from neural_compressor.common import options
from neural_compressor.torch.utils import (
    get_accelerator,
    is_optimum_habana_available,
    get_attr,
    set_attr,
    SaveLoadFormat,
    get_enum_from_format,
    UNIT_MAPPING,
    write_json_file,
)


MAX_FILE_SIZE = 5  # GB
cur_accelerator = get_accelerator()
tp_module_list = (
    # raw module
    "LinearLayer",
    "LinearLayerAllReduce",
    "LmHeadLinearAllreduce",
    "RowParallelLinear",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    # patched module
    "PatchedLinear",
    "PatchedLinearAllReduce",
    "PatchedLmHeadLinearAllreduce",
    "PatchedColumnParallelLinear",
    "PatchedRowParallelLinear",
)


##################################### save ##################################
def remove_rank_suffix(name, local_rank, world_size):
    """Remove rank suffix from key name."""
    return name.removesuffix(f"_{local_rank}_{world_size}")


def add_rank_suffix(name, local_rank, world_size):
    """Add rank suffix to name."""
    return f"{name}_{local_rank}_{world_size}"


def save_rank_model(model, folder_prefix="", **kwargs):
    """Save state_dict for model from each rank."""
    # workaround for [SW-199005] [HQT] casted fp8 tensor cannot get data pointer
    cur_accelerator.synchronize()
    save_directory = add_rank_suffix(folder_prefix, get_local_rank(), get_world_size())
    os.makedirs(save_directory, exist_ok=True)
    safe_serialization = kwargs.get("safe_serialization", True)
    max_shard_size = kwargs.get("max_shard_size", f"{MAX_FILE_SIZE}GB")
    # save model state_dict and config.json
    model.save_pretrained(save_directory, max_shard_size=max_shard_size, safe_serialization=safe_serialization)


def find_tp_mod_list(model):
    """Get the module list that uses tensor parallel from model."""
    tp_mod_list = []
    for n, m in model.named_modules():
        if m.__class__.__name__ in tp_module_list:
            tp_mod_list.append(n)
    return tp_mod_list


def gather_state_dict(folder_prefix, file_name, tp_mod_list=[]):
    """Gather state_dict from files saved by each rank."""
    from safetensors.torch import load_file as safe_load_file

    world_size = get_world_size()

    def _is_in_list(name, tp_mod_list):
        for tp_name in tp_mod_list:
            if tp_name in name:
                return True
        return False

    gathered_state_dict = {}
    cur_state_dict = {}
    # load state_dict
    for i in range(world_size):  # TODO: assuming tp_size == world_size
        folder_name = add_rank_suffix(folder_prefix, i, world_size)
        cur_file = os.path.join(folder_name, file_name)
        cur_state_dict[i] = safe_load_file(cur_file)
    # gather state_dict
    for k, v in cur_state_dict[0].items():
        if _is_in_list(k, tp_mod_list):
            for i in range(world_size):  # TODO: assuming tp_size == world_size
                new_k = add_rank_suffix(k, i, world_size)
                gathered_state_dict[new_k] = cur_state_dict[i][k]
        else:
            gathered_state_dict[k] = cur_state_dict[0][k]
    return gathered_state_dict


def clean_rank_files(folder_prefix, file_name=None):
    """Clean files saved by each rank after gathering."""
    world_size = get_world_size()
    for i in range(world_size):  # TODO: assuming tp_size == world_size
        folder_name = add_rank_suffix(folder_prefix, i, world_size)
        if file_name is None:
            shutil.rmtree(folder_name, ignore_errors=True)
        else:
            cur_file = os.path.join(folder_name, file_name)
            os.remove(cur_file)


def parse_shard_size(size_str):
    """
    Parse a human-readable size string (e.g., '5GB', '10MB') into bytes as an integer.
    Args:
        size_str (str): The size string to parse, including a number and unit (e.g., '5GB').
    Returns:
        int: The size in bytes.
    Raises:
        ValueError: If the format of the input string is invalid.
    """
    size_str = size_str.strip().upper()

    for unit in UNIT_MAPPING:
        if size_str.endswith(unit):
            number = float(size_str[:-len(unit)].strip())
            return int(number * UNIT_MAPPING[unit])

    raise ValueError(f"Invalid size format: {size_str}")


def save_state_dict_sharded_safetensors(state_dict, save_dir, max_shard_size="5GB"):
    """Save the state_dict into safetensors files. If the model fits within max_shard_size, save as a single file.
    Otherwise, split into shards.
    Args:
        state_dict (dict): The model's state_dict.
        save_dir (str): The directory to save the files.
        max_shard_size (string): The maximum size for each file in bytes, default is 5GB.
    """
    from safetensors.torch import save_file

    os.makedirs(save_dir, exist_ok=True)

    total_model_size = sum(tensor.element_size() * tensor.numel() for tensor in state_dict.values())
    max_shard_size = parse_shard_size(max_shard_size)
    if total_model_size <= max_shard_size:
        # Save as a single file if the total size fits within max_shard_size
        single_file_path = os.path.join(save_dir, "model.safetensors")
        save_file(state_dict, single_file_path)
    else:
        # Save as multiple shards if the total size exceeds max_shard_size
        current_file_size = 0
        shard = {}
        shard_id = 1
        file_list = []

        for key, tensor in state_dict.items():
            tensor_size = tensor.element_size() * tensor.numel()

            if current_file_size + tensor_size > max_shard_size and shard:
                shard_file = os.path.join(save_dir, f"model-{shard_id:05d}-of-00000.safetensors")
                save_file(shard, shard_file)
                file_list.append(shard_file)
                shard = {}
                shard_id += 1
                current_file_size = 0

            shard[key] = tensor
            current_file_size += tensor_size

        if shard:
            shard_file = os.path.join(save_dir, f"model-{shard_id:05d}-of-00000.safetensors")
            save_file(shard, shard_file)
            file_list.append(shard_file)

        num_shards = len(file_list)
        for _, file_path in enumerate(file_list):
            new_file_name = file_path.replace("00000", f"{num_shards:05d}")
            os.rename(file_path, new_file_name)


def convert_weight_to_vllm_compatible(state_dict):
    """To convert INC fp8 model weight format to the vllm compatible format quantized by llm-compressor.
    Args:
        state_dict (dict): state_dict from INC model.
    Returns:
        The modified state_dict includes weight and scale adapted to llm-compressor.
    """
    keys_to_remove = []
    keys_to_add = {}

    for key in list(state_dict.keys()):
        if "scale_weight" in key:
            weight_scale = key.replace("scale_weight", "weight_scale")
            keys_to_add[weight_scale] = state_dict[key].contiguous().to("cpu")
            keys_to_remove.append(key)
        elif "k_cache.dequant_output.scale" in key:
            kv_scale = key.replace("k_cache.dequant_output.scale", "kv_scale")
            v_scale = key.replace("k_cache", "v_cache")
            k_scale_inv = key.replace("k_cache.dequant_output.scale", "k_cache.quant_input.scale_inv")
            v_scale_inv = key.replace("k_cache.dequant_output.scale", "v_cache.quant_input.scale_inv")

            keys_to_add[kv_scale] = max(state_dict[key], state_dict[v_scale]).contiguous().to("cpu")
            keys_to_remove.extend([key, v_scale, k_scale_inv, v_scale_inv])
        elif "scale_input" in key:
            input_scale = key.replace("scale_input", "input_scale")
            scale_input_inv = key.replace("scale_input", "quant_input.scale_inv")

            keys_to_add[input_scale] = state_dict[key].contiguous().to("cpu")
            keys_to_remove.extend([key, scale_input_inv])
        elif "proj.weight" in key:
            state_dict[key] = state_dict[key].t().contiguous().to("cpu")

    # Remove unnecessary keys
    for key in keys_to_remove:
        state_dict.pop(key, None)

    # Add new keys
    state_dict.update(keys_to_add)

    return state_dict


def generate_scheme(
    dynamic=False, observer="minmax", observer_kwargs={}, strategy="tensor", symmetric=True, type="float"
):
    """To generate the scheme requested by vllm compatible config."""
    return {
        "actorder": None,
        "block_structure": None,
        "dynamic": dynamic,
        "group_size": None,
        "num_bits": 8,
        "observer": observer,
        "observer_kwargs": observer_kwargs,
        "strategy": strategy,
        "symmetric": symmetric,
        "type": type,
    }


def convert_config_to_vllm_compatible(config):
    """Convert INC quantization_config to vllm compatible config.
    Args:
        config(dict): inc quantization config dictionary.
    """
    config_groups = {
        "group_0": {
            "input_activations": generate_scheme(
                dynamic=False, observer="minmax", observer_kwargs={}, strategy="tensor", symmetric=True, type="float"
            ),
            "output_activations": None,
            "targets": ["Linear"],
            "weights": generate_scheme(
                dynamic=False, observer="minmax", observer_kwargs={}, strategy="tensor", symmetric=True, type="float"
            ),
        }
    }
    ignore = [] if "names" not in config.blocklist else config.blocklist["names"]
    kv_cache_scheme = generate_scheme(
        dynamic=False, observer="minmax", observer_kwargs={}, strategy="tensor", symmetric=True, type="float"
    )
    quantization_config = {
        "config_groups": config_groups,
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "ignore": ignore,
        "kv_cache_scheme": kv_cache_scheme,
    }
    return quantization_config


def check_config_for_vllm_compatible(config):
    """
    check config parameters for saving vllm compatible format.
    """
    error_message = (
        "Save format='vllm' only supports per-tensor static quantization. The allowlist only supports "
        "Linear-related types (e.g., LinearReduce, LinearLayer) and KVCache-related types (e.g., VLLMKVCache)."
    )
    assert "_PCS_" not in config.scale_method.upper(), error_message
    assert "types" in config.allowlist, error_message
    types = config.allowlist["types"]
    assert all("kvcache" in t.lower() or "linear" in t.lower() for t in types), error_message


def save_for_multi_devices(model, checkpoint_dir="saved_results", format="huggingface", **kwargs):
    """Save FP8 model for multi-devices.
    Args:
        model (torch.nn.Module): fp8 model object.
        checkpoint_dir (str, optional): path to checkpoint. Defaults to "saved_results".
        format (str, optional): defaults to 'huggingface'.
    """
    from safetensors.torch import save_file as safe_save_file
    folder_prefix = os.path.join(options.workspace, checkpoint_dir)
    save_rank_model(model, folder_prefix=folder_prefix, **kwargs)
    # Ensure all ranks have saved their model before proceeding
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    rank_directory = add_rank_suffix(folder_prefix, 0, world_size)
    files_list = find_safetensors_files(rank_directory)
    # use rank:0 process to gather checkpoint files
    if local_rank == 0:
        tp_mod_list = find_tp_mod_list(model)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # get the safetensors file list from one folder
        # based on the safetensors file name to collect tensors from shard folders
        for file_name in files_list:
            gathered_state_dict = gather_state_dict(
                folder_prefix=folder_prefix, file_name=file_name, tp_mod_list=tp_mod_list
            )
            if format == SaveLoadFormat.VLLM:
                gathered_state_dict = update_to_vllm_compatible(model, gathered_state_dict)
            safe_save_file(gathered_state_dict, os.path.join(checkpoint_dir, file_name), metadata={"format": "pt"})
            clean_rank_files(folder_prefix=folder_prefix, file_name=file_name)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    clean_rank_files(folder_prefix=folder_prefix)


def save_for_single_device(model, checkpoint_dir="saved_results", format="huggingface", **kwargs):
    """Save FP8 model for single device.

    Args:
        model (torch.nn.Module): fp8 model object.
        checkpoint_dir (str, optional): path to checkpoint. Defaults to "saved_results".
        format (str, optional): defaults to 'huggingface'.
    """
    from huggingface_hub import save_torch_model

    os.makedirs(checkpoint_dir, exist_ok=True)
    # TODO: [SW-207766] RuntimeError: "fill_empty_deterministic_" not implemented for 'Float8_e4m3fn'
    tmp_deterministic_algorithms_flag = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    # TODO: [SW-199005] [HQT] casted fp8 tensor cannot get data pointer
    cur_accelerator.synchronize()
    if format == SaveLoadFormat.VLLM:
        state_dict = convert_weight_to_vllm_compatible(state_dict=model.state_dict())
        save_state_dict_sharded_safetensors(state_dict, checkpoint_dir, max_shard_size=f"{MAX_FILE_SIZE}GB")
    else:
        safe_serialization = kwargs.get("safe_serialization", True)
        max_shard_size = kwargs.get("max_shard_size", f"{MAX_FILE_SIZE}GB")
        # save model state_dict and config.json
        model.save_pretrained(checkpoint_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
        torch.use_deterministic_algorithms(tmp_deterministic_algorithms_flag)


def save(model, checkpoint_dir="saved_results", format="huggingface", **kwargs):
    """Save FP8 model.

    Args:
        model (torch.nn.Module): fp8 model object.
        checkpoint_dir (str, optional): path to checkpoint. Defaults to "saved_results".
        format (str, optional): defaults to 'huggingface'.
    """
    world_size = get_world_size()
    local_rank = get_local_rank()
    format = get_enum_from_format(format)
    model = process_model_for_scalar_scale(model)
    if world_size > 1:
        assert torch.distributed.is_initialized(), "Distributed environment is not initialized."
    if format == SaveLoadFormat.VLLM:
        check_config_for_vllm_compatible(model.qconfig[next(iter(model.qconfig))])
    if world_size > 0:
        save_for_multi_devices(model, checkpoint_dir=checkpoint_dir, format=format, **kwargs)
    else:
        save_for_single_device(model, checkpoint_dir=checkpoint_dir, format=format, **kwargs)

    # save "quantization_config" in config.json
    if local_rank in [0, -1]:
        # Ensure those codes run on a single rank.
        configs_mapping = model.qconfig
        config_object = configs_mapping[next(iter(configs_mapping))]
        config_object.mode = "LOAD"
        config_object.world_size = world_size  # record world_size for loading
        # Flux pipeline has FrozenDict as config
        if not isinstance(model.config, dict):
            update_model_config(model, format, config_object)
            model.config.save_pretrained(checkpoint_dir)
        write_json_file(os.path.join(checkpoint_dir, "quantization_config.json"), config_object.to_dict())

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(checkpoint_dir)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


##################################### load ##################################

def load_empty_raw_model(model_name_or_path, **kwargs):
    """Initialize BF16 model with meta tensor."""
    import transformers
    from accelerate import init_empty_weights

    # Handling model objects not in AutoModelForCausalLM
    model = kwargs.get("original_model", None)
    # Handle Flux pipeline without AutoConfig
    try:
        config = transformers.AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        quantization_config = config.quantization_config if hasattr(config, "quantization_config") else None
        hp_dtype = config.torch_dtype
    except:
        config, hp_dtype = model.config, torch.bfloat16
        quantization_config = kwargs.get("quantization_config", None)
        setattr(model.config, "quantization_config", quantization_config)

    if quantization_config is not None and "hp_dtype" in quantization_config:
        hp_dtype = HpDtype[quantization_config["hp_dtype"].upper()].value

    # fp8 model provided by neuralmagic.
    if (
        "quant_method" in quantization_config
        and quantization_config["quant_method"] in ["fp8", "compressed-tensors"]
    ):
        from_neuralmagic = True
        if (
            "kv_cache_scheme" in quantization_config
            and quantization_config["kv_cache_scheme"] is not None
        ):
            from_neuralmagic_with_kv = True
        else:
            from_neuralmagic_with_kv = False
    else:
        from_neuralmagic = False
        from_neuralmagic_with_kv = False

    if from_neuralmagic_with_kv:
        config.flash_attention_fp8 = True
        if is_optimum_habana_available:
            from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
            adapt_transformers_to_gaudi()
        else:
            raise ValueError("Please install optimum-habana to load fp8 kv cache model.")

    if model is None:
        with init_empty_weights(include_buffers=False):
            model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=hp_dtype)
    world_size = get_world_size()
    if world_size > 1:
        import deepspeed
        from neural_compressor.torch.utils import get_non_persistent_buffers, load_non_persistent_buffers

        # TODO: [SW-199728] [DeepSpeed] Buffers initialized by model are not correct after tensor parallel
        # get_non_persistent_buffers and load_non_persistent_buffers are workarounds of [SW-199728]
        non_persistent_buffers = get_non_persistent_buffers(model)
        ds_inference_kwargs = {
            "dtype": hp_dtype,
            "tensor_parallel": {"tp_size": world_size},
        }
        model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = model.module
        load_non_persistent_buffers(model, non_persistent_buffers)
    model.to(hp_dtype)

    try:
        generation_config = transformers.GenerationConfig.from_pretrained(model_name_or_path, **kwargs)
        model.generation_config = generation_config
    except:  # Since model.generation_config is optional, relaxed exceptions can handle more situations.
        logger.warning("model.generation_config is not loaded correctly.")
    return model, from_neuralmagic, from_neuralmagic_with_kv


def find_safetensors_files(model_name_or_path, **kwargs):
    """Get safetensors file list."""
    is_local = os.path.isdir(model_name_or_path)
    if is_local:
        safetensors_files = []
        for root, dirs, files in os.walk(model_name_or_path):
            for file in files:
                if file.endswith(".safetensors"):
                    safetensors_files.append(file)
        return safetensors_files
    else:
        from transformers.modeling_utils import get_checkpoint_shard_files
        from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file

        # download online files
        resolved_archive_file = cached_file(
            model_name_or_path, SAFE_WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
        )
        if resolved_archive_file is None:
            # state_dict is sharded.
            resolved_archive_file = cached_file(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
            # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                model_name_or_path,
                resolved_archive_file,
                **kwargs,
            )
        # for the model only with 1 model.safetensors file.
        if isinstance(resolved_archive_file, str):
            resolved_archive_file = [resolved_archive_file]
        return resolved_archive_file


def shard_state_dict(state_dict, return_all_rank=False, src_world_size=None):
    """Shard state_dict for current local_rank."""
    def get_rank_state_dict(state_dict, local_rank, world_size):
        rank_state_dict = {}
        for k, v in state_dict.items():
            if k.endswith(f"_{world_size}") and not k.endswith(f"_{local_rank}_{world_size}"):
                # only collect current rank state_dict and common state_dict (e.g., embedding)
                continue
            new_k = remove_rank_suffix(k, local_rank, world_size)
            rank_state_dict[new_k] = v
        return rank_state_dict

    if return_all_rank:
        all_rank_state_dict = {}
        for rank in range(src_world_size):
            rank_state_dict = get_rank_state_dict(state_dict, rank, src_world_size)
            all_rank_state_dict[rank] = rank_state_dict
        return all_rank_state_dict
    else:
        return get_rank_state_dict(state_dict, local_rank, world_size)


def split_rank_state_dict(model, gathered_state_dict):
    """Split state_dict for current local_rank."""
    rank_state_dict = {}
    for name, param in model.named_parameters():
        if name in gathered_state_dict:
            full_weight = gathered_state_dict[name]
            if len(param.shape) != 0 and full_weight.shape != param.shape:
                # clone to release reference to the original tensor
                if full_weight.shape[0] != param.shape[0]:
                    split_weight = split_weights(full_weight, world_size, local_rank, split_axis=0).clone()
                elif full_weight.shape[1] != param.shape[1]:
                    split_weight = split_weights(full_weight, world_size, local_rank, split_axis=1).clone()
                else:
                    split_weight = split_weights(full_weight, world_size, local_rank, split_axis=0).clone()
            else:
                split_weight = full_weight
            rank_state_dict[name] = split_weight

    return rank_state_dict


def get_inc_fp8config(model, from_neuralmagic=False, from_neuralmagic_with_kv=False):
    """Get INC FP8 Config.

    Args:
        model: empty model.
        from_neuralmagic(bool, optional): whether provided from nerualmagic modelhub.
        from_neuralmagic_with_kv(bool, optional): whether provided from nerualmagic modelhub and quantized kv_cache.

    Returns:
        INC FP8 Config.
    """
    from neural_compressor.torch.quantization import FP8Config
    if from_neuralmagic:
        if "ignore" in model.config.quantization_config.keys():
            blocklist =  {"types": [], "names": model.config.quantization_config["ignore"]}
        elif "ignored_layers" in model.config.quantization_config.keys():
            blocklist =  {"types": [], "names": model.config.quantization_config["ignored_layers"]}
        else:
            blocklist =  {"types": [], "names": ["lm_head"]}
        if "target" in model.config.quantization_config.keys():
            allowlist = {"types": model.config.quantization_config["target"], "names": []}
        else:
            if from_neuralmagic_with_kv:
                allowlist = {"types": ["Linear", "LinearLayer", "LinearAllreduce", "KVCache"], "names": []}
            else:
                allowlist = {"types": ["Linear", "LinearLayer", "LinearAllreduce"], "names": []}
        qconfig = FP8Config(mode="LOAD", allowlist=allowlist, blocklist=blocklist, scale_format="CONST")
    else:
        if hasattr(model, "qconfig") and model.qconfig is not None:
            configs_mapping = model.qconfig
            qconfig = configs_mapping[next(iter(configs_mapping))]
        else:
            qconfig = FP8Config.from_dict(model.config.quantization_config)
    return qconfig


def load(model_name_or_path, format="huggingface", device="hpu", **kwargs):
    """Load FP8 model.

    Args:
        model_name_or_path (str): model name or path.
        format (str, optional): format to load. Defaults to 'huggingface'.
        device (str, optional): device to use. Defaults to 'hpu'.
        kwargs (dict, optional): remaining dictionary of keyword arguments for loading huggingface models.

    Returns:
        FP8 model.
    """
    format = get_enum_from_format(format)
    world_size = kwargs.get("world_size", get_world_size())
    assert format == SaveLoadFormat.HUGGINGFACE, "Currently, only huggingface models are supported."
    assert device in ["hpu", "cpu"], "Currently, only hpu & cpu device is supported for FP8 model."

    from safetensors.torch import load_file as safe_load_file
    from neural_compressor.torch.algorithms.fp8_quant import prep_model

    model, from_neuralmagic, from_neuralmagic_with_kv = load_empty_raw_model(model_name_or_path, **kwargs)
    qconfig = get_inc_fp8config(model, from_neuralmagic, from_neuralmagic_with_kv)
    qconfig.save_temp_json_file()  # generate qconfig.json_file
    src_world_size = qconfig.world_size if hasattr(qconfig, "world_size") else world_size
    need_unify_weights = True if src_world_size > world_size else False
    if world_size > 1:
        assert torch.distributed.is_initialized(), "Distributed environment is not initialized."

    init_quantized_func_wrapper_factory()
    # replace modules to patched modules
    prep_model(model, qconfig.json_file)
    model = process_model_for_scalar_scale(model)
    # get the safetensors file list from one folder
    files_list = find_safetensors_files(model_name_or_path, **kwargs)  # TODO: get online files
    # based on the safetensors file name to collect tensors from shard folders
    for file_name in files_list:
        cur_file = os.path.join(model_name_or_path, file_name)
        gathered_state_dict = safe_load_file(cur_file)
        if from_neuralmagic or from_neuralmagic_with_kv:
            gathered_state_dict = convert_weight_to_inc(state_dict=gathered_state_dict)
        if src_world_size > 1:
            # only return state_dict for the current local_rank
            if from_neuralmagic or from_neuralmagic_with_kv:
                rank_state_dict = split_rank_state_dict(model, gathered_state_dict)
            else:
                rank_state_dict = shard_state_dict(gathered_state_dict, return_all_rank=need_unify_weights, src_world_size=src_world_size)
                if need_unify_weights:
                    # Assuming: src_world_size % world_size == 0
                    if src_world_size % world_size != 0:
                        raise EnvironmentError(
                            f"Original world_size: {src_world_size} must be divisible by target world_size: {world_size}."
                        )
                    rank_state_dict = get_new_rank_state_dict(rank_state_dict, model, world_size)
        else:
            rank_state_dict = gathered_state_dict
        model.load_state_dict(rank_state_dict, assign=True, strict=False)
        load_scale_params(model, rank_state_dict)  # ensure per-channel scale is loaded correctly
    clear_quantized_func_wrapper_factory()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model = model.to(cur_accelerator.name())
    model = model.eval()
    cur_accelerator.synchronize()
    return model


def convert_weight_to_inc(state_dict):
    """To convert the vllm compatible fp8 model weight to INC format,
       one is operators' name are different, the other is to adapt weight on G2
       due to the torch.float8_e4m3fn scope [-240, 240].

    Args:
        state_dict (dict): state_dict from modelhub.

    Returns:
        state_dict includes weight and scale adapted to INC format.
    """
    import habana_frameworks.torch.utils.experimental as htexp
    on_gaudi2 = htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi2
    key_name = state_dict.keys()
    for key in list(key_name):
        if "weight_scale" in key:
            scale_weight = key.replace("weight_scale", "scale_weight")
            if on_gaudi2:
                # dequant_weight
                weight_key = key.replace("weight_scale", "weight")
                qweight = state_dict[weight_key].t().to(torch.bfloat16).to("hpu")
                scale = state_dict[key].to("hpu")
                dequant_weight = qweight * scale
                # recompute scale, qweight
                recompute_scale = scale * (torch.finfo(torch.float8_e4m3fn).max /
                                        torch.finfo(torch.float8_e4m3fnuz).max)
                qweight = torch.ops.hpu.cast_to_fp8_v2(dequant_weight, 1.0 / recompute_scale, False, False, torch.float8_e4m3fn)[0]
                state_dict[weight_key] = qweight
                state_dict[scale_weight] = recompute_scale
            else:
                state_dict[scale_weight] = state_dict[key].to("hpu")
            state_dict.pop(key)
        elif "kv_scale" in key:
            k_scale_inv = key.replace("kv_scale", "k_cache.quant_input.scale_inv")
            v_scale_inv = key.replace("kv_scale", "v_cache.quant_input.scale_inv")
            k_scale = key.replace("kv_scale", "k_cache.dequant_output.scale")
            v_scale = key.replace("kv_scale", "v_cache.dequant_output.scale")
            state_dict[k_scale_inv] = 1 / state_dict[key].to("hpu")
            state_dict[v_scale_inv] = 1 / state_dict[key].to("hpu")
            state_dict[k_scale] = state_dict[key].to("hpu")
            state_dict[v_scale] = state_dict[key].to("hpu")
            state_dict.pop(key)
        elif "input_scale" in key:
            scale_input_inv = key.replace("input_scale", "quant_input.scale_inv")
            scale_input = key.replace("input_scale", "scale_input")
            state_dict[scale_input_inv] = 1 / state_dict[key].to("hpu")
            state_dict[scale_input] = state_dict[key].to("hpu")
            state_dict.pop(key)
        elif "proj.weight" in key and not on_gaudi2:
            state_dict[key] = state_dict[key].detach().t().to("hpu")
        else:
            pass
        cur_accelerator.synchronize()
    return state_dict


def split_weights(weight, tp_size, tp_rank, split_axis=0):
    """
    Args:
        weight (torch.Tensor): weight tensor.
        tp_size (int): tensor parallel size.
        tp_rank (int): tensor parallel rank.
        split_axis (int): split by column or line, 0 or 1.
    Returns:
        torch.Tensor: split weight tensor.
    """
    split_size = weight.shape[split_axis] // tp_size
    start_idx = tp_rank * split_size
    end_idx = (tp_rank + 1) * split_size

    if len(weight.shape) == 1:
        return weight[start_idx:end_idx]
    elif split_axis == 0:
        return weight[start_idx:end_idx, :]
    elif split_axis == 1:
        return weight[:, start_idx:end_idx]
    else:
        raise ValueError("split_axis must be 0 (row) or 1 (column).")


def process_model_for_scalar_scale(model):
    from neural_compressor.torch.algorithms.fp8_quant.patched_module_base import PatchedModuleBase
    from neural_compressor.torch.algorithms.fp8_quant._core.quant_dequant import QuantDequantBase
    for mod in model.modules():
        if isinstance(mod, (PatchedModuleBase, QuantDequantBase)):
            update_state_dict_method(mod)
    return model


def update_model_config(model, format, config_object):
    """Update model config for saving."""
    # update config for tie_word_embeddings
    if model.config.tie_word_embeddings:
        from transformers.modeling_utils import _get_tied_weight_keys
        embedding_weight = model.get_input_embeddings().weight
        for key in _get_tied_weight_keys(model):
            if model.state_dict()[key].data_ptr() != embedding_weight.data_ptr():
                model.config.tie_word_embeddings = False
    # update per format
    if format == SaveLoadFormat.VLLM:
        quantization_config = convert_config_to_vllm_compatible(config_object)
        model.config.quantization_config = quantization_config
    else:
        model.config.quantization_config = config_object


def load_scale_params(model, new_scale_params):
    """
    Load scale parameters into the model. If the lengths do not match, replace the old scales with the new ones.
    
    Args:
        model (torch.nn.Module): The model to load scale parameters into.
        new_scale_params (dict): A dictionary containing the new scale parameters.
    """
    for name, param in model.named_parameters():
        if name in new_scale_params:
            new_scale = new_scale_params[name]
            old_scale = param.data
            # update per-tensor scale with per-channel scale
            if old_scale.shape != new_scale.shape:
                param.data = new_scale


def get_new_rank_state_dict(all_rank_state_dict, model, world_size=get_world_size(), local_rank=get_local_rank()):
    """Get new rank state_dict for world_size.

    Args:
        all_rank_state_dict (dict): {0: state_dict, 1: state_dict, ...} for all ranks.
        model (torch.nn.Module): A quantized model with empty weights in load mode.
        world_size (int, optional): Target world size. Defaults to world_size.
        local_rank (int, optional): Current local rank. Defaults to local_rank.
    """

    def dq_q_weight(weight, previous_scale, target_scale):
        """dequantize and quantize weight with different scales."""
        cast_to_op = get_quantized_func_wrapper(OP_TYPE.CAST_TO_FP8, ScaleFormat.CONST)
        cast_from_op = get_quantized_func_wrapper(OP_TYPE.CAST_FROM_FP8, ScaleFormat.CONST)
        hp_weight = cast_from_op(weight, previous_scale, torch.bfloat16)
        # cast_to_op should use scale_inv = 1/ scale
        fp8_weight = cast_to_op(hp_weight, 1 / target_scale, False, False, dtype=weight.dtype)
        return fp8_weight

    def reorder_dict(d):
        """Reorder dict by moving weights to the end so that scales are process first."""
        from collections import OrderedDict
        without_weight = OrderedDict((k, v) for k, v in d.items() if not k.endswith('.weight'))
        with_weight = OrderedDict((k, v) for k, v in d.items() if k.endswith('.weight'))
        reordered_dict = OrderedDict()
        reordered_dict.update(without_weight)
        reordered_dict.update(with_weight)
        return reordered_dict

    def get_src_ranks(src_world_size, dst_world_size, local_rank):
        """Get src_ranks for current local_rank. 
            e.g.
                If src_world_size=8, dst_world_size=2:
                    1. local_rank=0, src_ranks=[0, 1, 2, 3]
                    2. local_rank=1, src_ranks=[4, 5, 6, 7]
        """
        shard_size = src_world_size // dst_world_size
        rank_combination = [[] for _ in range(dst_world_size)]
        for i in range(src_world_size):
            dst_rank =  i // shard_size
            rank_combination[dst_rank].append(i)
        src_ranks = rank_combination[local_rank]
        logger.info(f"local_rank={local_rank}, src_ranks={src_ranks}")
        return src_ranks

    def adjust_local_rank_and_world_size(world_size, local_rank):
        """Adjust world_size and local_rank for single card."""
        if world_size in [0, -1]:
            world_size = 1
        if local_rank == -1:
            local_rank = 0
        return world_size, local_rank

    def check_tp_status(target_weight_name, new_rank_state_dict, params_dict, scale):
        """Check is_tp, is_per_channel, is_tp_output_dim for tensor paralleled weight."""
        assert target_weight_name in params_dict, f"Weight {target_weight_name} not found in state_dict."
        target_weight_shape = params_dict[target_weight_name].shape
        cur_weight_shape = new_rank_state_dict[target_weight_name].shape
        is_tp = True if target_weight_shape != cur_weight_shape else False
        is_per_channel = True if scale.numel() != 1 else False
        # output dim of PatchedModule is 1, LinearAllReduce is sharding input dim
        is_tp_output_dim = True if target_weight_shape[1] != cur_weight_shape[1] else False
        return is_tp, is_per_channel, is_tp_output_dim

    def update_corresponding_weight(previous_state_dict, cur_state_dict, target_weight_name, new_scale, previous_scale, cur_scale):
        """Update corresponding weight based on new_scale, previous_scale, cur_scale."""
        device = cur_accelerator.name()
        if (new_scale != previous_scale).any():
            # update previous weight
            previous_state_dict[target_weight_name] = dq_q_weight(previous_state_dict[target_weight_name], previous_scale, new_scale)
        if (new_scale != cur_scale).any():
            # update new weight
            cur_state_dict[target_weight_name] = cur_state_dict[target_weight_name].to(device)
            cur_state_dict[target_weight_name] = dq_q_weight(cur_state_dict[target_weight_name], cur_scale, new_scale)


    # Ensure world_size and local_rank are meaningful even in single card
    world_size, local_rank = adjust_local_rank_and_world_size(world_size, local_rank)
    # get source ranks for current local_rank, source ranks are used to merge state_dict
    src_ranks = get_src_ranks(len(all_rank_state_dict), world_size, local_rank)

    # merge state_dict for current local_rank
    params_dict = dict(model.named_parameters())
    device = cur_accelerator.name()
    new_rank_state_dict = {}
    # merge state_dict for current local_rank
    for rank in src_ranks:
        state_dict = all_rank_state_dict[rank]
        # reorder dict for post-process weight with all-reduced scale
        state_dict = reorder_dict(state_dict)
        for k, v in state_dict.items():
            v = v.to(device)
            if k not in new_rank_state_dict:
                new_rank_state_dict[k] = v
            else:
                module_name, param_name = k.rsplit('.', 1)
                module = get_attr(model, module_name)
                # process scales and weights based on is_per_channel, is_tp, is_tp_output_dim
                if hasattr(module, "scale_members") and param_name in module.scale_members:
                    previous_scale = new_rank_state_dict[k]
                    cur_scale = v
                    if param_name == "scale_weight":
                        target_weight_name = module_name + ".weight"
                        is_tp, is_per_channel, is_tp_output_dim = check_tp_status(
                            target_weight_name, new_rank_state_dict, params_dict, cur_scale
                        )
                    # select correct way to update scales and weights
                    if param_name == "scale_weight" and is_tp and is_per_channel and is_tp_output_dim:
                        # only cat per-channel scales and weight
                        new_rank_state_dict[k] = torch.cat([new_rank_state_dict[k], v], dim=0)
                    else:
                        # select correct scale with min/max
                        if isinstance(module, QuantDequantBase) and param_name == "scale_inv":
                            new_rank_state_dict[k] = torch.min(new_rank_state_dict[k], v)
                        else:
                            new_rank_state_dict[k] = torch.max(new_rank_state_dict[k], v)
                        # update corresponding weight based on new_scale, previous_scale, cur_scale
                        if (previous_scale != cur_scale).any() and param_name == "scale_weight":
                            new_scale = new_rank_state_dict[k]
                            update_corresponding_weight(
                                new_rank_state_dict, state_dict, target_weight_name, new_scale, previous_scale, cur_scale
                            )
                elif k in params_dict and params_dict[k].shape != new_rank_state_dict[k].shape:
                    # Detect tp dim by comparing current weight shape with the one in model, it's more robust.
                    # e.g. LinearAllReduce(input dim) and LinearLayer(output dim) are using different dims.
                    tp_dim = 0 if params_dict[k].shape[0] != new_rank_state_dict[k].shape[0] else 1
                    new_rank_state_dict[k] = torch.cat([new_rank_state_dict[k], v], dim=tp_dim)
    return new_rank_state_dict


def update_to_vllm_compatible(model, gathered_state_dict):
    """update gathered_state_dict to vllm compatible format.

    Args:
        model (model): the quantized model.
        gathered_state_dict (dict): state_dict for all ranks.
    Returns:
        gathered_state_dict (dict): vllm compatible state_dict.
    """
    # for example, tp = 2, make
    # 'model.layers.21.mlp.down_proj.weight_0_2', 'model.layers.21.mlp.down_proj.weight_1_2'
    #  --> 'model.layers.21.mlp.down_proj.weight'
    import transformers
    from accelerate import init_empty_weights
    from neural_compressor.torch.algorithms.fp8_quant import prep_model

    with init_empty_weights(include_buffers=False):
        reference_model = transformers.AutoModelForCausalLM.from_config(model.config)
    # replace modules with patched modules
    inc_config = get_inc_fp8config(model)
    inc_config.mode = "LOAD"
    inc_config.save_temp_json_file()
    prep_model(reference_model, inc_config.json_file)
    # gather weights into 1 rank
    rank_state_dict = shard_state_dict(gathered_state_dict, return_all_rank=True, src_world_size=world_size)
    rank_state_dict = get_new_rank_state_dict(rank_state_dict, reference_model, world_size=1)
    # rename param names
    gathered_state_dict = convert_weight_to_vllm_compatible(state_dict=rank_state_dict)
    return gathered_state_dict
