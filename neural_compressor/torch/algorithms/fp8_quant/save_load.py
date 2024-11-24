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

import gc
import os
import shutil
import time
import copy

import torch

from ._quant_common.quant_config import local_rank, world_size, HpDtype
from neural_compressor.torch.utils import get_accelerator, is_optimum_habana_available, get_attr, set_attr


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

def save_rank_model(model, folder_prefix=""):
    """Save state_dict for model from each rank."""
    from huggingface_hub import save_torch_model

    save_directory = f"{folder_prefix}_{local_rank}_{world_size}"
    os.makedirs(save_directory, exist_ok=True)
    # workaround for [SW-199005] [HQT] casted fp8 tensor cannot get data pointer
    cur_accelerator.synchronize()
    save_torch_model(model, save_directory, max_shard_size=f"{MAX_FILE_SIZE/world_size}GB", metadata={"format": "pt"})


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

    def _is_in_list(name, tp_mod_list):
        for tp_name in tp_mod_list:
            if tp_name in name:
                return True
        return False

    gathered_state_dict = {}
    cur_state_dict = {}
    # load state_dict
    for i in range(world_size):  # TODO: assuming tp_size == world_size
        folder_name = f"{folder_prefix}_{i}_{world_size}"
        cur_file = os.path.join(folder_name, file_name)
        if not os.path.exists(cur_file):
            time.sleep(10)  # Waiting for other ranks to finish saving
        cur_state_dict[i] = safe_load_file(cur_file)
    # gather state_dict
    for k, v in cur_state_dict[0].items():
        if _is_in_list(k, tp_mod_list):
            for i in range(world_size):  # TODO: assuming tp_size == world_size
                new_k = f"{k}_{i}_{world_size}"
                gathered_state_dict[new_k] = cur_state_dict[i][k]
        else:
            gathered_state_dict[k] = cur_state_dict[0][k]
    return gathered_state_dict


def clean_rank_files(folder_prefix, file_name=None):
    """Clean files saved by each rank after gathering."""
    for i in range(world_size):  # TODO: assuming tp_size == world_size
        folder_name = f"{folder_prefix}_{i}_{world_size}"
        if file_name is None:
            shutil.rmtree(folder_name, ignore_errors=True)
        else:
            cur_file = os.path.join(folder_name, file_name)
            shutil.rmtree(cur_file, ignore_errors=True)


def save(model, checkpoint_dir="saved_results", format="huggingface"):
    """Save FP8 model.

    Args:
        model (torch.nn.Module): fp8 model object.
        checkpoint_dir (str, optional): path to checkpoint. Defaults to "saved_results".
        format (str, optional): _description_. Defaults to 'huggingface'.
    """
    assert format == "huggingface", (
        "Currently, only huggingface models are supported." + "Please set format='huggingface'."
    )
    from safetensors.torch import save_file as safe_save_file

    # cancel tied weights to avoid reloading issue when lm_head is not quantized.
    for key in model._tied_weights_keys:
        weight = get_attr(model, key)
        set_attr(model, key, copy.deepcopy(weight))

    if world_size > 0:
        save_rank_model(model, folder_prefix=checkpoint_dir)
        rank_directory = f"{checkpoint_dir}_{0}_{world_size}"
        files_list = find_safetensors_files(rank_directory)
        # use rank:0 process to gather checkpoint files
        if local_rank == 0:
            tp_mod_list = find_tp_mod_list(model)
            os.makedirs(checkpoint_dir, exist_ok=True)
            # get the safetensors file list from one folder
            # based on the safetensors file name to collect tensors from shard folders
            for file_name in files_list:
                gathered_state_dict = gather_state_dict(
                    folder_prefix=checkpoint_dir, file_name=file_name, tp_mod_list=tp_mod_list
                )
                safe_save_file(gathered_state_dict, os.path.join(checkpoint_dir, file_name), metadata={"format": "pt"})
                clean_rank_files(folder_prefix=checkpoint_dir, file_name=file_name)
            clean_rank_files(folder_prefix=checkpoint_dir)
    else:
        from huggingface_hub import save_torch_model

        os.makedirs(checkpoint_dir, exist_ok=True)
        # TODO: [SW-207766] RuntimeError: "fill_empty_deterministic_" not implemented for 'Float8_e4m3fn'
        tmp_deterministic_algorithms_flag = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        # TODO: [SW-199005] [HQT] casted fp8 tensor cannot get data pointer
        cur_accelerator.synchronize()
        save_torch_model(model, checkpoint_dir, max_shard_size=f"{MAX_FILE_SIZE}GB", metadata={"format": "pt"})
        torch.use_deterministic_algorithms(tmp_deterministic_algorithms_flag)

    # save "quantization_config" in config.json
    configs_mapping = model.qconfig
    config_object = configs_mapping[next(iter(configs_mapping))]
    config_object.mode = "LOAD"
    model.config.quantization_config = config_object
    model.config.save_pretrained(checkpoint_dir)


##################################### load ##################################


def load_empty_raw_model(model_name_or_path, **kwargs):
    """Initialize BF16 model with meta tensor."""
    import transformers
    from accelerate import init_empty_weights
    config = transformers.AutoConfig.from_pretrained(model_name_or_path, **kwargs)
    # fp8 model provided by neuralmagic.
    if (
        "quant_method" in config.quantization_config
        and config.quantization_config["quant_method"] in ["fp8", "compressed-tensors"]
    ):
        from_neuralmagic = True
        if (
            "kv_cache_scheme" in config.quantization_config
            and config.quantization_config["kv_cache_scheme"] is not None
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

    from neural_compressor.torch.utils import get_non_persistent_buffers, load_non_persistent_buffers

    hp_dtype = config.torch_dtype
    if hasattr(config, "quantization_config") and "hp_dtype" in config.quantization_config:
        hp_dtype = HpDtype[config.quantization_config["hp_dtype"].upper()].value
    if world_size > 1:
        import deepspeed

        with init_empty_weights(include_buffers=False):
            model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=hp_dtype)
        # TODO: [SW-199728] [DeepSpeed] Buffers initialized by model are not correct after tensor parallel
        # get_non_persistent_buffers and load_non_persistent_buffers are workrounds of [SW-199728]
        non_persistent_buffers = get_non_persistent_buffers(model)
        ds_inference_kwargs = {
            "dtype": hp_dtype,
            "tensor_parallel": {"tp_size": world_size},
        }
        model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = model.module
        load_non_persistent_buffers(model, non_persistent_buffers)
        model.to(hp_dtype)
    else:
        with init_empty_weights(include_buffers=False):
            model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=hp_dtype)
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


def shard_state_dict(state_dict):
    """Shard state_dict for current local_rank."""
    rank_state_dict = {}
    for k, v in state_dict.items():
        if k.endswith(f"_{local_rank}_{world_size}"):
            new_k = k.rstrip(f"_{local_rank}_{world_size}")
            rank_state_dict[new_k] = v.to("hpu")
        else:
            rank_state_dict[k] = v.to("hpu")
    return rank_state_dict

def split_rank_state_dict(model, gathered_state_dict):
    """split state_dict for current local_rank."""
    rank_state_dict = {}
    for name, param in model.named_parameters():
        if name in gathered_state_dict:
            full_weight = gathered_state_dict[name]
            if len(param.shape) != 0 and full_weight.shape != param.shape:
                if full_weight.shape[0] != param.shape[0]:
                    split_weight = split_weights(full_weight, world_size, local_rank, split_axis=0)
                elif full_weight.shape[1] != param.shape[1]:
                    split_weight = split_weights(full_weight, world_size, local_rank, split_axis=1)
                else:
                    split_weight = split_weights(full_weight, world_size, local_rank, split_axis=0)
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
    assert format == "huggingface", "Currently, only huggingface models are supported."
    assert device == "hpu", "Currently, only hpu device is supported for FP8 model."
    from safetensors.torch import load_file as safe_load_file

    from neural_compressor.torch.algorithms.fp8_quant import prep_model

    model, from_neuralmagic, from_neuralmagic_with_kv = load_empty_raw_model(model_name_or_path, **kwargs)
    qconfig = get_inc_fp8config(model, from_neuralmagic, from_neuralmagic_with_kv)
    qconfig.save_temp_json_file()  # generate qconfig.json_file

    # replace modules to patched modules
    prep_model(model, qconfig.json_file)
    # get the safetensors file list from one folder
    files_list = find_safetensors_files(model_name_or_path, **kwargs)  # TODO: get online files
    # based on the safetensors file name to collect tensors from shard folders
    for file_name in files_list:
        cur_file = os.path.join(model_name_or_path, file_name)
        gathered_state_dict = safe_load_file(cur_file)
        if from_neuralmagic or from_neuralmagic_with_kv:
            import habana_frameworks.torch.utils.experimental as htexp
            gathered_state_dict = convert_weight_to_inc(
                                        state_dict=gathered_state_dict,
                                        on_gaudi2=htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi2
                                        )
        if world_size > 0:
            # only return state_dict for the current local_rank
            if from_neuralmagic or from_neuralmagic_with_kv:
                rank_state_dict = split_rank_state_dict(model, gathered_state_dict)
            else:
                rank_state_dict = shard_state_dict(gathered_state_dict)
            model.load_state_dict(rank_state_dict, assign=True, strict=False)
        else:
            model.load_state_dict(gathered_state_dict, assign=True, strict=False)

    if from_neuralmagic or from_neuralmagic_with_kv:
        model.tie_weights()
    model = model.to(cur_accelerator.name())
    model = model.eval()
    cur_accelerator.synchronize()
    # make sure cpu and hpu memory are all released.
    gc.collect()
    return model


def convert_weight_to_inc(state_dict, on_gaudi2=False):
    """To convert the vllm compatable fp8 model weight to INC format,
       one is operators' name are different, the other is to adapt weight on G2
       due to the torch.float8_e4m3fn scope [-240, 240].

    Args:
        state_dict (dict): state_dict from modelhub.
        on_gaudi2 (bool, optional): whether is on Gaudi2. Defaults to False.

    Returns:
        state_dict includes weight and scale adapted to INC format.
    """
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
