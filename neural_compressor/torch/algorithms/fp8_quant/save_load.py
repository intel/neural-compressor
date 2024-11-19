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

import torch

from ._quant_common.quant_config import local_rank, world_size
from neural_compressor.torch.utils import get_accelerator


MAX_FILE_SIZE = 5  # GB
cur_accelerator = get_accelerator()
tp_module_list = (
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
    from neural_compressor.torch.utils import get_non_persistent_buffers, load_non_persistent_buffers

    if world_size > 1:
        import deepspeed

        config = transformers.AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        with init_empty_weights(include_buffers=False):
            model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        # TODO: [SW-199728] [DeepSpeed] Buffers initialized by model are not correct after tensor parallel
        # get_non_persistent_buffers and load_non_persistent_buffers are workrounds of [SW-199728]
        non_persistent_buffers = get_non_persistent_buffers(model)
        ds_inference_kwargs = {
            "dtype": torch.bfloat16,
            "tensor_parallel": {"tp_size": world_size},
        }
        model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = model.module
        load_non_persistent_buffers(model, non_persistent_buffers)
    else:
        config = transformers.AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        with init_empty_weights(include_buffers=False):
            model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    return model


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

    model = load_empty_raw_model(model_name_or_path, **kwargs)
    from neural_compressor.torch.algorithms.fp8_quant import prep_model
    from neural_compressor.torch.quantization import FP8Config

    qconfig = FP8Config.from_dict(model.config.quantization_config)
    qconfig.save_temp_json_file()  # generate qconfig.json_file
    # replace modules to patched modules
    prep_model(model, qconfig.json_file)
    # get the safetensors file list from one folder
    files_list = find_safetensors_files(model_name_or_path, **kwargs)  # TODO: get online files
    # based on the safetensors file name to collect tensors from shard folders
    for file_name in files_list:
        cur_file = os.path.join(model_name_or_path, file_name)
        gathered_state_dict = safe_load_file(cur_file)
        if world_size > 0:
            # only return state_dict for the current local_rank
            rank_state_dict = shard_state_dict(gathered_state_dict)
            model.load_state_dict(rank_state_dict, assign=True, strict=False)
        else:
            model.load_state_dict(gathered_state_dict, assign=True, strict=False)
    model = model.eval()
    model = model.to(cur_accelerator.name())
    cur_accelerator.synchronize()
    # make sure cpu and hpu memory are all released.
    gc.collect()
    return model
