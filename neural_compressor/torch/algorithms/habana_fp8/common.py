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
import tempfile
import os
from pathlib import Path
from typing import Union
from collections import namedtuple
import torch

def save_calib_result(model):
    import habana_quantization_toolkit
    habana_quantization_toolkit.finish_measurements(model)

def save_fp8_model(model, fname):
    # TODO: save fp8 model
    pass


def update_stats_path_in_config(old_stats_path, new_stats_path):
    from habana_quantization_toolkit._hook_method import config

    new_base_name=new_stats_path.split('/')[-1]
    new_folder_name=new_stats_path[:-(len(new_base_name))]
    os.makedirs(new_folder_name, exist_ok=True)
    old_dump_stats_base_path = config.cfg["dump_stats_base_path"]
    config.cfg["dump_stats_base_path"] = config.cfg["dump_stats_base_path"].replace(old_dump_stats_base_path, new_folder_name)

    stats_path_related_keys = ["dump_stats_path", "dump_stats_base_path",
        "shape_file", "scale_file", "measure_file"]
    for key in stats_path_related_keys:
        config.cfg[key] = config.cfg[key].replace(old_stats_path, new_stats_path)

    # remove old dump_stats_path folder
    if old_dump_stats_base_path != new_folder_name:
        os.removedirs(old_dump_stats_base_path)

def update_mode(calib_step=False, quant_step=False):
    from habana_quantization_toolkit._hook_method import config
    from habana_quantization_toolkit._quant_common.quant_config import QuantMode

    if calib_step:
        config.cfg["mode"] = QuantMode.MEASURE
    if quant_step:
        config.cfg["mode"] = QuantMode.QUANTIZE


def get_mod_list(model):
    from habana_quantization_toolkit._hook_method import config

    _update_mod_dict(config)
    allowlist=set(config.cfg['mod_dict'].keys())
    blocklist=set()
    for type_st in config.cfg['blocklist']['types']:
        blocklist.add(type_st)
    allowlist.difference_update(blocklist)
    allowlist_tuple=tuple(allowlist)

    mod_list = []
    for name, mod in model.named_modules():
        mod_type=mod.__class__.__name__
        if (mod_type in allowlist_tuple) and \
        (_is_substr(config.cfg['allowlist']['names'], name) or \
        len(config.cfg['allowlist']['names'])==0) and \
        (not _is_substr(config.cfg['blocklist']['names'], name)):
            mod_list.append(name)
    if config.cfg['verbose']:
        print(f"Module list: {mod_list}")
    return mod_list

def _get_mod_default_dict():
    from collections import namedtuple
    from habana_quantization_toolkit._quant_common.helper_modules import (
        PatchedMatmul, PatchedLinear, PatchedKVCache, PatchedConv2d,
        PatchedLoRACompatibleLinear, PatchedLoRACompatibleConv, PatchedSoftmax
    )
    module_info = namedtuple('ModuleInfo', ['type', 'patched_module'])
    mod_default_dict= {
                    "Matmul": module_info('matmul', PatchedMatmul),
                    "Linear": module_info('linear', PatchedLinear),
                    "FalconLinear": module_info('linear', PatchedLinear),
                    "KVCache": module_info('kv_cache', PatchedKVCache),
                    "Conv2d": module_info('linear', PatchedConv2d),
                    "LoRACompatibleLinear": module_info('linear', PatchedLoRACompatibleLinear),
                    "LoRACompatibleConv": module_info('linear', PatchedLoRACompatibleConv),
                    "Softmax": module_info('softmax', PatchedSoftmax)
                    }
    return mod_default_dict

def _update_mod_dict(config):
    mod_default_dict = _get_mod_default_dict()
    config.cfg['mod_dict'].update({k: mod_default_dict[k].type for k in mod_default_dict})


def _is_substr(substr_list, target):
    return any([x in target for x in substr_list])


def generate_model_info(model):
    mod_inst_info = namedtuple('ModInstInfo', ['name', 'parent'])
    parent_child_mod_dict={}
    def create_mod_info_recursion(parent):
        for name, mod in parent.named_children():
            parent_child_mod_dict[mod]=mod_inst_info(name=name, parent=parent)
            create_mod_info_recursion(mod)
    create_mod_info_recursion(model)
    return parent_child_mod_dict

from .helper_modules import Linear
revert_module_dict = {"PatchedLinear": Linear}

def patch_module(parent_child_mod_dict, patched_mod):
    parent = parent_child_mod_dict[patched_mod].parent
    name = parent_child_mod_dict[patched_mod].name
    origin_mod = revert_module_dict[patched_mod.__class__.__name__](patched_mod)
    origin_mod.forward = patched_mod.forward_orig
    setattr(parent, name, origin_mod)

def restore_patched_module(patched_model):
    parent_child_mod_dict = generate_model_info(patched_model)
    with torch.no_grad():
        for name, patched_mod in patched_model.named_modules():
            patched_mod_type_str = patched_mod.__class__.__name__
            if patched_mod_type_str in revert_module_dict:
                patch_module(parent_child_mod_dict, patched_mod)

def with_patched_module(model):
    for name, mod in model.named_modules():
        mod_type=mod.__class__.__name__
        if mod_type in revert_module_dict.keys():
            return True
    return False
