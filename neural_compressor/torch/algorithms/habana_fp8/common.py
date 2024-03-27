import json
import tempfile
import os
from pathlib import Path
from typing import Union

def save_calib_result(model, fname):
    from habana_quantization_toolkit._hook_method import config

    update_stats_path_in_config(old_stats_path=config.cfg["dump_stats_path"], new_stats_path=fname)

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
    whitelist=set(config.cfg['mod_dict'].keys())
    blacklist=set()
    for type_st in config.cfg['blacklist']['types']:
        blacklist.add(type_st)
    whitelist.difference_update(blacklist)
    whitelist_tuple=tuple(whitelist)

    mod_list = []
    for name, mod in model.named_modules():
        mod_type=mod.__class__.__name__
        if (mod_type in whitelist_tuple) and \
        (_is_substr(config.cfg['whitelist']['names'], name) or \
        len(config.cfg['whitelist']['names'])==0) and \
        (not _is_substr(config.cfg['blacklist']['names'], name)):
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
