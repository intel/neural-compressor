import torch
import torch.nn as nn
from os import environ
import os
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.core.quantization import _check_params_as_const, _mark_params_as_const
from .._quant_common.quant_config import get_hqt_config
from .._quant_common.helper_modules import quant_dequant
from .measure import load_measurements
from .scale import scale_method_mapping, get_config, scaling_methods
from .common import mod_default_dict, generate_model_info, parent_child_mod_dict

def patch_module(mod, qconfig, mod_dict):
    parent=parent_child_mod_dict[mod].parent
    name=parent_child_mod_dict[mod].name
    patched_mod=mod_dict[mod.__class__.__name__].patched_module(mod, qconfig)
    setattr(parent, name, patched_mod)

def apply_hf_hook(module):
    if hasattr(module, '_hf_hook'):
        module._hf_hook.pre_forward(module)
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")
    if hasattr(module, "_old_forward"):
        module.forward = module._old_forward
        delattr(module, "_old_forward")

def prepare_model(model, qconfig, mod_list, hp_dtype=torch.float):
    config = get_hqt_config(model)
    patched_modules = []
    patched_module_types = set()
    with (torch.no_grad()):
        for name, mod in model.named_modules():
            # When offloading weight to disk, need to transfer the weight from disk to cpu using hf_hook
            apply_hf_hook(mod)
            if name in mod_list:
                mod_config=qconfig[name]
                for param in mod_config.params:
                    param_config=mod_config.params[param]
                    p=getattr(mod, param)
                    pq=quant_dequant(p.to("hpu"), scale_quant_fcn=param_config.scale_quant_fcn, quant_fcn=param_config.quant_fcn, scale_dequant_fcn=param_config.scale_dequant_fcn, dequant_fcn=param_config.dequant_fcn)
                    delattr(mod, param)
                    setattr(mod, param, nn.Parameter(pq))
                    pq = getattr(mod, param)
                    pq.requires_grad_(False)
                    htcore.mark_step()
                patch_module(mod, mod_config, mod_default_dict)
                patched_modules.append(name)
                patched_module_types.add(type(mod))
    if config.cfg['verbose']:
        print("Patched module types: ", patched_module_types)
        print("Patched modules: ", patched_modules)
        print("Total patched modules: ", len(patched_modules))
    model = model.to("hpu")
    htcore.mark_step()

def quantize(model, mod_list):
    config = get_hqt_config(model)
    environ['USE_SCALE'] = '1' # TODO SW-166049 remove once tpc libs use scale by deafult
    generate_model_info(model)
    hp_dtype = config.cfg['hp_dtype']
    lp_dtype = config.cfg['fp8_config']
    measurement=load_measurements(model, config.cfg['measure_file'])
    # FIXME make sure this takes unit_scale or measured scale, from Configs
    scaling_method_name=scale_method_mapping[(config.cfg['scale_method'], config.cfg['observer'])]
    scaling_method = scaling_methods[scaling_method_name]
    params = config.cfg['scale_params']
    params['hp_dtype'] = hp_dtype
    params['lp_dtype'] = lp_dtype
    qconfig = get_config(model, measurement, mod_default_dict, scaling_method, params, config.cfg['scale_file'], False, mod_list)
    prepare_model(model, qconfig, mod_list, hp_dtype=hp_dtype)
