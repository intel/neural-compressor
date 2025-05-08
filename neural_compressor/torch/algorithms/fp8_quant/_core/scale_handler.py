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

import torch
import types
from .._quant_common.quant_config import ScaleFormat
from .common import is_runtime_scale_patching


def add_scale_registry(patched_mod):
    """Update scale registry"""
    patched_mod.scale_members = set()
    patched_mod.register_scale = types.MethodType(register_scale, patched_mod)
    return patched_mod


def register_scale(patched_mod, name, scale, scale_format):
    """Register the scale name into patched_mod.scale_member_list so that the scalar scale is updated into state_dict"""
    if name in patched_mod.scale_members:
        raise ValueError("scale member {} already exists".format(name))
    scale = create_scale_tensor(scale, scale_format)
    patched_mod.scale_members.add(name)
    setattr(patched_mod, name, scale)


def create_scale_tensor(orig_tensor, scale_format):
    if is_runtime_scale_patching() and scale_format in ScaleFormat.__members__.values():
        return orig_tensor.to("cpu").to(torch.float)
    if scale_format == ScaleFormat.CONST:
        if isinstance(orig_tensor, torch.Tensor):
            return torch.nn.Parameter(orig_tensor)
        elif isinstance(orig_tensor, list):
            return [torch.nn.Parameter(x) for x in orig_tensor]
    elif scale_format == ScaleFormat.SCALAR:
        if isinstance(orig_tensor, (torch.Tensor, float)):
            return scale_to_scalar(orig_tensor)
        elif isinstance(orig_tensor, list):
            return [scale_to_scalar(x) for x in orig_tensor]
    else:
        raise ValueError("unexpected scale format value {}".format(scale_format))


# scalar scale is a performance optimization for LLM layers in small BS
def scale_to_scalar(scale):
    if isinstance(scale, torch.Tensor):  # tensor case
        if scale.numel() == 1:
            return scale.item()
        else:
            raise Exception("scale as scalar isn't supported for scale tensors of dim > 0")
    elif isinstance(scale, float):  # already scalar case
        return scale
    else:
        raise Exception(f"Unexpected scale instance type: {type(scale).__name__}, expected Torch.tensor or float number")


def get_scale_dtype(scale):
    if isinstance(scale, torch.Tensor):  # tensor case
        return scale.dtype
    elif isinstance(scale, float):  # already scalar case
        return type(scale).__name__
    elif scale is None: # possible dynamic scalar case
        return None
    else:
        raise Exception(f"Unexpected scale instance type: {type(scale).__name__}, expected Torch.tensor or float number")


def get_param_scales_from_scalar(patched_mod, prefix, dtype=torch.bfloat16, device=torch.device('hpu')):
    """Get all scales in param_list, used for saving scalar scales"""
    scale_dict = {}
    for name in patched_mod.scale_members:
        if hasattr(patched_mod, name) and isinstance(getattr(patched_mod, name), float):
            scale_dict.update({
                # E.g. lm_head.scale_input
                prefix + name: torch.tensor(getattr(patched_mod, name), dtype=dtype, device=device),
            })
    return scale_dict


def get_param_scales_from_list(patched_mod, prefix, dtype=torch.bfloat16, device=torch.device('hpu')):
    """Get all scales in param_list, used for saving scalar scales"""
    scale_dict = {}
    for name in patched_mod.scale_members:
        if hasattr(patched_mod, name) and isinstance(getattr(patched_mod, name), list):
            scale_dict.update({
                # E.g. lm_head.scale_input
                prefix + name: torch.cat(
                    [torch.tensor(v, dtype=dtype, device=device) for v in getattr(patched_mod, name)]
                ),
            })
    return scale_dict


def set_param_scales_as_scalar(patched_mod, state_dict):
    """Set all scales in param_list, used for loading scalar scales"""
    state_dict_keys = list(state_dict.keys())
    for name in patched_mod.scale_members:
        if hasattr(patched_mod, name) and isinstance(getattr(patched_mod, name), float):
            for k in state_dict_keys:
                if name == k.split('.')[-1]:
                    v = state_dict.pop(k)
                    setattr(patched_mod, name, v.item())
    return state_dict


def set_param_scales_into_list(patched_mod, state_dict):
    """Set all scales in param_list, used for loading scalar scales"""
    state_dict_keys = list(state_dict.keys())
    for name in patched_mod.scale_members:
        if hasattr(patched_mod, name) and isinstance(getattr(patched_mod, name), list):
            for k in state_dict_keys:
                if name == k.split('.')[-1]:
                    v = state_dict.pop(k)
                    if patched_mod.scale_format == ScaleFormat.SCALAR:
                        assert v.dim() == 2, "Unexpected scale shape, please raise an issue."
                        v = [v[i].item() for i in range(v.size(0))]
                    else:
                        v = [v[i] for i in range(v.size(0))]
                    setattr(patched_mod, name, v)
    return state_dict


def get_state_dict(patched_mod, *args, destination=None, prefix='', keep_vars=False):
    """replace torch.nn.Module.state_dict"""
    cur_state_dict = torch.nn.Module.state_dict(patched_mod, *args, destination=destination, prefix=prefix, keep_vars=keep_vars)
    device = torch.device('hpu')
    dtype = patched_mod.hp_dtype
    if patched_mod.scale_format == ScaleFormat.SCALAR:
        scale_dict = get_param_scales_from_scalar(patched_mod, prefix, dtype=dtype, device=device)
        cur_state_dict.update(scale_dict)
    get_param_scales_from_list(patched_mod, prefix, dtype=dtype, device=device)
    return cur_state_dict


def load_state_dict(patched_mod, state_dict, prefix, local_metadata, strict,
                    missing_keys, unexpected_keys, error_msgs):
    """replace torch.nn.Module._load_from_state_dict"""
    if patched_mod.scale_format == ScaleFormat.SCALAR:
        state_dict = set_param_scales_as_scalar(patched_mod, state_dict)
    set_param_scales_into_list(patched_mod, state_dict)
    torch.nn.Module._load_from_state_dict(patched_mod, state_dict, prefix, local_metadata, strict,
                    missing_keys, unexpected_keys, error_msgs)


def update_state_dict_method(patched_mod):
    """update fetch and load state_dict method for scalar scales"""
    patched_mod.state_dict = types.MethodType(get_state_dict, patched_mod)
    patched_mod._load_from_state_dict = types.MethodType(load_state_dict, patched_mod)
