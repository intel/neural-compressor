# Copyright (c) 2025 Intel Corporation
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

import functools
import json
import os

import numpy as np
import torch
from enum import Enum, auto
from functools import lru_cache 
from ..utils.logger import logger
from neural_compressor.torch.algorithms.fp8_quant.model_configs import ModuleConfig
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator

cur_device = auto_detect_accelerator().current_device_name()

UNMEASURED_MODELS = "UnmeasuredModels"

def dequant_original_fp8_weight_if_needed(mod: torch.nn.Module, param: torch.Tensor) -> torch.Tensor:
    if param.dtype in [torch.float8_e4m3fn]:
        if hasattr(mod, "get_dequant_weights_func"):
            dequant_weights_func = mod.get_dequant_weights_func()
            if dequant_weights_func is not None:
                param = dequant_weights_func(mod)
            else:
                raise RuntimeError(f"Got fp8 weight for {mod}, but dequant function is None, please check.")
        else:
            RuntimeError(f"Got fp8 weight for {mod}, but dequant function is not found, please check.")
            
    return param

class QuantTensorType(Enum):
    MEASUREMENTS = auto()
    CONST = auto()
    DYNAMIC = auto()


class ShapeList:
    data = None


def rec_fn(x, fn):
    if isinstance(x, dict):
        return {k: rec_fn(x[k], fn) for k in x}
    elif isinstance(x, list):
        return [rec_fn(k, fn) for k in x]
    elif isinstance(x, tuple):
        return tuple([rec_fn(k, fn) for k in x])
    else:
        return fn(x)


def save_json(d, fname):
    with open(fname, "w") as f:
        json.dump(d, f, indent=4)


def load_json(fname):
    with open(fname, "r") as f:
        d = json.load(f)
    return d


def save_npz(d, fname):
    np.savez(fname, d)


def load_npz(fname):
    d = np.load(fname, allow_pickle=True)
    return d["arr_0"].item()


def save_file(model, d, source_format, fname, mode, num_samples=0):
    from .._quant_common.quant_config import get_hqt_config
    config = get_hqt_config(model)
    logger.debug("Saving %s file: %s", mode, fname)
    ext = os.path.splitext(fname)[1]
    target_format = file_functions[ext]['format']
    dc = rec_fn(d, format_functions[(source_format, target_format)])
    df = {
        "GlobalRank": config.cfg["global_rank"],
        "LocalRank": config.cfg["local_rank"],
        "Mode": mode,
        "Nodes": dc,
    }
    if num_samples > 0:
        df = { "NumSamples": num_samples, **df}
    try:
        file_functions[ext]['save'](df, fname)
    except:
        pass


def load_file(fname, target_format, fail_on_file_not_exist):
    logger.debug("Loading file: %s", fname)
    ext = os.path.splitext(fname)[1]
    source_format = file_functions[ext]['format']
    d = {}
    if os.path.isfile(fname):
        d = file_functions[ext]['load'](fname)
    elif fail_on_file_not_exist:
        raise FileNotFoundError(f"Failed to load file {fname}")
    if "Nodes" in d:
        dc = {k: ModuleConfig(**fix_fields(d["Nodes"][k])) for k in d["Nodes"]}
        dc = {k: module_convert(dc[k], format_functions[(source_format, target_format)]) for k in dc}
    else:
        dc = {}
    return dc


# convert module config data to other format
def module_convert(m, fcn):
    mt = ModuleConfig(
        tuple([fcn(x) for x in m.inputs]),
        (
            tuple(
                [fcn(m.outputs)],
            )
            if type(m.outputs) == np.ndarray
            else tuple([fcn(y) for y in m.outputs])
        ),
        {k: fcn(m.params[k]) for k in m.params},
    )
    return mt


def fix_fields(d):
    if "input" in d:
        d["inputs"] = d.pop("input")
    if "output" in d:
        d["outputs"] = d.pop("output")
    return d


def save_scales(model, d, source_format, fname):
    """Saves scales measured of a given model.

    Args:
        model : The measured model.
        d : Modules_names to configuration dictionary.
        source_format : How the data is stored in memory.
        fname : File to save the scales to.
    """
    dc = {k: d[k].__dict__ for k in d}
    save_file(model, dc, source_format, fname, "Scale")


def load_scales(fname, target_format):
    """Loads scales from given file.

    Args:
        fname : File to load the scales from.
        target_format: How the data is stored in file.
    """
    logger.debug("Loading scales file %s", fname)
    d = load_file(fname, target_format, False)
    return d


def convert_scales_to_tensors_dict(scales_obj, scales_file_format, hp_dtype, device=cur_device):
    scales_temp = {k: scales_obj[k].__dict__ for k in scales_obj}
    scales_temp = format_functions_rec((scales_file_format, torch.Tensor))(scales_temp)
    scales_temp = rec_fn(scales_temp, lambda x: x.to(dtype=hp_dtype, device=device))
    scales = {k: ModuleConfig(**scales_temp[k]) for k in scales_temp}
    return scales


file_functions = {
    ".json": {'format': list, 'save': save_json, 'load': load_json},
    ".npz": {'format': np.ndarray, 'save': save_npz, 'load': load_npz}
}

format_functions = {
    (torch.Tensor, torch.Tensor): lambda x: x,
    (np.ndarray, np.ndarray): lambda x: x,
    (list, list): lambda x: x,
    (torch.Tensor, np.ndarray): lambda x: x.detach().cpu().float().numpy(),
    (torch.Tensor, list): lambda x: x.detach().cpu().float().numpy().tolist(),
    (np.ndarray, torch.Tensor): torch.tensor,
    (np.ndarray, list): lambda x: x.tolist(),
    (list, torch.Tensor): torch.tensor,
    (list, np.ndarray): lambda x: np.array(x),
    (list, ShapeList): lambda x: [int(s) for s in x[0]],
}


format_functions_rec = lambda k: functools.partial(rec_fn, fn=format_functions[k])


def get_device_type_for_scales(mod):
    from .._quant_common.quant_config import get_hqt_config
    config = get_hqt_config(mod).cfg
    return config["device_for_scales"]


@lru_cache
def is_runtime_scale_patching():
    return os.getenv("RUNTIME_SCALE_PATCHING", "False").lower() in ["true", "1"]

#TODO [SW-224612]: Use cguid to calc scales and remove the check
@lru_cache
def is_calc_scale_with_cguid():
    return os.getenv("CALC_SCALE_WITH_CGUID", "True").lower() in ["true", "1"]

#TODO [SW-224612]: Use cguid to calc scales and remove the check
@lru_cache
def is_calc_scale_rounding_with_cguid():
    return is_calc_scale_with_cguid() and os.getenv("CALC_ROUNDING_WITH_CGUID", "False").lower() in ["true", "1"]
