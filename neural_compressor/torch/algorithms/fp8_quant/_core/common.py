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

import functools
import importlib.util
import json
import os
from collections import namedtuple

import numpy as np
import torch

from .._quant_common.helper_modules import *
from .._quant_common.quant_config import VERBOSE, get_hqt_config

deepspeed_exists = False
if importlib.util.find_spec("deepspeed"):  # check if deepspeed is installed
    deepspeed_exists = True

module_info = namedtuple("ModuleInfo", ["type", "patched_module"])

module_config = namedtuple("ModuleConfig", ["inputs", "outputs", "params"], defaults=((None,), None, {}))
module_extra_config = namedtuple(
    "ModuleExtraConfig",
    ["inputs", "outputs", "params", "scale", "config_params"],
    defaults=((None,), None, {}, None, {}),
)
quantdequant_config = namedtuple(
    "QuantDequantConfig", ["scale_quant_fcn", "quant_fcn", "scale_dequant_fcn", "dequant_fcn", "lp_dtype", "hp_dtype"]
)
mod_type = namedtuple("ModuleType", ["inputs", "params"])
mod_types = {
    "linear": mod_type(1, ["weight"]),
    "matmul": mod_type(2, []),
    "kv_cache": mod_type(1, []),
    "softmax": mod_type(1, []),
}
descale_fcn = lambda x, scale: torch.mul(x, scale)
scale_fcn = lambda x, scale: torch.div(x, scale)
mat_scale_fcn = lambda x, scale_col, scale_row: torch.div(torch.div(x, scale_col), scale_row)
cast_fcn = lambda x, dtype: x.to(dtype=dtype)
cast_to_fp8_fcn = lambda x, dtype, scale_inv=None: torch.ops.hpu.cast_to_fp8_v2(x, scale_inv, False, False, dtype)[0]
cast_from_fp8_fcn = lambda x, dtype, scale=None: torch.ops.hpu.cast_from_fp8(x, scale, dtype)


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


def np_to_pt(x):
    return rec_fn(x, lambda x: torch.tensor(x) if isinstance(x, np.ndarray) else x)


def pt_to_np(x):
    return rec_fn(x, lambda x: x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else x)


def np_to_list(x):
    return rec_fn(x, lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def list_to_np(x):
    return rec_fn(x, lambda x: np.array(x) if isinstance(x, list) else x)


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


def save_file(model, d, source_format, fname, mode):
    config = get_hqt_config(model)
    if VERBOSE:
        print(f"Saving {mode} file: {fname}")
    ext = os.path.splitext(fname)[1]
    target_format = file_functions[ext][0]
    dc = rec_fn(d, format_functions[(source_format, target_format)])
    df = {"GlobalRank": config.cfg["global_rank"], "LocalRank": config.cfg["local_rank"], "Mode": mode, "Nodes": dc}
    try:
        file_functions[ext][1](df, fname)
    except:
        pass


def dict_to_namedtuple(d, cls):
    m = cls(**d)
    return m


def module_convert(m, fcn):
    mt = module_config(tuple([fcn(x) for x in m.inputs]), fcn(m.outputs), {k: fcn(m.params[k]) for k in m.params})
    return mt


def fix_fields(d):
    if "input" in d:
        d["inputs"] = d.pop("input")
    if "output" in d:
        d["outputs"] = d.pop("output")
    return d


def load_file(fname, target_format, fail_on_file_not_exist):
    if VERBOSE:
        print(f"Loading file: {fname}")
    ext = os.path.splitext(fname)[1]
    source_format = file_functions[ext][0]
    d = {}
    if os.path.isfile(fname):
        d = file_functions[ext][2](fname)
    elif fail_on_file_not_exist:
        raise FileNotFoundError(f"Failed to load file {fname}")
    if "Nodes" in d:
        dc = {k: dict_to_namedtuple(fix_fields(d["Nodes"][k]), module_config) for k in d["Nodes"]}
        dc = {k: module_convert(dc[k], format_functions[(source_format, target_format)]) for k in dc}
    else:
        dc = {}
    return dc


def save_scales(model, d, source_format, fname):
    dc = {k: d[k]._asdict() for k in d}
    save_file(model, dc, source_format, fname, "Scale")


def load_scales(fname, target_format):
    if VERBOSE:
        print(f"Loading scales file {fname}")
    d = load_file(fname, target_format, False)
    return d


def convert_scales_to_tensors_dict(scales_obj, scales_file_format, hp_dtype):
    scales_temp = {k: scales_obj[k]._asdict() for k in scales_obj}
    scales_temp = format_functions_rec((scales_file_format, torch.Tensor))(scales_temp)
    scales_temp = rec_fn(scales_temp, lambda x: x.to(dtype=hp_dtype, device="hpu"))
    scales = {k: module_config(**scales_temp[k]) for k in scales_temp}
    return scales


file_functions = {".json": (list, save_json, load_json), ".npz": (np.ndarray, save_npz, load_npz)}

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

mod_default_dict = {
    "Matmul": module_info("matmul", PatchedMatmul),
    "Linear": module_info("linear", PatchedLinear),
    "FalconLinear": module_info("linear", PatchedLinear),
    "KVCache": module_info("kv_cache", PatchedKVCache),
    "Conv2d": module_info("linear", PatchedConv2d),
    "LoRACompatibleLinear": module_info("linear", PatchedLoRACompatibleLinear),
    "LoRACompatibleConv": module_info("linear", PatchedLoRACompatibleConv),
    "Softmax": module_info("softmax", PatchedSoftmax),
}


if deepspeed_exists:
    mod_default_dict.update(
        {
            "LinearLayer": module_info("linear", PatchedLinear),
            "LinearAllreduce": module_info("linear", PatchedLinearAllReduce),
            "ScopedLinearAllReduce": module_info("linear", PatchedLinearAllReduce),
            "LmHeadLinearAllreduce": module_info("linear", PatchedLmHeadLinearAllreduce),
        }
    )

mod_inst_info = namedtuple("ModInstInfo", ["name", "parent"])

parent_child_mod_dict = {}


def generate_model_info(model):
    def create_mod_info_recursion(parent):
        for name, mod in parent.named_children():
            parent_child_mod_dict[mod] = mod_inst_info(name=name, parent=parent)
            create_mod_info_recursion(mod)

    create_mod_info_recursion(model)
