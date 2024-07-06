import torch

from ..fp_utils import *
from ..common import *


def linear_unit_scale_scales(mod, measurement, params):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    weight_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    output_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    return ModuleConfig((input_scale,), (output_scale,), {"weight": weight_scale})


def fsdpa_unit_scale_scales(mod, measurement, params):
    device = torch.device("hpu")
    hp_dtype = torch.float32  #  params["hp_dtype"]
    q_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    k_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    v_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    softmax_scale = torch.tensor(1.0, dtype=hp_dtype, device=device)
    input_scale = (q_scale, k_scale, v_scale, softmax_scale)
    output_scale = (torch.tensor(1.0, dtype=hp_dtype, device=device),)
    return ModuleConfig(input_scale, output_scale, {})


def matmul_unit_scale_scales(mod, measurement, params):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = (
        torch.tensor(1.0, dtype=hp_dtype, device=device),
        torch.tensor(1.0, dtype=hp_dtype, device=device),
    )
    output_scale = (torch.tensor(1.0, dtype=hp_dtype, device=device),)
    return ModuleConfig(input_scale, output_scale, {})


def softmax_unit_scale_scales(mod, measurement, params):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = (torch.tensor(1.0, dtype=hp_dtype, device=device),)
    output_scale = (torch.tensor(1.0, dtype=hp_dtype, device=device),)
    return ModuleConfig(input_scale, output_scale)


def kv_cache_unit_scale_scales(mod, measurement, params):
    device = torch.device("hpu")
    hp_dtype = params["hp_dtype"]
    input_scale = (torch.tensor(1.0, dtype=hp_dtype, device=device),)
    output_scale = (torch.tensor(1.0, dtype=hp_dtype, device=device),)
    return ModuleConfig(input_scale, output_scale)
