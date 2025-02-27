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

import importlib.util

from ..model_configs import ModuleInfo, ModuleType
from .._quant_common.helper_modules import *

from neural_compressor.torch.algorithms.fp8_quant.model_configs import (
    get_patched_module_table,
    get_patched_module_type_table,
)
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator
deepspeed_exists = False
if importlib.util.find_spec("deepspeed"):  # check if deepspeed is installed
    deepspeed_exists = True

mod_default_dict = get_patched_module_table()
mod_types = get_patched_module_type_table()


def get_white_list():
    return list(mod_default_dict.keys())


class ModInstInfo:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent


parent_child_mod_dict = {}


def generate_model_info(model):
    def create_mod_info_recursion(parent):
        for name, mod in parent.named_children():
            parent_child_mod_dict[mod] = ModInstInfo(name, parent)
            create_mod_info_recursion(mod)

    create_mod_info_recursion(model)

_mod_types = {
    "linear": ModuleType(1, ["weight"], 1, False),
    "matmul": ModuleType(2, [], 1, False),
    "kv_cache": ModuleType(1, [], 1, False),
    "softmax": ModuleType(1, [], 1, True),
    "fused_sdpa": ModuleType(3, [], 2, True),
    "dynamic_moe": ModuleType(1, [], 9, True),
}


_mod_default_dict = {
    "Matmul": ModuleInfo("matmul", PatchedMatmul),
    "Linear": ModuleInfo("linear", PatchedLinear),
    "ParallelLMHead": ModuleInfo("linear", PatchedParallelLMHead),
    "RowParallelLinear": ModuleInfo("linear", PatchedRowParallelLinear),
    "ColumnParallelLinear": ModuleInfo("linear", PatchedColumnParallelLinear),
    "MergedColumnParallelLinear": ModuleInfo("linear", PatchedColumnParallelLinear),
    "QKVParallelLinear": ModuleInfo("linear", PatchedColumnParallelLinear),
    "FalconLinear": ModuleInfo("linear", PatchedLinear),
    "KVCache": ModuleInfo("kv_cache", PatchedKVCache),
    "VLLMKVCache": ModuleInfo("kv_cache", PatchedVLLMKVCache),
    "Conv2d": ModuleInfo("linear", PatchedConv2d),
    "LoRACompatibleLinear": ModuleInfo("linear", PatchedLoRACompatibleLinear),
    "LoRACompatibleConv": ModuleInfo("linear", PatchedLoRACompatibleConv),
    "Softmax": ModuleInfo("softmax", PatchedSoftmax),
    "ModuleFusedSDPA": ModuleInfo("fused_sdpa", PatchedModuleFusedSDPA),
    "MoeMatmul": ModuleInfo("linear", PatchedMoeMatmul),
    "ReplicatedLinear": ModuleInfo("linear", PatchedReplicatedLinear),
    "FusedMoE": ModuleInfo("linear", PatchedMixtralMoE, False),
    "GaudiMixtralSparseMoeBlock": ModuleInfo("dynamic_moe", PatchedGaudiMixtralSparseMoeBlock),
    "VllmMixtureOfExpertsOp": ModuleInfo("dynamic_moe", PatchedVllmMixtureOfExpertsOp),
}


if deepspeed_exists:
    _mod_default_dict.update(
        {
            "LinearLayer": ModuleInfo("linear", PatchedLinear),
            "LinearAllreduce": ModuleInfo("linear", PatchedLinearAllReduce),
            "ScopedLinearAllReduce": ModuleInfo("linear", PatchedLinearAllReduce),
            "LmHeadLinearAllreduce": ModuleInfo("linear", PatchedLmHeadLinearAllreduce),
        }
    )


@functools.lru_cache(maxsize=None)
def _import_hpu_modules():
    from neural_compressor.torch.algorithms.fp8_quant.patched_module_base import (
        PATCHED_MODULE_TABLE,
        PATCHED_MODULE_TYPES_TABLE,
    )

    cur_accelerator = auto_detect_accelerator()
    if not cur_accelerator.current_device_name().startswith("hpu"):
        return
    PATCHED_MODULE_TABLE["hpu"].update(_mod_default_dict)
    PATCHED_MODULE_TYPES_TABLE["hpu"].update(_mod_types)


_import_hpu_modules()
