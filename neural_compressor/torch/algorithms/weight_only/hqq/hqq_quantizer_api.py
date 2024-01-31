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


class EagerModeQuantizer:
    def prepare(self):
        pass

    def calibrate(self):
        pass

    def convert(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


from typing import Dict

from hqq_config import *


class HQQutizer(EagerModeQuantizer):
    def prepare(self, qconfig_mapping: Dict[str, HQQModuleConfig]):
        # Replace `Linear` with `HQQLinear`
        pass

    def convert(self):
        pass


class HQQGlobalOptions:
    use_half = True


from typing import Any, Callable, Dict, List, Optional, Tuple

from typing_extensions import TypeAlias

ConfigMappingType: TypeAlias = Dict[str, str]
algos_mapping: Dict[str, Callable] = {}
config_mapping: Dict[str, Any] = {}

from inc_accelerator import auto_detect_accelerator


def has_child(module: torch.nn.Module) -> bool:
    return len(list(module.named_children())) > 0


hqq_global_option = HQQGlobalOptions()


def _replace_with_custom_fn_if_matches_filter(
    model: torch.nn.Module,
    replacement_fn: Callable,
    filter_fn: Callable,
    cur_fqn: str = "",
    config_mapping: Optional[ConfigMappingType] = None,
) -> None:
    """For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`"""
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if cur_fqn == "":
            new_fqn = name
        else:
            new_fqn = f"{cur_fqn}.{name}"
        if filter_fn(child, new_fqn, config_mapping):
            new_child = replacement_fn(child.to(auto_detect_accelerator().current_device()), new_fqn, config_mapping)
            print(f'quantized {new_fqn}, device: {getattr(new_child, "device", None)}')
            setattr(model, name, new_child)
        elif not has_child(child) and hqq_global_option:  # TODO: merge it into `filter_fn`
            print(f"halfing {new_fqn}")
            new_child = child.half().to(auto_detect_accelerator().current_device())
            print(f'halfing {new_fqn}, device: {getattr(new_child, "device", None)}')
            setattr(model, name, new_child)
        else:
            _replace_with_custom_fn_if_matches_filter(
                model=child,
                replacement_fn=replacement_fn,
                filter_fn=filter_fn,
                cur_fqn=new_fqn,
                config_mapping=config_mapping,
            )


from hqq_utils import HQQLinear


def patch_hqq(mod, config):
    new_mod = HQQLinear.from_float(mod, config)
    return new_mod


def filter_fn(mod: torch.nn.Module, name: str, config_mapping: ConfigMappingType) -> bool:
    return isinstance(mod, torch.nn.Linear) and name in config_mapping


def replacement_fn(mod: torch.nn.Module, name: str, config_mapping: ConfigMappingType) -> torch.nn.Module:
    config = config_mapping.get(name, None)
    print(f"Replace module {name}")
    return patch_hqq(mod, config)


def get_model_info(model: torch.nn.Module) -> List[Tuple[str, Callable]]:
    white_list = (torch.nn.Linear,)
    filter_result = []
    for op_name, module in model.named_modules():
        if isinstance(module, white_list):
            pair = (op_name, type(module).__name__)
            filter_result.append(pair)
    return filter_result


from hqq_utils import default_hqq_quant_config


def get_default_hqq_config_mapping(model):
    mode_info = get_model_info(model)
    config_mapping = {}
    for name, _ in mode_info:
        config_mapping[name] = default_hqq_quant_config
    return config_mapping


def hqq_entry(model, config_mapping):
    _replace_with_custom_fn_if_matches_filter(
        model, replacement_fn=replacement_fn, filter_fn=filter_fn, config_mapping=config_mapping
    )
    return model
