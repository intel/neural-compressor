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

from typing import Callable, List, Optional, Tuple

import torch

from neural_compressor.torch.utils import logger
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator

from .config import ConfigMappingType, default_hqq_module_config, hqq_global_option
from .core import HQQLinear


def _has_child(module: torch.nn.Module) -> bool:
    return len(list(module.named_children())) > 0


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
            logger.debug("Quantize linear module %s.", new_fqn)
            setattr(model, name, new_child)
        elif not _has_child(child):  # TODO: merge it into `filter_fn`
            if hqq_global_option.use_half:
                logger.debug("Half module %s.", new_fqn)
                child = child.half()
            new_child = child.to(auto_detect_accelerator().current_device())
            setattr(model, name, new_child)
        else:
            _replace_with_custom_fn_if_matches_filter(
                model=child,
                replacement_fn=replacement_fn,
                filter_fn=filter_fn,
                cur_fqn=new_fqn,
                config_mapping=config_mapping,
            )


def patch_hqq_moduile(mod, config):
    new_mod = HQQLinear.from_float(mod, config)
    return new_mod


def filter_fn(mod: torch.nn.Module, name: str, config_mapping: ConfigMappingType) -> bool:
    return isinstance(mod, torch.nn.Linear) and name in config_mapping


def replacement_fn(mod: torch.nn.Module, name: str, config_mapping: ConfigMappingType) -> torch.nn.Module:
    config = config_mapping.get(name, None)
    logger.debug("Replace module %s", name)
    return patch_hqq_moduile(mod, config)


class EagerModeQuantizer:
    def __init__(self, config_mapping) -> None:
        self.config_mapping = config_mapping

    def prepare(self, model: torch.nn.Module, inplace=True) -> Optional[torch.nn.Module]:
        pass

    def convert(self, model: torch.nn.Module, inplace=True) -> Optional[torch.nn.Module]:
        pass

    def save(self):
        pass


class HQQuantizer(EagerModeQuantizer):
    def __init__(self, config_mapping: ConfigMappingType) -> None:
        super().__init__(config_mapping)

    def prepare(self, model: torch.nn.Module, inplace=True) -> Optional[torch.nn.Module]:
        _replace_with_custom_fn_if_matches_filter(
            model, replacement_fn=replacement_fn, filter_fn=filter_fn, config_mapping=self.config_mapping
        )
        return model

    def save(self, model, path):
        # TODO: to implement it in the next PR
        pass
