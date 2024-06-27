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

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.utils import logger
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator

from .config import ConfigMappingType, HQQModuleConfig, QTensorConfig, hqq_global_option
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


class HQQuantizer(Quantizer):
    def __init__(self, quant_config: ConfigMappingType) -> None:
        """Init a HQQuantizer object.

        Args:
            quant_config (ConfigMappingType): quantization config for ops.
        """
        quant_config = self._parse_hqq_configs_mapping(quant_config)
        super().__init__(quant_config=quant_config)

    @torch.no_grad()
    def prepare(self, model: torch.nn.Module, *args, **kwargs) -> Optional[torch.nn.Module]:
        """Prepares a given model for quantization.

        Will return model directly in HQQ algorithm.

        Args:
            model (torch.nn.Module): The model to be prepared.
        """
        return model

    @torch.no_grad()
    def convert(self, model: torch.nn.Module, *args, **kwargs) -> Optional[torch.nn.Module]:
        """Converts a prepared model to a quantized model.

        Args:
            model (torch.nn.Module): The prepared model to be converted.

        Returns:
            Optional[torch.nn.Module]: A quantized model.
        """
        _replace_with_custom_fn_if_matches_filter(
            model, replacement_fn=replacement_fn, filter_fn=filter_fn, config_mapping=self.quant_config
        )
        return model

    def save(self, model, path):
        # TODO: to implement it in the next PR
        pass

    def _convert_hqq_module_config(self, config) -> HQQModuleConfig:
        # TODO: (Yi) Please note that the configuration defined by INC should be separated from the algorithm.
        # * 3.x API use `bits` for woq while HQQ internal API use `nbits`, we should change it in algorithm_entry.py
        nbits = config.bits
        group_size = config.group_size
        quant_zero = config.quant_zero
        quant_scale = config.quant_scale
        scale_quant_group_size = config.scale_quant_group_size

        weight_qconfig = QTensorConfig(
            nbits=nbits,
            channel_wise=True,
            group_size=group_size,
            optimize=True,
            round_zero=True if nbits == 4 else False,
        )
        zero_qconfig = None
        if quant_zero:
            zero_qconfig = QTensorConfig(nbits=8, channel_wise=False, group_size=None, optimize=False)
        scale_qconfig = None
        if quant_scale:
            scale_qconfig = QTensorConfig(nbits=8, channel_wise=True, group_size=scale_quant_group_size, optimize=False)
        hqq_module_config = HQQModuleConfig(weight=weight_qconfig, scale=scale_qconfig, zero=zero_qconfig)
        logger.debug(hqq_module_config)
        return hqq_module_config

    def _parse_hqq_configs_mapping(self, configs_mapping):
        qconfig_mapping = {}
        for (op_name, op_type), quant_config in configs_mapping.items():
            if quant_config is not None and quant_config.dtype == "fp32":
                logger.warning("Fallback %s.", op_name)
                continue
            qconfig_mapping[op_name] = self._convert_hqq_module_config(quant_config)
        return qconfig_mapping
