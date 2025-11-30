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

from .._quant_common.helper_modules import *
# TODO [SW-217813]: support dynamic quantization in all ops and remove supported_dynamic_ops
from .._quant_common.quant_config import QuantMode, get_hqt_config, is_supported_dynamic_op
from ..utils.logger import logger
from .patching_common import mod_default_dict
from .measure import prepare_model as prepare_model_for_measure
from .quantize import quantize
from .scale_methods.scale_method_config import get_scale_method_from_config, ScaleMethodString, CfgStr
from .common import is_runtime_scale_patching
from neural_compressor.torch.utils.auto_accelerator import is_any_gaudi_accelerator
import os
import re


def update_mod_dict(config):
    assert (
        len(config.cfg["mod_dict"]) == 0
    ), f"Custom modules are not supported: {config.cfg['mod_dict'].keys()}. Please add it in the code."
    config.cfg["mod_dict"].update({k: mod_default_dict[k].type for k in mod_default_dict})


def print_init_info(config):
    import importlib.metadata

    try:
        versionStr = importlib.metadata.version("neural_compressor_pt")
    except:
        # in case the backend specific package is not installed, we can still get the git revision
        versionStr = importlib.metadata.version("neural_compressor")
    locationStr = versionStr.find("git") + 3
    logger.info("neural_compressor_pt Git revision = %s", versionStr[locationStr:])
    logger.info("neural_compressor_pt Configuration = %s", config)


def is_re_match(substr_list, target):
    for substr in substr_list:
        if re.search(substr, target):
            return True
    return False


def should_quantize(config, mod_type, name):
    def mod_is_not_blocked(mod_type, config):
        allowlist = set(config.cfg["mod_dict"].keys())
        blocklist = set()
        for type_st in config.cfg["blocklist"]["types"]:
            blocklist.add(type_st)
        allowlist.difference_update(blocklist)
        allowlist_tuple = tuple(allowlist)
        return (mod_type in allowlist_tuple)
    def allowlist_is_empty_or_allows_mod(mod_type, name, config):
        def mod_is_in_allowlist_config(mod_type, name, config):
            return ((mod_type in config.cfg["allowlist"]["types"]) or (is_re_match(config.cfg["allowlist"]["names"], name)))
        def is_allowlist_completely_empty(config):
            return ((len(config.cfg["allowlist"]["names"]) == 0) and len(config.cfg["allowlist"]["types"]) == 0)
        return (mod_is_in_allowlist_config(mod_type, name, config) or is_allowlist_completely_empty(config))
    def name_is_not_blocked(name, config):
        return (not is_re_match(config.cfg["blocklist"]["names"], name))
    def is_static_scale_method(config):
        return not config.cfg["dynamic_quantization"]
    def quantize_dynamic_op(config, mod_type):
        # TODO [SW-217813]: support dynamic quantization in all ops and remove supported_dynamic_ops
        return config.cfg["dynamic_quantization"] and is_supported_dynamic_op(mod_type)

    ret = (
        mod_is_not_blocked(mod_type, config)
        and allowlist_is_empty_or_allows_mod(mod_type, name, config)
        and name_is_not_blocked(name, config)
        # TODO [SW-217813]: support dynamic quantization in all ops and remove supported_dynamic_ops
        and (is_static_scale_method(config) or quantize_dynamic_op(config, mod_type))
    )
    logger.trace(f"should_quantize {name=} {mod_type=} returning {ret}")
    return ret


scaling_methods_list = [scale_method.name for scale_method in ScaleMethodString]
#exlude substrings of scaling methods which are not supported for runtime scale patching mode to reduce graph recompile.
exclude_substrings = ["PCS", "SMOOTHQUANT"]
runtime_scale_patching_supported_methods_list = [method for method in scaling_methods_list if not any(substr in method for substr in exclude_substrings)]


def set_runtime_scale_patching_mode(scale_method_config):
    import habana_frameworks.torch.utils.experimental as htexp # importing in local scope since it is gaudi specific
    scale_method = get_scale_method_from_config(scale_method_config[CfgStr.DEFAULT])
    if is_runtime_scale_patching() and hasattr(htexp, "_set_scale_attributes"):
        assert (
            scale_method.name in runtime_scale_patching_supported_methods_list
        ), f"Scaling method \"{scale_method}\" is not supported for runtime scale patching (graph recompile reduction). Cannot set scaling attributes."
        htexp._set_scale_attributes("HW" in scale_method.name or scale_method.name == "UNIT_SCALE",
                                    scaling_methods_list.index(scale_method.name) + 1)


def prepare_model(model):
    """Receives the parent module to quantize.
    Replaces its submodules with patched submodules that perform calibration and quantization.
    Returns the patched parent module that can perform calibration or quantization according to the configuration.

    Args:
        model (nn.module): The model that will be measured/quantized.
    """
    config = get_hqt_config(model)
    update_mod_dict(config)
    mod_list = []
    for name, mod in model.named_modules():
        mod_type = mod.__class__.__name__
        if should_quantize(config, mod_type, name):
            mod_list.append(name)

    print_init_info(config)
    logger.debug("Module list: %s", mod_list)
    logger.info("Total modules : %d", len(mod_list))
    if (config.cfg["mode"] == QuantMode.MEASURE) or (config.cfg["mode"] == QuantMode.SHAPE):
        return prepare_model_for_measure(model, mod_list)
    elif config.cfg["mode"] in [QuantMode.QUANTIZE, QuantMode.LOAD]:
        if is_any_gaudi_accelerator(config.cfg["device_type"]):
            set_runtime_scale_patching_mode(config.cfg["scale_method"])
        return quantize(model, mod_list)
