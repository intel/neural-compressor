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
from .._quant_common.quant_config import QuantMode, get_hqt_config, is_supported_dynamic_op, _dynamic_scale_methods
from ..utils.logger import logger
from .patching_common import mod_default_dict
from .measure import prepare_model as prepare_model_for_measure
from .quantize import quantize
from .scale import scale_method_mapping, scaling_params


def update_mod_dict(config):
    assert (
        len(config.cfg["mod_dict"]) == 0
    ), f"Custom modules are not supported: {config.cfg['mod_dict'].keys()}. Please add it in the code."
    config.cfg["mod_dict"].update({k: mod_default_dict[k].type for k in mod_default_dict})


def print_init_info(config):
    import importlib.metadata

    versionStr = importlib.metadata.version("neural_compressor_pt")
    locationStr = versionStr.find("git") + 3
    logger.info("neural_compressor_pt Git revision = %s", versionStr[locationStr:])
    logger.info("neural_compressor_pt Configuration = %s", config)


def is_substr(substr_list, target):
    return any([x in target for x in substr_list])


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
            return ((mod_type in config.cfg["allowlist"]["types"]) or (is_substr(config.cfg["allowlist"]["names"], name)))
        def is_allowlist_completely_empty(config):
            return ((len(config.cfg["allowlist"]["names"]) == 0) and len(config.cfg["allowlist"]["types"]) == 0)
        return (mod_is_in_allowlist_config(mod_type, name, config) or is_allowlist_completely_empty(config))
    def name_is_not_blocked(name, config):
        return (not is_substr(config.cfg["blocklist"]["names"], name))
    def is_static_scale_method(config):
        return config.cfg["scale_method"] not in _dynamic_scale_methods
    def quantize_dynamic_op(config, mod_type):
        # TODO [SW-217813]: support dynamic quantization in all ops and remove supported_dynamic_ops
        return config.cfg["scale_method"] in _dynamic_scale_methods and is_supported_dynamic_op(mod_type)

    ret = (
        mod_is_not_blocked(mod_type, config)
        and allowlist_is_empty_or_allows_mod(mod_type, name, config)
        and name_is_not_blocked(name, config)
        # TODO [SW-217813]: support dynamic quantization in all ops and remove supported_dynamic_ops
        and (is_static_scale_method(config) or quantize_dynamic_op(config, mod_type))
    )
    logger.trace(f"should_quantize {name=} {mod_type=} returning {ret}")
    return ret

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
        scaling_method_name = scale_method_mapping[(config.cfg["scale_method"], config.cfg["observer"])]
        scaling_params[scaling_method_name].update(config.cfg["scale_params"])
        config.cfg["scale_params"] = scaling_params[scaling_method_name]
        return quantize(model, mod_list)
