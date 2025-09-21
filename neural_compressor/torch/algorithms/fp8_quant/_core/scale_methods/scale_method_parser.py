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

from .scale_method_config import (
    dict_to_scale_method_config,
    get_scale_method_from_config,
    ScaleMethodString,
    scale_method_config_mapping,
    CfgStr,
    ScaleGranularity,
    ScaleValueType,
    ScaleRoundMethod
)
from ...utils.logger import logger


def get_enum_from_string(enum_class, string):
    """
    Convert a string to the corresponding enum value of enum_class.
    Raises ValueError if the string does not match any enum member.
    """
    try:
        return enum_class[string.upper()]
    except KeyError:
        valid = [e.name for e in enum_class]
        raise ValueError(
            f"Invalid input '{string}' for {enum_class.__name__}. Enter one of {valid}"
        )


def convert_scale_method_strings_to_enum(scale_method):
    """
    Recursively converts string values in a scale_method dict to their corresponding enums.
    Only dict-based configs are supported. Top-level strings are not allowed.
    """
    if not isinstance(scale_method, dict):
        raise ValueError("Scale method config must be a dictionary with keys like 'granularity', 'scale_value_type', etc.")

    result = {}
    for k, v in scale_method.items():
        if k == "granularity":
            result[k] = get_enum_from_string(ScaleGranularity, v) if isinstance(v, str) else v
        elif k == "scale_value_type":
            result[k] = get_enum_from_string(ScaleValueType, v) if isinstance(v, str) else v
        elif k == "rounding_method":
            result[k] = get_enum_from_string(ScaleRoundMethod, v) if isinstance(v, str) else v
        else:
            # Recursively convert for all other keys (including nested dicts)
            result[k] = convert_scale_method_strings_to_enum(v) if isinstance(v, dict) else v
    return result


def parse_scale_method(scale_method):
    """
    Parses a user-provided scale method configuration into a normalized internal format.

    The input `scale_method` can be either:
      * A ScaleMethodString enum (applies a predefined method globally)
      * A dictionary specifying:
          - "default": the default scale method (a dict with 'weight' and 'activation' configs)
          - Optionally, per-node, per-layer, or per-layer-type overrides under the keys "nodes", "layers", and "layer_types".
            Each override must be a dict with 'weight' and 'activation' configs.

    This function converts all such specifications into a nested dictionary structure where each entry
    is mapped to a pair of ScaleMethodConfig objects (for weight and activation), ready for use by the quantization engine.
    """
    scale_method_config = {}
    if isinstance(scale_method, ScaleMethodString):
        logger.trace(f"Parsing scale method: using predefined ScaleMethodString '{scale_method.name}' as default.")
        scale_method_config[CfgStr.DEFAULT] = scale_method_config_mapping[scale_method]
        logger.trace(f"Scale method config found: {scale_method_config[CfgStr.DEFAULT]}")
    elif isinstance(scale_method, dict):
        logger.trace("Parsing scale method: using user-provided dictionary configuration.")
        # create default scale method
        scale_method_config_default = scale_method.get(CfgStr.DEFAULT.value, None)
        if scale_method_config_default is not None:
            logger.trace("Parsing scale method: found 'default' configuration.")
            weight_scale_method_default = scale_method_config_default.get(CfgStr.WEIGHT.value, {})
            activation_scale_method_default = scale_method_config_default.get(CfgStr.ACTIVATION.value, {})
            scale_method_config[CfgStr.DEFAULT] = {
                CfgStr.WEIGHT: dict_to_scale_method_config(weight_scale_method_default),
                CfgStr.ACTIVATION: dict_to_scale_method_config(activation_scale_method_default),
            }
            logger.trace(f"Scale method config found for default: {scale_method_config[CfgStr.DEFAULT]}")
        else:
            logger.trace("Parsing scale method: missing 'default' configuration, raising ValueError.")
            raise ValueError("Scale method config should contain a default scale method.")
        def process_scale_method_keys(key):
            scale_method_config_items = scale_method.get(key.value, None)
            if scale_method_config_items is not None:
                logger.trace(f"Parsing scale method: found override section '{key.name}'.")
                scale_method_config[key] = {}
                for item_key, scale_method_config_item in scale_method_config_items.items():
                    logger.trace(f"Parsing scale method: processing {key.name} override for '{item_key}'.")
                    if item_key not in scale_method_config[key]:
                        scale_method_config[key][item_key] = {}
                    weight_scale_method = scale_method_config_item.get(CfgStr.WEIGHT.value, {})
                    activation_scale_method = scale_method_config_item.get(CfgStr.ACTIVATION.value, {})
                    scale_method_config[key][item_key][CfgStr.WEIGHT] = dict_to_scale_method_config(
                        weight_scale_method, scale_method_config[CfgStr.DEFAULT][CfgStr.WEIGHT]
                    )
                    scale_method_config[key][item_key][CfgStr.ACTIVATION] = dict_to_scale_method_config(
                        activation_scale_method, scale_method_config[CfgStr.DEFAULT][CfgStr.ACTIVATION]
                    )
                    logger.trace(f"Scale method config found for {key.name} '{item_key}': {scale_method_config[key][item_key]}")
        # create scale methods per nodes, layers, and layer types
        process_scale_method_keys(CfgStr.NODES)
        process_scale_method_keys(CfgStr.LAYERS)
        process_scale_method_keys(CfgStr.LAYER_TYPES)
    else:
        logger.trace("Parsing scale method: invalid config type, raising ValueError.")
        raise ValueError("Invalid scale method config. It should be either a string or a dictionary.")
    logger.trace("Parsing scale method: finished parsing configuration.")
    return scale_method_config


def validate_and_populate_scale_method(scale_method_dict):
    """
    Validates all scale methods (default, nodes, layers, layer_types)
    according to supported scale methods in ScaleMethodString and
    populates their params from scale_method_config_mapping as reference.
    """
    def update_params(target_cfg, ref_cfg):
        for key in [CfgStr.WEIGHT, CfgStr.ACTIVATION]:
            if key in target_cfg and key in ref_cfg:
                ref_params = getattr(ref_cfg[key], "params", {})
                tgt_params = getattr(target_cfg[key], "params", {})
                if ref_params:
                    for param_key, param_val in ref_params.items():
                        if param_key not in tgt_params or tgt_params[param_key] is None:
                            tgt_params[param_key] = param_val
                    target_cfg[key].params = tgt_params

    for key in [CfgStr.DEFAULT, CfgStr.NODES, CfgStr.LAYERS, CfgStr.LAYER_TYPES]:
        if key in scale_method_dict:
            configs = scale_method_dict[key]
            # For nodes/layers/layer_types, configs is a dict; for default, it's a config dict
            if isinstance(configs, dict) and key is not CfgStr.DEFAULT:
                for subkey, config in configs.items():
                    ref_method = get_scale_method_from_config(config)
                    if ref_method is None:
                        raise ValueError(
                            f"Unsupported config: scale method config for {key}[{subkey}] = {config} is not supported."
                        )
                    ref_cfg = scale_method_config_mapping[ref_method]
                    update_params(config, ref_cfg)
            else:
                ref_method = get_scale_method_from_config(configs)
                if ref_method is None:
                    raise ValueError(
                        f"Unsupported config: scale method config for {key} = {configs} is not supported."
                    )
                ref_cfg = scale_method_config_mapping[ref_method]
                update_params(configs, ref_cfg)