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

import torch
from ..fp_utils import get_fp8_hw_alligned_scales
from ...utils.logger import logger
from enum import Enum, auto
import re
import json

class ScaleMethodString(Enum):
    UNIT_SCALE = auto()
    HW_ALIGNED_SINGLE_SCALE = auto()
    MAXABS_HW = auto()
    MAXABS_POW2 = auto()
    ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2 = auto()
    ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2 = auto()
    ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2 = auto()
    ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2 = auto()
    MAXABS_HW_OPT_WEIGHT = auto()
    MAXABS_POW2_OPT_WEIGHT = auto()
    MAXABS_ARBITRARY = auto()
    ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW = auto()

class ScaleGranularity(Enum):
    PTS = auto()
    PCS = auto()

class ScaleValueType(Enum):
    MAXABS = auto()
    FIXED_VALUE = auto()
    OPT = auto()
    DUMMY_SCALES = auto()

class ScaleRoundMethod(Enum):
    IDENTITY = auto()
    POW2 = auto()
    HW_ALIGNED = auto()
    HW_ALIGNED_FIXED = auto()
    SCALE_UNIT = auto()

class CfgStr(Enum):
    ACTIVATION = "activation"
    WEIGHT = "weight"
    DEFAULT = "default"
    NODES= "nodes"
    LAYERS= "layers"
    LAYER_TYPES= "layer_types"
    GRANULARITY = "granularity"
    SCALE_VALUE_TYPE = "scale_value_type"
    ROUNDING_METHOD = "rounding_method"
    BACKOFF = "backoff"
    PARAMS = "params"
    LAYERS_DOT_PATTERN= r"layers\.(\d+)"
    LAYERS_SLASH_PATTERN= r"layers/(\d+)"

class ScaleMethodConfig:
    def __init__(self, 
                 granularity=ScaleGranularity.PTS,
                 scale_value_type=ScaleValueType.MAXABS,
                 rounding_method=ScaleRoundMethod.IDENTITY,
                 backoff=1.0,
                 params=None):
        self.granularity = granularity
        self.scale_value_type = scale_value_type
        self.rounding_method = rounding_method
        self.backoff = backoff
        self.params = params if params is not None else {}

    def __repr__(self):
        return f"ScaleMethodConfig(granularity={self.granularity}, scale_value_type={self.scale_value_type}, rounding_method={self.rounding_method}, backoff={self.backoff}, params={self.params})"

    def __hash__(self):
        # TODO: For supporting a user custom ScaleMethodConfig, we need to include the backoff and the params in the hash.
        return hash((
            self.granularity,
            self.scale_value_type,
            self.rounding_method
        ))
    
    def __eq__(self, other):
        if not isinstance(other, ScaleMethodConfig):
            return False
        
        # Only check the three fields that define uniqueness
        return (self.granularity == other.granularity and
                self.scale_value_type == other.scale_value_type and
                self.rounding_method == other.rounding_method)

scale_method_config_mapping = {
    ScaleMethodString.UNIT_SCALE:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(scale_value_type=ScaleValueType.FIXED_VALUE, rounding_method= ScaleRoundMethod.SCALE_UNIT),
        CfgStr.ACTIVATION: ScaleMethodConfig(scale_value_type=ScaleValueType.FIXED_VALUE, rounding_method= ScaleRoundMethod.SCALE_UNIT)
    },
    ScaleMethodString.HW_ALIGNED_SINGLE_SCALE:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(scale_value_type=ScaleValueType.FIXED_VALUE, rounding_method= ScaleRoundMethod.HW_ALIGNED_FIXED),
        CfgStr.ACTIVATION: ScaleMethodConfig(scale_value_type=ScaleValueType.FIXED_VALUE, rounding_method= ScaleRoundMethod.HW_ALIGNED_FIXED)
    },
    ScaleMethodString.MAXABS_HW:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.5),
        CfgStr.ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.25)
    },
    ScaleMethodString.MAXABS_POW2:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(rounding_method= ScaleRoundMethod.POW2, backoff= 0.5),
        CfgStr.ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.POW2, backoff= 0.25)
    },
    ScaleMethodString.MAXABS_ARBITRARY:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(backoff= 0.5),
        CfgStr.ACTIVATION: ScaleMethodConfig(backoff= 0.25)
    },
    ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.5),
        CfgStr.ACTIVATION: ScaleMethodConfig(granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2)
    },
    ScaleMethodString.MAXABS_HW_OPT_WEIGHT:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(scale_value_type= ScaleValueType.OPT, rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.5, params={"weight_scales": get_fp8_hw_alligned_scales(torch.float8_e4m3fn)}),
        CfgStr.ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.25)
    },
    ScaleMethodString.MAXABS_POW2_OPT_WEIGHT:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(scale_value_type= ScaleValueType.OPT, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5, params={"weight_scales": [2.0**s for s in range(-10, 10)]}),
        CfgStr.ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.POW2, backoff= 0.25)
    },
    ScaleMethodString.ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5),
        CfgStr.ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.25)
    },
    ScaleMethodString.ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(scale_value_type = ScaleValueType.OPT, granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5, params={"weight_scales": [2.0**s for s in range(-3, 5)]}),
        CfgStr.ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.25)
    },
    ScaleMethodString.ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5),
        CfgStr.ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.POW2, backoff= 0.25)
    },
    ScaleMethodString.ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2:
    {
        CfgStr.WEIGHT:     ScaleMethodConfig(scale_value_type = ScaleValueType.OPT, granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5, params={"weight_scales": [2.0**s for s in range(-3, 5)]}),
        CfgStr.ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.POW2, backoff= 0.25)
    },
}

reverse_scale_method_mapping = {
    (configs[CfgStr.WEIGHT], configs[CfgStr.ACTIVATION]): scale_method
    for scale_method, configs in scale_method_config_mapping.items()
}

def get_scale_method_from_config(config_dict):
    """
    Given a dictionary with 'weight' and 'activation' ScaleMethodConfig objects,
    returns the corresponding ScaleMethodString if it matches a known configuration,
    otherwise returns None.
    """
    weight_config = config_dict[CfgStr.WEIGHT]
    activation_config = config_dict[CfgStr.ACTIVATION]
    config_key = (weight_config, activation_config)
    return reverse_scale_method_mapping.get(config_key, None)

def dict_to_scale_method_config(scale_method, scale_method_config_default=None):
    """
    Converts a dictionary specifying scale method parameters (granularity, scale_value_type,
    rounding_method, backoff, params) into a ScaleMethodConfig object.
    Tries to get each parameter from scale_method, then from scale_method_config_default, then uses the hardcoded default.
    Always converts string enum values to enum objects.
    """
    def get_param(key, default_value):
        if key in scale_method:
            return scale_method[key]
        elif scale_method_config_default is not None and hasattr(scale_method_config_default, key):
            return getattr(scale_method_config_default, key)
        else:
            return default_value
    scale_method_config = ScaleMethodConfig(
        granularity=get_param(CfgStr.GRANULARITY.value, ScaleGranularity.PTS),
        scale_value_type=get_param(CfgStr.SCALE_VALUE_TYPE.value, ScaleValueType.MAXABS),
        rounding_method=get_param(CfgStr.ROUNDING_METHOD.value, ScaleRoundMethod.IDENTITY),
        backoff=get_param(CfgStr.BACKOFF.value, 1.0),
        params=get_param(CfgStr.PARAMS.value, None),
    )
    return scale_method_config

def find_node_scale_method_config(scale_method_config, node_name, layer_type):
    """
    Finds and returns the appropriate scale method config for a given node name and layer type.
    Checks nodes, layers (by index), and layer_types in order, falling back to the default if not found.
    """
    # Try node-specific config
    scale_method_config_nodes = scale_method_config.get(CfgStr.NODES, None)
    if scale_method_config_nodes and scale_method_config_nodes.get(node_name, None) is not None:
        logger.trace(f"Scale method config node specific found for node: {node_name}")
        return scale_method_config_nodes[node_name]

    # Try layer index config (dot or slash pattern)
    match = re.search(CfgStr.LAYERS_DOT_PATTERN.value, node_name) # Match ".layers.<number>."
    if not match:
        match = re.search(CfgStr.LAYERS_SLASH_PATTERN.value, node_name) # Match "/layers/<number>/"
    if match:
        layer_index = match.group(1)
        scale_method_config_layers = scale_method_config.get(CfgStr.LAYERS, None)
        if scale_method_config_layers and scale_method_config_layers.get(layer_index, None) is not None:
            logger.trace(f"Scale method config found for layer index: {layer_index} for node: {node_name}")
            return scale_method_config_layers[layer_index]

    # Try layer type config
    scale_method_config_layer_types = scale_method_config.get(CfgStr.LAYER_TYPES, None)
    if scale_method_config_layer_types and scale_method_config_layer_types.get(layer_type, None) is not None:
        logger.trace(f"Scale method config found for layer type: {layer_type} for node: {node_name}")
        return scale_method_config_layer_types[layer_type]

    # Fallback to default config
    logger.trace(f"Using default scale method config for node: {node_name}, layer_type: {layer_type}")
    return scale_method_config[CfgStr.DEFAULT]

def _check_config(
    config,
    scale_value_type_weight=None,
    scale_value_type_activation=None,
    rounding_method_weight=None,
    rounding_method_activation=None,
    granularity_weight=None,
    granularity_activation=None,
):
    """
    Checks if the given config matches all specified criteria for weight and activation
    (scale_value_type, rounding_method, granularity). Returns True if all match or are None.
    """
    checks = [
        (scale_value_type_weight is None or config[CfgStr.WEIGHT].scale_value_type == scale_value_type_weight,
         f"scale_value_type_weight: expected {scale_value_type_weight}, got {config[CfgStr.WEIGHT].scale_value_type}"),
        (scale_value_type_activation is None or config[CfgStr.ACTIVATION].scale_value_type == scale_value_type_activation,
         f"scale_value_type_activation: expected {scale_value_type_activation}, got {config[CfgStr.ACTIVATION].scale_value_type}"),
        (rounding_method_weight is None or config[CfgStr.WEIGHT].rounding_method == rounding_method_weight,
         f"rounding_method_weight: expected {rounding_method_weight}, got {config[CfgStr.WEIGHT].rounding_method}"),
        (rounding_method_activation is None or config[CfgStr.ACTIVATION].rounding_method == rounding_method_activation,
         f"rounding_method_activation: expected {rounding_method_activation}, got {config[CfgStr.ACTIVATION].rounding_method}"),
        (granularity_weight is None or config[CfgStr.WEIGHT].granularity == granularity_weight,
         f"granularity_weight: expected {granularity_weight}, got {config[CfgStr.WEIGHT].granularity}"),
        (granularity_activation is None or config[CfgStr.ACTIVATION].granularity == granularity_activation,
         f"granularity_activation: expected {granularity_activation}, got {config[CfgStr.ACTIVATION].granularity}"),
    ]
    for passed, msg in checks:
        if not passed:
            logger.trace(f"Scale method config validation failed: {msg}")
            return False
    return True

def check_scale_method_fields(
    scale_method_dict,
    scale_method = None,
    scale_value_type_weight=None,
    scale_value_type_activation=None,
    rounding_method_weight=None,
    rounding_method_activation=None,
    granularity_weight=None,
    granularity_activation=None,
    reducer=all,
):
    """
    Iterates over all scale method configs (default, nodes, layers, layer_types) in the given dict and compares them to the given conditions.
    For each config, verifies that the specified fields match the provided values (if not None).
    If scale_method is given, uses its config as reference for the checks.
    The reducer argument determines if all (default) or any must match (use reducer=any).
    Returns the result of the reducer over all configs.
    """
    if scale_method is not None:
        scale_method_config = scale_method_config_mapping[scale_method]
        scale_method_config_weight = scale_method_config[CfgStr.WEIGHT]
        scale_method_config_activation = scale_method_config[CfgStr.ACTIVATION]
        scale_value_type_weight = scale_method_config_weight.scale_value_type
        scale_value_type_activation = scale_method_config_activation.scale_value_type
        rounding_method_weight = scale_method_config_weight.rounding_method
        rounding_method_activation = scale_method_config_activation.rounding_method
        granularity_weight = scale_method_config_weight.granularity
        granularity_activation = scale_method_config_activation.granularity

    configs_to_check = []
    for key in [CfgStr.DEFAULT, CfgStr.NODES, CfgStr.LAYERS, CfgStr.LAYER_TYPES]:
        if key in scale_method_dict:
            configs = scale_method_dict[key]
            if isinstance(configs, dict) and key is not CfgStr.DEFAULT:
                configs_to_check.extend(configs.values())
            else:
                configs_to_check.append(configs)
    return reducer(_check_config(config, scale_value_type_weight, scale_value_type_activation, rounding_method_weight, rounding_method_activation, granularity_weight, granularity_activation) for config in configs_to_check)

def dump_scale_method_config_by_mod_map(scale_method_config_by_mod_map, filename="scale_method_config_by_mod_map.json"):
    """
    Dumps the scale_method_config_by_mod_map to a JSON file.
    """
    serializable = {}
    for mod_name, cfgs in scale_method_config_by_mod_map.items():
        serializable[mod_name] = {}
        for key, cfg in cfgs.items():
            serializable[mod_name][ key.name if hasattr(key, "name") else str(key) ] = {
                    CfgStr.GRANULARITY.value: cfg.granularity.name,
                    CfgStr.SCALE_VALUE_TYPE.value: cfg.scale_value_type.name,
                    CfgStr.ROUNDING_METHOD.value: cfg.rounding_method.name,
                    CfgStr.BACKOFF.value: cfg.backoff,
                    CfgStr.PARAMS.value: cfg.params,
            }
    with open(filename, "w") as f:
        json.dump(serializable, f, indent=2)

def load_scale_method_config_by_mod_map(filename):
    """
    Loads scale_method_config_by_mod_map from a JSON file and reconstructs the original structure.
    Returns: dict[str, dict[CfgStr, ScaleMethodConfig]]
    """
    with open(filename, "r") as f:
        data = json.load(f)
    result = {}
    for mod_name, cfgs in data.items():
        result[mod_name] = {}
        for key, cfg_dict in cfgs.items():
            enum_key = CfgStr[key.upper()]
            result[mod_name][enum_key] = ScaleMethodConfig(
                granularity=ScaleGranularity[cfg_dict[CfgStr.GRANULARITY.value]],
                scale_value_type=ScaleValueType[cfg_dict[CfgStr.SCALE_VALUE_TYPE.value]],
                rounding_method=ScaleRoundMethod[cfg_dict[CfgStr.ROUNDING_METHOD.value]],
                backoff=cfg_dict.get(CfgStr.BACKOFF.value, 1.0),
                params=cfg_dict.get(CfgStr.PARAMS.value, None),
            )
    return result


