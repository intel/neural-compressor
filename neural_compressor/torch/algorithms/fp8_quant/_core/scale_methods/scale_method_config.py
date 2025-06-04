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
from .scale_method_factory import ScaleGranularity, ScaleValueType, ScaleRoundMethod
from ..fp_utils import get_fp8_hw_alligned_scales
from enum import Enum, auto

class ScaleMethodString(Enum):
    UNIT_SCALE = auto()
    HW_ALIGNED_SINGLE_SCALE = auto()
    MAXABS_HW = auto()
    MAXABS_POW2 = auto()
    SMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2 = auto()
    WEAKSMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2 = auto()
    ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2 = auto()
    ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2 = auto()
    ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2 = auto()
    ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2 = auto()
    SMOOTHQUANT_OPT = auto()
    MAXABS_HW_OPT_WEIGHT = auto()
    MAXABS_POW2_OPT_WEIGHT = auto()
    MAXABS_ARBITRARY = auto()
    ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW = auto()
    ACT_MAXABS_PTS_POW2_WEIGHT_MAXABS_PTS_POW2_HW = auto()

ACTIVATION = "activation"
WEIGHT = "weight"
DEFAULT = "default"

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
        WEIGHT:     ScaleMethodConfig(scale_value_type=ScaleValueType.FIXED_VALUE, rounding_method= ScaleRoundMethod.SCALE_UNIT),
        ACTIVATION: ScaleMethodConfig(scale_value_type=ScaleValueType.FIXED_VALUE, rounding_method= ScaleRoundMethod.SCALE_UNIT)
    },
    ScaleMethodString.HW_ALIGNED_SINGLE_SCALE:
    {
        WEIGHT:     ScaleMethodConfig(scale_value_type=ScaleValueType.FIXED_VALUE, rounding_method= ScaleRoundMethod.HW_ALIGNED_FIXED),
        ACTIVATION: ScaleMethodConfig(scale_value_type=ScaleValueType.FIXED_VALUE, rounding_method= ScaleRoundMethod.HW_ALIGNED_FIXED)
    },
    ScaleMethodString.MAXABS_HW:
    {
        WEIGHT:     ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.5),
        ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.25)
    },
    ScaleMethodString.MAXABS_POW2:
    {
        WEIGHT:     ScaleMethodConfig(rounding_method= ScaleRoundMethod.POW2, backoff= 0.5),
        ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.POW2, backoff= 0.25)
    },
    ScaleMethodString.MAXABS_ARBITRARY:
    {
        WEIGHT:     ScaleMethodConfig(backoff= 0.5),
        ACTIVATION: ScaleMethodConfig(backoff= 0.25)
    },
    ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW:
    {
        WEIGHT:     ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.5),
        ACTIVATION: ScaleMethodConfig(granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2)
    },
    ScaleMethodString.ACT_MAXABS_PTS_POW2_WEIGHT_MAXABS_PTS_POW2_HW:
    {
        WEIGHT:     ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.5),
        ACTIVATION: ScaleMethodConfig(granularity= ScaleGranularity.PTS, rounding_method= ScaleRoundMethod.POW2)
    },
    ScaleMethodString.MAXABS_HW_OPT_WEIGHT:
    {
        WEIGHT:     ScaleMethodConfig(scale_value_type= ScaleValueType.OPT, rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.5, params={"weight_scales": get_fp8_hw_alligned_scales(torch.float8_e4m3fn)}),
        ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.25)
    },
    ScaleMethodString.MAXABS_POW2_OPT_WEIGHT:
    {
        WEIGHT:     ScaleMethodConfig(scale_value_type= ScaleValueType.OPT, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5, params={"weight_scales": [2.0**s for s in range(-10, 10)]}),
        ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.POW2, backoff= 0.25)
    },
    ScaleMethodString.ACT_MAXABS_HW_WEIGHTS_PCS_MAXABS_POW2:
    {
        WEIGHT:     ScaleMethodConfig(granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5),
        ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.25)
    },
    ScaleMethodString.SMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2:
    {
        WEIGHT:     ScaleMethodConfig(scale_value_type = ScaleValueType.SMOOTHQUANT_MAXABS, granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5),
        ACTIVATION: ScaleMethodConfig(scale_value_type = ScaleValueType.SMOOTHQUANT_MAXABS, granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.25, params={"alpha": 0.5})
    },
    ScaleMethodString.WEAKSMOOTHQUANT_WEIGHTS_OUTPUT_CHANNEL_MAXABS_POW2:
    {
        WEIGHT:     ScaleMethodConfig(scale_value_type = ScaleValueType.SMOOTHQUANT_WEAK, granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5),
        ACTIVATION: ScaleMethodConfig(scale_value_type = ScaleValueType.SMOOTHQUANT_WEAK, granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.25, params={"alpha": 0.5})
    },
    ScaleMethodString.ACT_MAXABS_HW_WEIGHTS_PCS_OPT_POW2:
    {
        WEIGHT:     ScaleMethodConfig(scale_value_type = ScaleValueType.OPT, granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5, params={"weight_scales": [2.0**s for s in range(-3, 5)]}),
        ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.HW_ALIGNED, backoff= 0.25)
    },
    ScaleMethodString.ACT_MAXABS_POW2_WEIGHTS_PCS_MAXABS_POW2:
    {
        WEIGHT:     ScaleMethodConfig(granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5),
        ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.POW2, backoff= 0.25)
    },
    ScaleMethodString.ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2:
    {
        WEIGHT:     ScaleMethodConfig(scale_value_type = ScaleValueType.OPT, granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5, params={"weight_scales": [2.0**s for s in range(-3, 5)]}),
        ACTIVATION: ScaleMethodConfig(rounding_method= ScaleRoundMethod.POW2, backoff= 0.25)
    },
    ScaleMethodString.SMOOTHQUANT_OPT: 
    {
        WEIGHT:     ScaleMethodConfig(scale_value_type = ScaleValueType.SMOOTHQUANT_OPT, granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.5, params={"transformed_weight_scales": [2.0**s for s in range(-3, 5)]}),
        ACTIVATION: ScaleMethodConfig(scale_value_type = ScaleValueType.SMOOTHQUANT_OPT, granularity= ScaleGranularity.PCS, rounding_method= ScaleRoundMethod.POW2, backoff= 0.25,  params={"alpha": 0.5})
    },
}

reverse_scale_method_mapping = {
    (configs[WEIGHT], configs[ACTIVATION]): scale_method
    for scale_method, configs in scale_method_config_mapping.items()
}

def get_scale_method_from_config(config_dict):
    # Find the matching ScaleMethodString given a dict with 'weight' and 'activation' ScaleMethodConfig objects.
    weight_config = config_dict[WEIGHT]
    activation_config = config_dict[ACTIVATION]
    config_key = (weight_config, activation_config)
    return reverse_scale_method_mapping.get(config_key, None)

def parse_scale_method_config(scale_method_config):
    # The scale_method_config can be provided as a ScaleMethodString enum or as a dictionary.
    # If provided as a dictionary, it can define the granularity, scale_value_type,
    # rounding_method, backoff, and params for both weight and activation.
    if isinstance(scale_method_config, ScaleMethodString):
        scale_method_config = {
            DEFAULT: scale_method_config_mapping[scale_method_config]
        }
        return scale_method_config
    elif isinstance(scale_method_config, dict):
        weight_scale_method = scale_method_config[DEFAULT].get(WEIGHT, {})
        activation_scale_method = scale_method_config[DEFAULT].get(ACTIVATION, {})
        scale_method_config = {
            DEFAULT: {
                WEIGHT: create_scale_method_config(weight_scale_method),
                ACTIVATION: create_scale_method_config(activation_scale_method),
            }
        }
    else:
        raise ValueError("Invalid scale method config. It should be either a ScaleMethodString enum or a dictionary.")
    return scale_method_config

def create_scale_method_config(scale_method):
    scale_method_config = ScaleMethodConfig(
        granularity= ScaleGranularity[scale_method.get("granularity", ScaleGranularity.PTS).upper()],
        scale_value_type= ScaleValueType[scale_method.get("scale_value_type", ScaleValueType.MAXABS).upper()],
        rounding_method= ScaleRoundMethod[scale_method.get("rounding_method", ScaleRoundMethod.IDENTITY).upper()],
        backoff= scale_method.get("backoff", 1.0),
        params= scale_method.get("params", None),
    )
    return scale_method_config