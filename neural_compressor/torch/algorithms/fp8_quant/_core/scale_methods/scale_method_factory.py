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
from enum import Enum, auto

from .round_scales_function import *
from ..common import get_device_type_for_scales
from .scales_method import *
from ...utils.logger import logger
from .scale_method_config import ScaleRoundMethod, ScaleValueType, ScaleGranularity, CfgStr

class QuantTensorName(Enum):
    INPUT = auto()
    WEIGHT_OUT_CH = auto()
    WEIGHT_IN_CH = auto()
    OUTPUT = auto()


HW_ALIGNED_ROUND_METHODS = [ScaleRoundMethod.HW_ALIGNED, ScaleRoundMethod.HW_ALIGNED_FIXED]

class ScaleMethodFactory:
    def __init__(self, config, params, mod, op_type):
        # config is dict mapping weight and activation to their ScaleMethodConfig(scale.py) which defines ScaleMethodString
        # params is `get_hqt_config(model).cfg["scale_params"]`, we copy in order not to damage it for other modules
        self.params = params.copy()
        self.mod = mod
        self.device_for_scales = get_device_type_for_scales(mod)
        self.op_type = op_type

        input_scale_method_config = config[CfgStr.ACTIVATION]
        weight_scale_method_config = config[CfgStr.WEIGHT]
        output_scale_method_config = input_scale_method_config

        self.scale_method_config_map = {
            QuantTensorName.INPUT: input_scale_method_config,
            QuantTensorName.WEIGHT_OUT_CH: weight_scale_method_config,
            QuantTensorName.WEIGHT_IN_CH: weight_scale_method_config,
            QuantTensorName.OUTPUT: output_scale_method_config,
        }

        logger.trace("%s %s", self.__class__.__name__, self.__dict__)

    def get_scale_method(self, tensor_type, is_dynamic=False):
        backoff = 1.0 if is_dynamic else self.scale_method_config_map[tensor_type].backoff
        scale_round_method = self.scale_method_config_map[tensor_type].rounding_method
        scale_value_type = self.scale_method_config_map[tensor_type].scale_value_type
        scale_granularity = self.scale_method_config_map[tensor_type].granularity

        # create scale_round_method according to ScaleRoundMethod enum name
        scale_round_method_name = scale_round_method.name
        if scale_round_method in HW_ALIGNED_ROUND_METHODS:
            # HW_ALIGNED rounding methods use device for scales
            scale_round_method = scale_round_method_registry[scale_round_method_name](self.device_for_scales)
        else:
            scale_round_method = scale_round_method_registry[scale_round_method_name]()

        logger.trace(
            "get_scale_method backoff=%s scale_round_method=%s scale_value_type=%s scale_granularity=%s op_type=%s is_dynamic=%s",
            backoff,
            scale_round_method,
            scale_value_type,
            scale_granularity,
            self.op_type,
            is_dynamic,
        )

        match (scale_value_type, scale_granularity, tensor_type, self.op_type):
            ## dummy
            case (ScaleValueType.DUMMY_SCALES, _, _, _) if tensor_type not in [QuantTensorName.WEIGHT_IN_CH]:
                return DummyScales(scale_round_method, self.params, self.device_for_scales)
            ## fixed value
            case (ScaleValueType.FIXED_VALUE, _, _, _) if tensor_type not in [QuantTensorName.WEIGHT_IN_CH]:
                return FixedScale(scale_round_method, self.params, self.device_for_scales)
            ## output max abs and opt, for linear and matmul
            case (_, _, QuantTensorName.OUTPUT, _) \
                if self.op_type in {"linear", "matmul"}:
                    if scale_value_type in {ScaleValueType.MAXABS, ScaleValueType.OPT}:
                        return MulAdditionalScales(scale_round_method, self.params, self.device_for_scales)
            ## maxabs/opt in channel PTS
            case (_, ScaleGranularity.PTS, QuantTensorName.WEIGHT_IN_CH, _):
                return None
            case (ScaleValueType.MAXABS, ScaleGranularity.PTS, _, _):
                if is_dynamic:
                    return MaxAbsDynamicPts(scale_round_method, self.params, self.device_for_scales, backoff)
                return MaxAbsPts(scale_round_method, self.params, self.device_for_scales, backoff)
            ## maxabs/opt in channel PCS
            case (_, ScaleGranularity.PCS, QuantTensorName.WEIGHT_IN_CH, _)\
                if scale_value_type in {ScaleValueType.MAXABS, ScaleValueType.OPT}:
                in_channel_size = self.mod.weight.shape[1]
                return InputChannelScale(scale_round_method, self.params, self.device_for_scales, in_channel_size)
            ## maxabs PCS
            case (ScaleValueType.MAXABS, ScaleGranularity.PCS, _, _):
                if is_dynamic:
                    return MaxAbsDynamicPcs(scale_round_method, self.params, self.device_for_scales, backoff)
                return MaxAbsPcs(scale_round_method, self.params, self.device_for_scales, backoff)
            ## opt PTS
            case (ScaleValueType.OPT, ScaleGranularity.PTS, _, _):
                opt_list_of_scales = self.scale_method_config_map[tensor_type].params["weight_scales"]
                return OptScalesPts(scale_round_method, opt_list_of_scales, self.params, self.device_for_scales, backoff)
            case (ScaleValueType.OPT, ScaleGranularity.PCS, _, _):
                opt_list_of_scales = self.scale_method_config_map[tensor_type].params["weight_scales"]
                return OptScalesPcs(scale_round_method, opt_list_of_scales, self.params, self.device_for_scales, backoff)
            case _:
                raise NotImplementedError("the config: scale_round_method: " + \
                                          str(scale_round_method) +
                                          ", scale_value_type: " + str(scale_value_type) +
                                          ", scale_granularity: " + str(scale_granularity) +
                                          " in tensor: " + str(tensor_type) +
                                          " op type "  + self.op_type + " not implemented ")