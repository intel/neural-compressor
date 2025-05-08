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
# TODO [SW-217813]: support dynamic quantization in all ops and remove supported_dynamic_ops
from ..._quant_common.quant_config import is_supported_dynamic_op
from ..common import get_device_type_for_scales
from .scales_method import *
from ...utils.logger import logger
from ..._quant_common.quant_config import get_hqt_config


class QuantTensorName(Enum):
    INPUT = auto()
    WEIGHT_OUT_CH = auto()
    WEIGHT_IN_CH = auto()
    OUTPUT = auto()

class ScaleGranularity(Enum):
    PTS = auto()
    PCS = auto()

class ScaleValueType(Enum):
    MAXABS = auto()
    FIXED_VALUE = auto()
    OPT = auto()
    SMOOTHQUANT_MAXABS = auto()
    SMOOTHQUANT_OPT = auto()
    SMOOTHQUANT_WEAK = auto()
    DUMMY_SCALES = auto()
    DYNAMIC = auto()


def parse_rounding_method(config, device_for_scales):
    round_method = ScaleIdentity()
    if "single" in config and "hw" in config:
        round_method = ScaleHwAlignedFixed(device_for_scales)
    elif "unit" in config:
        round_method = ScaleUnit()
    elif "hw" in config:
        round_method = ScaleToHwAligned(device_for_scales)
    elif "pow" in config:
        round_method = ScaleToPow2()
    logger.trace(f"parse_rounding_method {config=} {round_method=}")
    return round_method


def parse_tensor_granularity(config):
    scale_granularity = ScaleGranularity.PTS
    if "pcs" in config or "smoothquant" in config:
        scale_granularity = ScaleGranularity.PCS
    return scale_granularity

# TODO [SW-217813]: support dynamic quantization in all ops and remove op_type
def parse_tensor_scale_value_type(config, op_type):
    scale_value_type = ScaleValueType.MAXABS
    if "unit" in config or "single" in config:
        scale_value_type = ScaleValueType.FIXED_VALUE
    elif "smoothquant" in config and "maxabs" in config:
        scale_value_type = ScaleValueType.SMOOTHQUANT_MAXABS
    elif "smoothquant" in config and "opt" in config:
        scale_value_type = ScaleValueType.SMOOTHQUANT_OPT
    elif "smoothquant" in config and "weak" in config:
        scale_value_type = ScaleValueType.SMOOTHQUANT_WEAK
    elif "opt" in config:
        scale_value_type = ScaleValueType.OPT
    elif "dummy" in config:
        scale_value_type = ScaleValueType.DUMMY_SCALES
    elif "dyn" in config and is_supported_dynamic_op(op_type):
        scale_value_type = ScaleValueType.DYNAMIC
    logger.trace(f"parse_tensor_scale_value_type {config=} {scale_value_type=}")
    return  scale_value_type


class ScaleMethodFactory:

    ## config string example: "act_maxabs_pts_weight_opt_pts_hw", round_method = pow2_hw, scale_value_type = maxabs, granularity = pts
    # all config strings in scale.py: scale_method_mapping
    def __init__(self, config, params, mod, op_type):
        # params is `get_hqt_config(model).cfg["scale_params"]`, we copy in order not to damage it for other modules
        self.params = params.copy()
        self.mod = mod
        self.device_for_scales = get_device_type_for_scales(mod)
        self.op_type = op_type
        if op_type == "row_parallel_linear" and get_hqt_config(mod).cfg["row_parallel_linear_allreduce_quantization"]:
             self.params["output_backoff"] = 1


        if "weight" in config and "smoothquant" not in config:
            parts = config.split("weight", 1)
            config_act, config_weight = parts[0].strip(), parts[1].strip()
        else:
            config_act = config_weight = config.strip()
        config_out = config_act
        self.scale_round_method_map = {QuantTensorName.INPUT: parse_rounding_method(config_act, self.device_for_scales),
                                 QuantTensorName.WEIGHT_OUT_CH: parse_rounding_method(config_weight, self.device_for_scales),
                                 QuantTensorName.WEIGHT_IN_CH: parse_rounding_method(config_weight, self.device_for_scales),
                                 QuantTensorName.OUTPUT: parse_rounding_method(config_out, self.device_for_scales)}
        self.scale_granularity_map = {QuantTensorName.INPUT: parse_tensor_granularity(config_act),
                                      QuantTensorName.WEIGHT_OUT_CH: parse_tensor_granularity(config_weight),
                                      QuantTensorName.WEIGHT_IN_CH: parse_tensor_granularity(config_weight),
                                      QuantTensorName.OUTPUT: parse_tensor_granularity(config_out)}
        # TODO [SW-217813]: support dynamic quantization in all ops and remove op_type
        self.scale_value_type_map = {QuantTensorName.INPUT: parse_tensor_scale_value_type(config_act, self.op_type),
                                     QuantTensorName.WEIGHT_OUT_CH: parse_tensor_scale_value_type(config_weight, self.op_type),
                                     QuantTensorName.WEIGHT_IN_CH: parse_tensor_scale_value_type(config_weight, self.op_type),
                                     QuantTensorName.OUTPUT: parse_tensor_scale_value_type(config_out, self.op_type)}
        self.scale_backoff_map = {QuantTensorName.INPUT: self.params.get("input_backoff", 1.0),
                                  QuantTensorName.WEIGHT_IN_CH: self.params.get("weight_backoff", 1.0),
                                  QuantTensorName.WEIGHT_OUT_CH: self.params.get("weight_backoff", 1.0),
                                  QuantTensorName.OUTPUT: self.params.get("output_backoff", self.params.get("input_backoff", 1.0)),} # get output_backoff, if doesn't exists use input_backoff, if doesn't exists use 1
        logger.debug("%s %s".format(self.__class__.__name__, self.__dict__))

    ## TODO remove after SW-217369
    ## config string example: "act_maxabs_pts_weight_opt_pts_hw", round_method = pow2_hw, scale_value_type = maxabs, granularity = pts
    # all config strings in scale.py: scale_method_mapping
    # returns MaxAbsPts obj with pow2_hw as scale_round_method
    def get_scale_method(self, tensor_name):
        backoff =  self.scale_backoff_map[tensor_name]
        scale_round_method = self.scale_round_method_map[tensor_name]
        scale_value_type = self.scale_value_type_map[tensor_name]
        scale_granularity = self.scale_granularity_map[tensor_name]

        match (scale_value_type, scale_granularity, tensor_name, self.op_type):
            ## dummy
            case (ScaleValueType.DUMMY_SCALES, _, _, _) if tensor_name not in [QuantTensorName.WEIGHT_IN_CH]:
                return DummyScales(scale_round_method, self.params, self.device_for_scales)
            ## fixed value
            case (ScaleValueType.FIXED_VALUE, _, _, _) if tensor_name not in [QuantTensorName.WEIGHT_IN_CH]:
                return FixedScale(scale_round_method, self.params, self.device_for_scales)
            ## output max abs and opt, for linear and matmul
            case (_, _, QuantTensorName.OUTPUT, _) \
                if self.op_type in {"linear", "matmul"}:
                    if scale_value_type in {ScaleValueType.MAXABS, ScaleValueType.OPT}:
                        return MulAdditionalScales(scale_round_method, self.params, self.device_for_scales)
                    if scale_value_type == ScaleValueType.DYNAMIC:
                        return MulAdditionalDynamicScales(scale_round_method, self.params, self.device_for_scales)
            ## maxabs/opt in channel PTS
            case (_, ScaleGranularity.PTS, QuantTensorName.WEIGHT_IN_CH, _) \
                if scale_value_type not in {ScaleValueType.SMOOTHQUANT_OPT, ScaleValueType.SMOOTHQUANT_MAXABS}:
                return None
            case (ScaleValueType.MAXABS, ScaleGranularity.PTS, _, _):
                return MaxAbsPts(scale_round_method, self.params, self.device_for_scales, backoff)
            ## maxabs/opt in channel PCS
            case (_, ScaleGranularity.PCS, QuantTensorName.WEIGHT_IN_CH, _)\
                if scale_value_type in {ScaleValueType.MAXABS, ScaleValueType.OPT}:
                in_channel_size = self.mod.weight.shape[1]
                return InputChannelScale(scale_round_method, self.params, self.device_for_scales, in_channel_size)
            ## maxabs PCS
            case (ScaleValueType.MAXABS, ScaleGranularity.PCS, _, _):
                return MaxAbsPcs(scale_round_method, self.params, self.device_for_scales, backoff)
            ## opt PTS
            case (ScaleValueType.OPT, ScaleGranularity.PTS, _, _):
                opt_list_of_scales = self.params["weight_scales"]
                return OptScalesPts(scale_round_method, opt_list_of_scales, self.params, self.device_for_scales, backoff)
            case (ScaleValueType.OPT, ScaleGranularity.PCS, _, _):
                opt_list_of_scales = self.params["weight_scales"]
                return OptScalesPcs(scale_round_method, opt_list_of_scales, self.params, self.device_for_scales, backoff)
            ## smooth quant
            case (_, ScaleGranularity.PCS, QuantTensorName.WEIGHT_IN_CH, _) \
                if scale_value_type in {ScaleValueType.SMOOTHQUANT_OPT, ScaleValueType.SMOOTHQUANT_MAXABS}:
                return WeightIchSmoothQuant(scale_round_method, self.params, self.device_for_scales)
            case (_,  ScaleGranularity.PCS, QuantTensorName.OUTPUT, _) \
                if scale_value_type in {ScaleValueType.SMOOTHQUANT_OPT, ScaleValueType.SMOOTHQUANT_MAXABS} \
                   and self.op_type in {"linear", "matmul"}:
                return UseFirstAdditionalScales(scale_round_method, self.params, self.device_for_scales)
            ## SMOOTHQUANT_MAXABS input and weight out channel
            case (ScaleValueType.SMOOTHQUANT_MAXABS, ScaleGranularity.PCS, QuantTensorName.WEIGHT_OUT_CH, _):
                return MaxAbsPcs(scale_round_method, self.params, self.device_for_scales, backoff)
            case (ScaleValueType.SMOOTHQUANT_MAXABS, ScaleGranularity.PCS, QuantTensorName.INPUT, _):
                return InputSmoothQuantMaxAbs(scale_round_method, self.mod.weight, self.params, self.device_for_scales, backoff)
            ## SMOOTHQUANT_OPT input and weight out channel
            case (ScaleValueType.SMOOTHQUANT_OPT, _, QuantTensorName.WEIGHT_OUT_CH, _):
                opt_list_of_scales = self.params["transformed_weight_scales"]
                return OptScalesPcs(scale_round_method, opt_list_of_scales, self.params, self.device_for_scales, backoff)
            case (ScaleValueType.SMOOTHQUANT_OPT, _, QuantTensorName.INPUT, _):
                backoff_weight =  self.params.get("weight_backoff", 1)
                return InputSmoothQuantOpt(scale_round_method, self.mod.weight, self.params, self.device_for_scales, backoff, backoff_weight)
            case (ScaleValueType.DYNAMIC, ScaleGranularity.PCS, QuantTensorName.INPUT, _):
                return MaxAbsDynamicPcs(scale_round_method, self.params, self.device_for_scales, backoff)
            case _:
                raise NotImplementedError("the config: scale_round_method: " + \
                                          str(scale_round_method) +
                                          ", scale_value_type: " + str(scale_value_type) +
                                          ", scale_granularity: " + str(scale_granularity) +
                                          " in tensor: " + str(tensor_name) +
                                          " op type "  + self.op_type + " not implemented ")
