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
from abc import abstractmethod
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import get_hqt_config
from .scale_method_factory import ScaleMethodFactory, QuantTensorName
from .scales_method import QuantTensorType
from ..common import ModuleConfig
from ..quant_dequant import DequantOutput, QuantDequant, QuantDequantNone, QuantInput, scale_fcn


class BaseOpQuantizer:

    def __init__(self, config, mod, measurement, params, op_type):
        self.scales_method_factory = ScaleMethodFactory(config, params, mod, op_type)
        self.mod = mod
        self.params = params
        self.measurement = measurement
        self.inputs_scales_creators = []
        self.output_scales_creators = []
        self.params_scales_creators = []

    def get_module_configuration(self):
        scale_format = get_hqt_config(self.mod).cfg["scale_format"]
        use_qdq = get_hqt_config(self.mod).cfg["use_qdq"]
        fake_quant = get_hqt_config(self.mod).cfg["fake_quant"]
        lp_dtype = self.params["lp_dtype"]
        hp_dtype = self.params["hp_dtype"]
        return scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype

    @abstractmethod
    def get_scales_module_config(self) -> ModuleConfig:
        raise NotImplementedError("`get_scales_module_config` function is not implemented")

    @abstractmethod
    def scales_module_config_to_q_and_dq(self, module) -> ModuleConfig:
        raise NotImplementedError("`scales_module_config_to_q_and_dq` function is not implemented")

    def init_scales_from_module_config(self, module):
        for idx, input in enumerate(module.inputs):
            if self.inputs_scales_creators[idx].scale is None:
                self.inputs_scales_creators[idx].scale = input
        for idx, output in enumerate(module.outputs):
            if self.output_scales_creators[idx].scale is None:
                self.output_scales_creators[idx].scale = output
                
    def calc_input_scales(self, num_of_inputs):
        input_scales = []
        for i in  range(num_of_inputs):
            input_measurement = self.measurement.inputs[i] if self.measurement is not None else []
            input_scales.append(
                self.inputs_scales_creators[i].calc_scales(input_measurement, QuantTensorType.MEASUREMENTS))
        return input_scales

    def calc_output_scales(self):
        output_measurement = self.measurement.outputs[0] if self.measurement is not None else []
        output_scales = self.output_scales_creators[0].calc_scales(output_measurement, QuantTensorType.MEASUREMENTS)
        return (output_scales, )

    def init_input_config(self, scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant):
        if use_qdq or fake_quant:
            input_config = [
                QuantDequant(s_inv, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq)
                for s_inv in scales_inv
            ]
        else:
            input_config =  [
                QuantInput(s_inv, lp_dtype, hp_dtype, scale_format=scale_format) for s_inv in scales_inv
            ]
        return input_config

class LinearOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, module_type):
        super().__init__(config, mod, measurement, params, module_type)
        self.inputs_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.INPUT))
        self.weight_och_scale_calc = self.scales_method_factory.get_scale_method(QuantTensorName.WEIGHT_OUT_CH)
        self.weight_ich_scale_calc = self.scales_method_factory.get_scale_method(QuantTensorName.WEIGHT_IN_CH)
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))

    def get_scales_module_config(self):
        input_scales = self.calc_input_scales(num_of_inputs=1)
        output_measurement = self.measurement.outputs[0] if self.measurement is not None else []
        rescaled_weight = self.mod.weight
        if self.weight_ich_scale_calc is not None:
            weight_scales_in_ch = self.weight_ich_scale_calc.calc_scales(input_scales[0], QuantTensorType.CONST)
            rescaled_weight = scale_fcn(self.mod.weight, weight_scales_in_ch.reshape([1, -1]))
        weights_scales_out_ch = self.weight_och_scale_calc.calc_scales(rescaled_weight, QuantTensorType.CONST)
        params_config = {"weight": weights_scales_out_ch} if (
                self.weight_ich_scale_calc is None) \
                else {"weight": {0: weights_scales_out_ch, 1: weight_scales_in_ch}}
        output_scales = self.output_scales_creators[0].calc_scales(output_measurement, QuantTensorType.MEASUREMENTS,
                                                                   input0=weights_scales_out_ch, input1=input_scales[0])
        return ModuleConfig(
            input_scales,
            (output_scales,),
                params_config,
        )

    def init_weight_config(self, scales, scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant):
        if use_qdq:
            # to ensure the weights to be loaded to the device in fp8
            weight_config = [QuantInput(scales_inv, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq),
                             DequantOutput(scales, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq)]
        elif fake_quant:
            weight_config = [QuantDequant(scales_inv, lp_dtype, hp_dtype, scale_format=scale_format)]
        else:
            weight_config = [QuantInput(scales_inv, lp_dtype, hp_dtype, scale_format=scale_format)]
        return weight_config

    def init_weights_from_module(self, params_config):
        if isinstance(params_config, dict):
            self.weight_och_scale_calc.scale = params_config[0]
            self.weight_ich_scale_calc.scale = params_config[1]
        else:
            self.weight_och_scale_calc.scale = params_config


    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        self.init_weights_from_module(module.params["weight"])
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = self.get_module_configuration()
        input_config = super().init_input_config((self.inputs_scales_creators[0].calc_invert_scales(),), lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
        # outputs as bf16, and descaled in gemm under PatchedLinear, so no need to work here
        output_config = [QuantDequantNone(lp_dtype, hp_dtype, scale_format=scale_format)]
        weight_config = self.init_weight_config(
            self.weight_och_scale_calc.scale,
            self.weight_och_scale_calc.calc_invert_scales(),
            lp_dtype,
            hp_dtype,
            scale_format,
            use_qdq,
            fake_quant,
        )
        params_config = {"weight": weight_config}
        if hasattr(self.mod, "bias") and (getattr(self.mod, "bias") is not None):
            # In PatchedLinear the bias is added to the output of gemm.
            # The output is expected to be descaled and in bf16, so we don't need to touch the bias.
            bias_config = [QuantDequantNone(lp_dtype, hp_dtype)]
            params_config.update({"bias": bias_config})
        return ModuleConfig(input_config, output_config, params_config)



class MatmulOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, module_type):
        super().__init__(config, mod, measurement, params, module_type)
        self.inputs_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.INPUT))
        self.inputs_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.INPUT))
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))


    def get_scales_module_config(self):
        input_scales = self.calc_input_scales(num_of_inputs=2)

        output_scales = input_scales[0] * input_scales[1]
        return ModuleConfig(
            input_scales,
            (output_scales,),
            {}
        )

    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = super().get_module_configuration()
        input_config = super().init_input_config((self.inputs_scales_creators[0].calc_invert_scales(),
                                                  self.inputs_scales_creators[1].calc_invert_scales()),
                                                 lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
        # outputs as bf16, and descaled in gemm under PatchedLinear, so no need to work here
        output_config = [QuantDequantNone(lp_dtype, hp_dtype, scale_format=scale_format)]
        return ModuleConfig(input_config, output_config)

class SoftmaxOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, module_type):
        super().__init__( config, mod, measurement, params, module_type)
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))

    def get_scales_module_config(self):
        output_scales = self.calc_output_scales()

        return ModuleConfig((),output_scales)

    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = super().get_module_configuration()
        output_config = [DequantOutput(self.output_scales_creators[0].scale, lp_dtype, hp_dtype, scale_format=scale_format)]
        return ModuleConfig([], output_config, {})

class FsdpaOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, module_type):
        super().__init__(config, mod, measurement, params, module_type)
        self.num_of_inputs = 4
        self.inputs_scales_creators = [self.scales_method_factory.get_scale_method(QuantTensorName.INPUT)
                                       for i in range(self.num_of_inputs)]
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))

    def get_scales_module_config(self):
        # 3 inputs calcs from input measurement
        input_scales = self.calc_input_scales(num_of_inputs=self.num_of_inputs - 1)
        # one input calcs from output measurement
        output1_measurement = self.measurement.outputs[1] if self.measurement is not None else []
        input_scales.append(self.inputs_scales_creators[self.num_of_inputs-1].calc_scales(output1_measurement, QuantTensorType.MEASUREMENTS))
        output_scales = self.calc_output_scales()
        return ModuleConfig(
            input_scales,
            output_scales,
            {}
        )
    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = super().get_module_configuration()
        input_scales_inv =  [self.inputs_scales_creators[i].calc_invert_scales() for i in range(len(self.inputs_scales_creators))]
        input_config = super().init_input_config(
            input_scales_inv
            , lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
        output_config =  [DequantOutput(self.output_scales_creators[0].scale, lp_dtype, hp_dtype, scale_format=scale_format)]
        return ModuleConfig(input_config, output_config, {})

class KVCacheOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, module_type):
        super().__init__(config, mod, measurement, params, module_type)
        self.inputs_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.INPUT))
        self.output_scales_creators.append(self.inputs_scales_creators[0])

    def get_scales_module_config(self):
        input_scales = self.calc_input_scales(num_of_inputs=1)
        self.output_scales_creators[0].scale = self.inputs_scales_creators[0].scale
        output_scales = [self.output_scales_creators[0].scale]
        return ModuleConfig(
            input_scales,
            output_scales,
            {}
        )

    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = super().get_module_configuration()
        input_scales_inv =  [self.inputs_scales_creators[i].calc_invert_scales() for i in range(len(self.inputs_scales_creators))]
        input_config = super().init_input_config(
            input_scales_inv
            , lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
        output_config = [DequantOutput(self.output_scales_creators[0].scale, lp_dtype, hp_dtype, scale_format=scale_format)]
        return ModuleConfig(input_config, output_config)


class DynamicMoeOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, module_type):
        super().__init__(config, mod, measurement, params, module_type)
        self.inputs_scales_creators = [self.scales_method_factory.get_scale_method(QuantTensorName.INPUT) for i in range(len(measurement.inputs) + mod.num_experts)]
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))

    def get_scales_module_config(self):
        num_of_inputs = len(self.measurement.inputs) if self.measurement is not None else 1
        num_of_experts = self.mod.num_experts if self.mod.num_experts is not None else 8
        input_scales = self.calc_input_scales(num_of_inputs=num_of_inputs)
        for i in  range(num_of_experts):
            output_measurement = self.measurement.outputs[i+1] if self.measurement is not None else []
            input_scales.append(
                self.inputs_scales_creators[num_of_inputs + i].calc_scales(output_measurement, QuantTensorType.MEASUREMENTS))
        output_scales = self.calc_output_scales()
        return ModuleConfig(
            input_scales,
            output_scales,
            {}
            )

    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = super().get_module_configuration()
        input_scales_inv =  [self.inputs_scales_creators[i].calc_invert_scales() for i in range(len(self.inputs_scales_creators))]
        input_config = super().init_input_config(
            input_scales_inv
            , lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant)
        output_config =  [QuantDequantNone(lp_dtype, hp_dtype, scale_format=scale_format)]
        return ModuleConfig(input_config, output_config)


ops_quantizer_map = {"linear": LinearOpQuantizer,
                      "matmul": MatmulOpQuantizer,
                      "fused_sdpa": FsdpaOpQuantizer,
                      "softmax": SoftmaxOpQuantizer,
                      "kv_cache": KVCacheOpQuantizer,
                      "dynamic_moe": DynamicMoeOpQuantizer
                     }

def get_op_quantizer(module_type, config, mod, measurement, params):
    return ops_quantizer_map[module_type](config, mod, measurement, params, module_type)

