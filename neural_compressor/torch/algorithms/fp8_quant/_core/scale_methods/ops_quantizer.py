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

import torch
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import get_hqt_config, is_supported_dynamic_op
from .scale_method_factory import ScaleMethodFactory, QuantTensorName, ScaleValueType
from ..common import ModuleConfig, QuantTensorType
from ..quant_dequant import DequantOutput, QuantDequant, QuantDequantNone, QuantInput, QuantDynamicInput
from ...utils.logger import logger
from neural_compressor.torch.algorithms.fp8_quant._core.common import dequant_original_fp8_weight_if_needed
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator
cur_device = auto_detect_accelerator().current_device_name()


class BaseOpQuantizer:

    def __init__(self, config, mod, measurement, params, mod_type_str):
        hqt_config = get_hqt_config(mod).cfg
        module_type = hqt_config["mod_dict"][mod_type_str]
        self.scales_method_factory = ScaleMethodFactory(config, params, mod, module_type)
        self.mod = mod
        self.params = params
        self.measurement = measurement
        self.inputs_scales_creators = []
        self.output_scales_creators = []
        self.params_scales_creators = []
        self.is_dynamic = hqt_config["dynamic_quantization"] and is_supported_dynamic_op(mod_type_str)

        logger.debug("%s %s", self.__class__.__name__, self.__dict__)

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
        for i in range(num_of_inputs):
            input_measurement = self.measurement.inputs[i] if self.measurement is not None else []
            input_scale = None
            if not self.is_dynamic:
                input_scale = self.inputs_scales_creators[i].calc_scales(
                    input_measurement, QuantTensorType.MEASUREMENTS
                )
            input_scales.append(input_scale)
        return input_scales

    def calc_output_scales(self):
        output_measurement = self.measurement.outputs[0] if self.measurement is not None else []
        output_scales = None
        if not self.is_dynamic:
            output_scales = self.output_scales_creators[0].calc_scales(output_measurement, QuantTensorType.MEASUREMENTS)
        return (output_scales,)

    def init_input_config(self, scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant):
        if use_qdq or fake_quant:
            input_config = [
                QuantDequant(s_inv, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq)
                for s_inv in scales_inv
            ]
        else:
            input_config = []
            for input_scales_creator, s_inv in zip(self.inputs_scales_creators, scales_inv):
                if self.is_dynamic:
                    input_config.append(
                        QuantDynamicInput(input_scales_creator, lp_dtype, hp_dtype, scale_format=scale_format)
                    )
                else:
                    input_config.append(QuantInput(s_inv, lp_dtype, hp_dtype, scale_format=scale_format))
        return input_config


class LinearOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, mod_type_str):
        super().__init__(config, mod, measurement, params, mod_type_str)
        hqt_config = get_hqt_config(mod).cfg
        module_type = hqt_config["mod_dict"][mod_type_str]
        if module_type == "row_parallel_linear" and get_hqt_config(mod).cfg["row_parallel_linear_allreduce_quantization"]:
            self.scales_method_factory.output_scale_method_config.backoff = 1.0
        self.inputs_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.INPUT, self.is_dynamic))
        self.weight_och_scale_calc = self.scales_method_factory.get_scale_method(QuantTensorName.WEIGHT_OUT_CH)
        self.weight_ich_scale_calc = self.scales_method_factory.get_scale_method(QuantTensorName.WEIGHT_IN_CH)
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT, self.is_dynamic))

    def get_scales_module_config(self):
        input_scales = self.calc_input_scales(num_of_inputs=1)
        output_measurement = self.measurement.outputs[0] if self.measurement is not None else []
        rescaled_weight = self.mod.weight if hasattr(self.mod, 'weight') else None
        if rescaled_weight is not None:
            rescaled_weight = dequant_original_fp8_weight_if_needed(self.mod, rescaled_weight)
        if self.scales_method_factory.scale_method_config_map[QuantTensorName.WEIGHT_IN_CH].scale_value_type != ScaleValueType.DUMMY_SCALES:
            # Calculating weight in hpu to support scale calculation CGUID torch.ops.hpu.calculate_scale_for_cast
            rescaled_weight = rescaled_weight.to(cur_device)
        if self.weight_ich_scale_calc is not None:
            weight_scales_in_ch = self.weight_ich_scale_calc.calc_scales(input_scales[0], QuantTensorType.CONST)
            rescaled_weight = torch.div(rescaled_weight, weight_scales_in_ch.reshape([1, -1]))
        weights_scales_out_ch = self.weight_och_scale_calc.calc_scales(rescaled_weight, QuantTensorType.CONST)

        params_config = (
            {"weight": weights_scales_out_ch}
            if (self.weight_ich_scale_calc is None)
            else {"weight": {0: weights_scales_out_ch, 1: weight_scales_in_ch}}
        )
        output_scales = None
        if not self.is_dynamic:
            output_scales = self.output_scales_creators[0].calc_scales(
                output_measurement, QuantTensorType.MEASUREMENTS, input0=weights_scales_out_ch, input1=input_scales[0]
            )
        return ModuleConfig(
            input_scales,
            (output_scales,),
            params_config,
        )

    def init_weight_config(self, scales, scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant):
        if use_qdq:
            # to ensure the weights to be loaded to the device in fp8
            weight_config = [
                QuantInput(scales_inv, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq),
                DequantOutput(scales, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq),
            ]
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

    def get_output_config(self, lp_dtype, hp_dtype, scale_format):
        output_config = [QuantDequantNone(lp_dtype, hp_dtype, scale_format=scale_format)]
        return output_config

    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        self.init_weights_from_module(module.params["weight"])
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = self.get_module_configuration()
        input_config = super().init_input_config(
            (self.inputs_scales_creators[0].calc_invert_scales(),),
            lp_dtype,
            hp_dtype,
            scale_format,
            use_qdq,
            fake_quant,
        )
        # outputs as bf16, and descaled in gemm under PatchedLinear, so no need to work here
        output_config = self.get_output_config(lp_dtype, hp_dtype, scale_format=scale_format)
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

class RowParallelLinearOpQuantizer(LinearOpQuantizer):
    def __init__(self, config, mod, measurement, params, mod_type_str):
        super().__init__(config, mod, measurement, params, mod_type_str)
        self.allreduce_quantization_enabled = get_hqt_config(mod).cfg["row_parallel_linear_allreduce_quantization"]
        if self.allreduce_quantization_enabled:
            self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))

    def init_scales_from_module_config(self, module):
        if not self.allreduce_quantization_enabled:
            return super().init_scales_from_module_config(module)
        for idx, input in enumerate(module.inputs):
            if self.inputs_scales_creators[idx].scale is None:
                self.inputs_scales_creators[idx].scale = input
        if self.output_scales_creators[0].scale is None:
            self.output_scales_creators[0].scale = module.outputs[0]
        if self.allreduce_quantization_enabled and self.output_scales_creators[1].scale is None:
                self.output_scales_creators[1].scale = module.outputs[1]

    def get_scales_module_config(self):
        if not self.allreduce_quantization_enabled:
            return super().get_scales_module_config()
        module_config = super().get_scales_module_config()
        output_measurement = self.measurement.outputs[1] if self.measurement is not None else []
        output_scales = self.output_scales_creators[1].calc_scales(output_measurement, QuantTensorType.MEASUREMENTS)
        module_config.outputs = (module_config.outputs[0], output_scales,)
        return module_config

    def get_output_config(self, lp_dtype, hp_dtype, scale_format):
        if not self.allreduce_quantization_enabled:
            return super().get_output_config(lp_dtype, hp_dtype, scale_format)
        scale_0 = self.output_scales_creators[0].scale
        inv_scale_0 = self.output_scales_creators[0].calc_invert_scales()
        output_config_dq_scatter_output = DequantOutput(scale_0, lp_dtype, hp_dtype, scale_format=scale_format)
        output_config_q_scatter_input = QuantInput(inv_scale_0, lp_dtype, hp_dtype, scale_format=scale_format)
        output_config = [output_config_dq_scatter_output,
                         output_config_q_scatter_input]
        inv_scale_1 = self.output_scales_creators[1].calc_invert_scales()
        scale_1 = self.output_scales_creators[1].scale
        output_config_q_gather_input = QuantInput(inv_scale_1, lp_dtype, hp_dtype, scale_format=scale_format)
        output_config_dq_gather_output = DequantOutput(scale_1, lp_dtype, hp_dtype, scale_format=scale_format)
        output_config.extend([output_config_q_gather_input, output_config_dq_gather_output])
        return output_config

class MatmulOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, mod_type_str):
        super().__init__(config, mod, measurement, params, mod_type_str)
        self.inputs_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.INPUT))
        self.inputs_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.INPUT))
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))

    def get_scales_module_config(self):
        input_scales = self.calc_input_scales(num_of_inputs=2)

        output_scales = input_scales[0] * input_scales[1]
        return ModuleConfig(input_scales, (output_scales,), {})

    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = super().get_module_configuration()
        input_config = super().init_input_config(
            (self.inputs_scales_creators[0].calc_invert_scales(), self.inputs_scales_creators[1].calc_invert_scales()),
            lp_dtype,
            hp_dtype,
            scale_format,
            use_qdq,
            fake_quant,
        )

        # 4bit->8bit inputs, no need to quant
        if hasattr(self.mod, "no_input_quant"):
            input_config[1] = QuantDequantNone(lp_dtype, hp_dtype, scale_format=scale_format)

        # outputs as bf16, and descaled in gemm under PatchedLinear, so no need to work here
        output_config = [QuantDequantNone(lp_dtype, hp_dtype, scale_format=scale_format)]
        return ModuleConfig(input_config, output_config)


class SoftmaxOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, mod_type_str):
        super().__init__(config, mod, measurement, params, mod_type_str)
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))

    def get_scales_module_config(self):
        output_scales = self.calc_output_scales()

        return ModuleConfig((), output_scales)

    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = super().get_module_configuration()
        output_config = [
            DequantOutput(self.output_scales_creators[0].scale, lp_dtype, hp_dtype, scale_format=scale_format)
        ]
        return ModuleConfig([], output_config, {})


class FsdpaOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, mod_type_str):
        super().__init__(config, mod, measurement, params, mod_type_str)
        self.num_of_inputs = 4
        self.inputs_scales_creators = [
            self.scales_method_factory.get_scale_method(QuantTensorName.INPUT) for i in range(self.num_of_inputs)
        ]
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))

    def get_scales_module_config(self):
        # 3 inputs calcs from input measurement
        input_scales = self.calc_input_scales(num_of_inputs=self.num_of_inputs - 1)
        # one input calcs from output measurement
        output1_measurement = self.measurement.outputs[1] if self.measurement is not None else []
        input_scales.append(
            self.inputs_scales_creators[self.num_of_inputs - 1].calc_scales(
                output1_measurement, QuantTensorType.MEASUREMENTS
            )
        )
        output_scales = self.calc_output_scales()
        return ModuleConfig(input_scales, output_scales, {})

    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = super().get_module_configuration()
        input_scales_inv = [
            self.inputs_scales_creators[i].calc_invert_scales() for i in range(len(self.inputs_scales_creators))
        ]
        input_config = super().init_input_config(
            input_scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant
        )
        output_config = [
            DequantOutput(self.output_scales_creators[0].scale, lp_dtype, hp_dtype, scale_format=scale_format)
        ]
        return ModuleConfig(input_config, output_config, {})


class KVCacheOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, mod_type_str):
        super().__init__(config, mod, measurement, params, mod_type_str)
        self.inputs_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.INPUT))
        self.output_scales_creators.append(self.inputs_scales_creators[0])

    # TODO: Remove after implementing lp_dtype in OHF.
    def init_input_config(self, scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant):
        input_config = super().init_input_config(scales_inv, lp_dtype, hp_dtype, scale_format, False, fake_quant)
        if use_qdq:
            input_config.extend([
                QuantDequant(s_inv, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq)
                for s_inv in scales_inv
            ])
        return input_config

    def get_scales_module_config(self):
        input_scales = self.calc_input_scales(num_of_inputs=1)
        self.output_scales_creators[0].scale = self.inputs_scales_creators[0].scale
        output_scales = [self.output_scales_creators[0].scale]
        return ModuleConfig(input_scales, output_scales, {})

    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = super().get_module_configuration()
        input_scales_inv = [
            self.inputs_scales_creators[i].calc_invert_scales() for i in range(len(self.inputs_scales_creators))
        ]
        # TODO: After implementing lp_dtype in OHF can call:
        # `super().init_input_config(scales_inv, lp_dtype, hp_dtype, scale_format, False, fake_quant)`
        input_config = self.init_input_config(
            input_scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant
        )
        output_config = [
            DequantOutput(self.output_scales_creators[0].scale, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=False)
        ]
        return ModuleConfig(input_config, output_config)


class DynamicMoeOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, mod_type_str):
        super().__init__(config, mod, measurement, params, mod_type_str)
        num_of_inputs = len(self.measurement.inputs) if self.measurement is not None else 1
        if hasattr(self.mod, "local_num_experts"):
            num_of_experts = self.mod.local_num_experts
        elif hasattr(self.mod, "num_experts"):
            num_of_experts = self.mod.num_experts
        else:
            num_of_experts = 8
        
        self.inputs_scales_creators = [
            self.scales_method_factory.get_scale_method(QuantTensorName.INPUT, is_dynamic=self.is_dynamic)
            for i in range(num_of_inputs + num_of_experts)
        ]
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))

    def get_scales_module_config(self):
        num_of_inputs = len(self.measurement.inputs) if self.measurement is not None else 1
        if hasattr(self.mod, "local_num_experts"):
            num_of_experts = self.mod.local_num_experts
        elif hasattr(self.mod, "num_experts"):
            num_of_experts = self.mod.num_experts
        else:
            num_of_experts = 8
        input_scales = self.calc_input_scales(num_of_inputs=num_of_inputs)
        for i in range(num_of_experts):
            output_measurement = self.measurement.outputs[i + 1] if self.measurement is not None else []
            input_scales.append(
                self.inputs_scales_creators[num_of_inputs + i].calc_scales(
                    output_measurement, QuantTensorType.MEASUREMENTS
                )
            )
        output_scales = self.calc_output_scales()
        return ModuleConfig(input_scales, output_scales, {})

    def scales_module_config_to_q_and_dq(self, module):
        self.init_scales_from_module_config(module)
        scale_format, use_qdq, fake_quant, lp_dtype, hp_dtype = super().get_module_configuration()
        input_scales_inv = [
            self.inputs_scales_creators[i].calc_invert_scales() for i in range(len(self.inputs_scales_creators))
        ]
        input_config = super().init_input_config(
            input_scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant
        )
        output_config = [QuantDequantNone(lp_dtype, hp_dtype, scale_format=scale_format)]
        return ModuleConfig(input_config, output_config)

    

class EmbeddingOpQuantizer(BaseOpQuantizer):

    def __init__(self, config, mod, measurement, params, module_type):
        super().__init__(config, mod, measurement, params, module_type)
        self.inputs_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.INPUT))
        self.weight_och_scale_calc = self.scales_method_factory.get_scale_method(QuantTensorName.WEIGHT_OUT_CH)
        self.weight_ich_scale_calc = self.scales_method_factory.get_scale_method(QuantTensorName.WEIGHT_IN_CH)
        self.output_scales_creators.append(self.scales_method_factory.get_scale_method(QuantTensorName.OUTPUT))

    def get_scales_module_config(self):
        weight = self.mod.weight if hasattr(self.mod, 'weight') else None
        input_scales = self.calc_input_scales(num_of_inputs=1)

        if self.weight_ich_scale_calc is not None:
            weight_scales_in_ch = self.weight_ich_scale_calc.calc_scales(input_scales[0], QuantTensorType.CONST)
            weight = torch.div(weight, weight_scales_in_ch.reshape([1, -1]))
        weights_scales_out_ch = self.weight_och_scale_calc.calc_scales(weight, QuantTensorType.CONST)

        params_config = (
            {"weight": weights_scales_out_ch}
            if (self.weight_ich_scale_calc is None)
            else {"weight": {0: weights_scales_out_ch, 1: weight_scales_in_ch}}
        )
        return ModuleConfig(
            (),
            (),
            params_config,
        )

    def init_weight_config(self, scales, scales_inv, lp_dtype, hp_dtype, scale_format, use_qdq, fake_quant):
        if use_qdq:
            # to ensure the weights to be loaded to the device in fp8
            weight_config = [
                QuantInput(scales_inv, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq),
                DequantOutput(scales, lp_dtype, hp_dtype, scale_format=scale_format, use_qdq=use_qdq),
            ]
        else:
            raise ValueError("For FP8 quantization, {} only supports QDQ mode now!".format(self.mod.__class__.__name__))
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
        return ModuleConfig([], [], params_config)


ops_quantizer_map = {"linear": LinearOpQuantizer,
                      "matmul": MatmulOpQuantizer,
                      "fused_sdpa": FsdpaOpQuantizer,
                      "softmax": SoftmaxOpQuantizer,
                      "kv_cache": KVCacheOpQuantizer,
                      "dynamic_moe": DynamicMoeOpQuantizer,
                      "row_parallel_linear": RowParallelLinearOpQuantizer,
                      "embedding": EmbeddingOpQuantizer,
                     }

def get_op_quantizer(config, mod, measurement, params, mod_type_str):
    hqt_config = get_hqt_config(mod).cfg
    module_type = hqt_config["mod_dict"][mod_type_str]
    return ops_quantizer_map[module_type](config, mod, measurement, params, mod_type_str)
