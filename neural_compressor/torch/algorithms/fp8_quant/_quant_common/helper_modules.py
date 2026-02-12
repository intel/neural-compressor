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
from abc import abstractmethod

import torch
import torch.nn as nn
import types
import functools

from .._core.quant_dequant import QuantDequant as qdq, QuantDynamicInput
from .._core.quantized_func_wrappers import get_quantized_func_wrapper, OP_TYPE
from .quant_config import QuantMode, get_hqt_config
from ..patched_module_base import PatchedModuleBase, get_call_wrapper
from .._core.scale_handler import get_scale_dtype, ScaleFormat
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator
cur_accelerator = auto_detect_accelerator()


class BMM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)


class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.matmul(*args, **kwargs)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clone()


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, invAttnHead=None):
        return torch.softmax(x, dim)


def measure_input(input, observer):
    for i in range(len(observer)):
        observer[i].measure(input[i])


def measure_output(output, observer):
    if observer:
        for i in range(len(observer)):
            observer[i].measure(output[i])


def get_current_repr(cls_instance, *member_names):
    curr_repr = ""
    if cls_instance.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
        first_name = True
        for name in member_names:
            if not first_name:
                curr_repr += ", "
            cur_attr = getattr(cls_instance, name)
            if isinstance(cur_attr, list) and len(cur_attr) > 1:
                dtype = get_scale_dtype(cur_attr[0])
                curr_repr += f"{name} type=list of {dtype}, length={len(cur_attr)}"
            else:
                # currently, only scale is called here.
                dtype = get_scale_dtype(cur_attr)
                curr_repr += f"{name} dtype={dtype}"
            first_name = False
    return curr_repr


def extra_representation(org_repr, org_name, curr_repr):
    repr = f"original={org_name}," + (" " + org_repr + "," if org_repr != "" else "")
    return f"{repr} {curr_repr}"


def _raise_lora_layer_error(layer_class):
    raise RuntimeError(
        f"{layer_class} quantization is not supported in case of lora_layer member is not None."
        f" Can add {layer_class} to 'blocklist' field in quantization config file"
    )


class PatchedMatmul(PatchedModuleBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            self.quant_input_0 = self._mod_extra_config.inputs[0]
            self.quant_input_1 = self._mod_extra_config.inputs[1]
            if not self.use_qdq and not self.fake_quant:
                self.register_scale("scale_input", mod_extra_config.scale.inputs[0], self.scale_format)
                self.register_scale("scale_other", mod_extra_config.scale.inputs[1], self.scale_format)
                self.matmul_fp8 = get_quantized_func_wrapper(OP_TYPE.MATMUL_GEMM, self.scale_format)
                # in DPQ we want to use the scales measured in the quantization process
                if hasattr(parent, 'scale_bf16_to_fp8') and parent.scale_bf16_to_fp8 > 0:
                    self.scale_other = torch.nn.Parameter(parent.scale_bf16_to_fp8)

    def forward_quant(self, input, other):
        qinput = self.quant_input_0(input)
        qother = self.quant_input_1(other)
        output = self.matmul_fp8(qinput,
                                 qother,
                                 out_dtype=self._mod_extra_config.config_params["hp_dtype"],
                                 scale_input_inv=self.scale_input,
                                 scale_other_inv=self.scale_other)
        return output

    def forward_qdq(self, input, other):
        qinput = self.quant_input_0(input)
        qother = self.quant_input_1(other)
        output = torch.matmul(qinput, qother)
        return output

    def forward_measure(self, input, other):
        measure_input((input, other), observer=self._mod_extra_config.inputs)
        output = self.orig_mod(input, other)
        measure_output((output,), self._mod_extra_config.outputs)
        return output

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_other") if not self.use_qdq else "",
        )


def init_mixture_of_experts_linears(instance):
    parent_name = instance.orig_mod_parent.__class__.__name__
    if parent_name == "MixtralBlockSparseTop2MLP" or parent_name == "GaudiDeepseekV3MLP":
        # this linear is part of MixtureOfExperts block
        # MoE linears hold the weights but their forward logic is done using the dynamic op
        # therefore no measure object is saved causing no quant object as well
        # adding a fake 0 data input to instantiate the needed measurment objects
        measure_input((torch.tensor(0),), observer=instance._mod_extra_config.inputs)


class PatchedLinearBase(PatchedModuleBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)

    # TODO [SW-224538]: Move init_linear to PatchedLinearBase __init__
    def init_linear(self, mod_extra_config):
        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            # When offloading weights to disk using device_map, the module forward is overridden.
            # __dict__.update call again overrides the PatchedLinear forward with the forward that device_map planted.
            # So need to set PatchedLinear forward to be the right forward.
            self.weight = nn.Parameter(self.weight.t().contiguous())
            self.quant_input = self._mod_extra_config.inputs[0]
            self.dequant_output = self._mod_extra_config.outputs[0]

            # When offloading weights to disk using device_map, the module forward is overridden.
            # __dict__.update call again overrides the PatchedLinear forward with the forward that device_map planted.
            # So need to set PatchedLinear forward to be the right forward.
            if self.use_qdq:
                self.dequant_weights = self._mod_extra_config.params["weight"][1]
            elif self.fake_quant:
                self.qdq_weights = self._mod_extra_config.params["weight"][0]
            else:
                self.matmul_fp8 = get_quantized_func_wrapper(OP_TYPE.LINEAR_GEMM, self.scale_format)
                # input0 is None when initializing a dynamic quantization op
                self.register_scale("scale_input", mod_extra_config.scale.inputs[0], self.scale_format)
                if isinstance(mod_extra_config.scale.params["weight"], (torch.Tensor, float)):
                    self.register_scale("scale_weight", mod_extra_config.scale.params["weight"], self.scale_format)
                elif isinstance(mod_extra_config.scale.params["weight"], dict):
                    # PCQ weight is calculated with actual weight [0] and ones [1]
                    # only ScaleFormat.CONST is supported for per-channel scale now.
                    self.register_scale("scale_weight", mod_extra_config.scale.params["weight"][0], ScaleFormat.CONST)

            self.is_dynamic_quantization = isinstance(self.quant_input, QuantDynamicInput)
            self.quant_input_func = self.quant_input_and_get_scale_dynamic if self.is_dynamic_quantization else self.quant_input_and_get_scale_static
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            init_mixture_of_experts_linears(self)

    def quant_input_and_get_scale_dynamic(self, input):
        return self.quant_input(input)

    def quant_input_and_get_scale_static(self, input):
        return self.quant_input(input), self.scale_input

    def run_linear_quant(self, input, bias):
        qinput, scale_input = self.quant_input_func(input)
        y = self.matmul_fp8(qinput,
                            self.weight,
                            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
                            scale_input_inv=scale_input,
                            scale_other_inv=self.scale_weight)
        output = y + bias if (bias is not None) else y
        return output

    def run_linear_qdq(self, input, bias):
        if self.fake_quant:
            qweight = self.qdq_weights(self.weight, )
        else:
            qweight = self.dequant_weights(self.weight, )
        qinput = self.quant_input(input)
        y = torch.matmul(qinput, qweight)
        output = y + bias if (bias is not None) else y
        return output

    def forward_qdq(self, input):
        return self.run_linear_qdq(input, self.bias)

    def forward_quant(self, input):
        return self.run_linear_quant(input, self.bias)

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_weight") if not self.use_qdq else get_current_repr(self),
        )


class PatchedLinear(PatchedLinearBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.init_linear(mod_extra_config)

    def forward_measure(self, input):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output = self.orig_mod(input)
        measure_output((output,), self._mod_extra_config.outputs)
        return output
    
    
class PatchedParallelLMHead(PatchedLinearBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        # ParallelLMHead inherits from VocabParallelEmbedding (nn.module) which has a member called
        # "quant_method" of type UnquantizedEmbeddingMethod that inherits from QuantizeMethodBase
        # (both are not nn.module) and implement an "apply" method by using torch.nn.functional.linear
        # (Apply the weights in layer to the input tensor.)
        # ParallelLMHead's forward method should not be called because LMHead's weights should be used
        # in the sampler. (The forward itself throws RuntimeError exception)
        # So in order to quantize that quant_method we patch only the "apply" method.
        self.init_linear(mod_extra_config)
        self.orig_linear_quant_apply = self.orig_mod.quant_method.apply
        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            if self.use_qdq or self.fake_quant:
                self.quant_method.apply = self.apply_qdq
            else:
                self.quant_method.apply = self.apply_quant
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            self.quant_method.apply = self.apply_measure

    def apply_quant(self, layer, x, bias):
        return self.run_linear_quant(x, bias)

    def apply_qdq(self, layer, x, bias):
        return self.run_linear_qdq(x, bias)

    def apply_measure(self, layer, x, bias):
        measure_input((x,), observer=self._mod_extra_config.inputs)
        output = self.orig_linear_quant_apply(layer, x, bias)
        measure_output((output,), self._mod_extra_config.outputs)
        return output


class PatchedReplicatedLinear(PatchedLinearBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.init_linear(mod_extra_config)

    def forward_qdq(self, input):
        bias = self.bias if not self.skip_bias_add else None
        output = self.run_linear_qdq(input, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def forward_quant(self, input):
        bias = self.bias if not self.skip_bias_add else None
        output = self.run_linear_quant(input, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def forward_measure(self, input):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output, output_bias = self.orig_mod(input)
        measure_output((output,), self._mod_extra_config.outputs)
        return output, output_bias


class PatchedLinearAllReduce(PatchedLinearBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.init_linear(mod_extra_config)
        self.scoped_version = mod.__class__.__name__ == "ScopedLinearAllReduce"

    def forward_qdq(self, input):
        # pre_all_reduce
        output = self.run_linear_qdq(input, None)

        if not self.scoped_version:
            self.all_reduce(output)
            output = self.post_all_reduce(output)
        return output

    def forward_quant(self, input):
        # pre_all_reduce
        output = self.run_linear_quant(input, None)
        # TODO: enable dynamic quantization for output
        dqoutput = self.dequant_output(output)
        if not self.scoped_version:
            self.all_reduce(dqoutput)
            dqoutput = self.post_all_reduce(dqoutput)
        return dqoutput

    def forward_measure(self, input):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        measure_output((output,), self._mod_extra_config.outputs)
        # in scoped version all reduce is being called outside of the layer
        if not self.scoped_version:
            self.all_reduce(output)
            output = self.post_all_reduce(output)
        return output

    def all_reduce(self, input):
        if self.mp_group is not None:
            from deepspeed import comm as dist

            dist.inference_all_reduce(input, group=self.mp_group)

    def post_all_reduce(self, input):
        output = input + self.bias if (self.bias is not None) else input
        return output


class PatchedRowParallelLinear(PatchedLinearBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        from .._core.external_func_impl import get_external_row_parallel_collective_func
        self.row_parallel_collective_func = get_external_row_parallel_collective_func()
        # TODO [SW-224403]: Enable dynamic quantization in row parallel allreduce
        allreduce_quantization_enable = get_hqt_config(mod).cfg["row_parallel_linear_allreduce_quantization"]
        if self.quantization_mode in (QuantMode.MEASURE, QuantMode.SHAPE):
            self.forward = self.forward_measure_reduce if self.reduce_results and self.tp_size > 1 else self.forward_measure_no_reduce

        elif self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            if self.fake_quant or self.use_qdq:
                self.forward = self.forward_qdq
            else:
                if self.reduce_results and self.tp_size > 1:
                    if allreduce_quantization_enable:
                        self.forward = self.forward_quant_reduce_in_lp
                    else:
                        self.forward = self.forward_quant_reduce_in_hp
                else:
                    self.forward = self.forward_quant_no_reduce
        self.init_linear(mod_extra_config)
        if self.quantization_mode == QuantMode.QUANTIZE:
            if allreduce_quantization_enable:
                self.dequant_scatter_output = self._mod_extra_config.outputs[0]
                self.quant_scatter_input = self._mod_extra_config.outputs[1]
                self.quant_gather_input = self._mod_extra_config.outputs[2]
                self.dequant_gather_output = self._mod_extra_config.outputs[3]
        from torch import distributed as dist
        self.world_size = dist.get_world_size()

    def resolve_input(self, input_):
        """
        this code is copied from vllm RowParallelLinear forward method
        """
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()
        return input_parallel

    def forward_qdq(self, input):
        # TODO: [SW-208441] Support all_reduce_fp8 in forward_qdq in PatchedRowParallelLinear
        resolved_input = self.resolve_input(input)
        output = self.run_linear_qdq(resolved_input, None)

        if self.reduce_results:
            output = self.row_parallel_collective_func(output)
        return self.bias_add(output)

    def lp_matmul_hp(self, input):
        """
        Perfoms a matmul with lp inputs and returns output in hp
        """
        resolved_input = self.resolve_input(input)
        return self.run_linear_quant(resolved_input, None)

    def forward_quant_no_reduce(self, input):
        matmul_output_hp = self.lp_matmul_hp(input)
        return self.bias_add(matmul_output_hp)

    def forward_quant_reduce_in_lp(self, input):
        matmul_output_hp = self.lp_matmul_hp(input)
        # performing all_reduce in fp8 should be done only at decode phase
        # decode phase is indicated by input.shape[1] == 1
        if input.shape[1] == 1:
            allreduce_output_hp = self.quant_all_reduce_sum(matmul_output_hp)
        else:
            allreduce_output_hp = self.row_parallel_collective_func(matmul_output_hp)
        return self.bias_add(allreduce_output_hp)

    def forward_quant_reduce_in_hp(self, input):
        matmul_output_hp = self.lp_matmul_hp(input)
        all_reduce_output_hp = self.row_parallel_collective_func(matmul_output_hp)
        return self.bias_add(all_reduce_output_hp)

    def measure_input_and_matmul(self, input):
        resolved_input = self.resolve_input(input)
        measure_input((resolved_input,), observer=self._mod_extra_config.inputs)
        return self.orig_mod.quant_method.apply(self.orig_mod, resolved_input)

    def forward_measure_no_reduce(self, input):
        output = self.measure_input_and_matmul(input)
        temp = torch.empty(1)
        measure_output((output, temp,), self._mod_extra_config.outputs)
        return self.bias_add(output)

    def forward_measure_reduce(self, input):
        from torch import distributed as dist
        output = self.measure_input_and_matmul(input)
        max_output = output.clone()
        dist.all_reduce(max_output, op=dist.ReduceOp.MAX)
        all_reduce_output = self.row_parallel_collective_func(output)
        measure_output((max_output, all_reduce_output,), self._mod_extra_config.outputs)
        return self.bias_add(all_reduce_output)

    def scatter(self, tensor):
        from torch import distributed as dist
        buffer = torch.empty_like(tensor)
        # Perform all_to_all_single communication
        dist.all_to_all_single(buffer, tensor)
        # Reshape output_tensor to (world_size, -1)
        return buffer.view(self.world_size, -1)

    def gather(self, tensor, reduced_chunk, original_shape):
        from torch import distributed as dist
        # Gather the reduced chunks from all processes
        dist.all_gather_into_tensor(tensor, reduced_chunk)
        # Reshape back to the original tensor shape
        return tensor.view(original_shape)

    def quant_all_reduce_sum(self, scatter_input_hp):
        scatter_input_lp = self.quant_scatter_input(scatter_input_hp)
        original_shape = scatter_input_lp.shape
        buffer_lp = self.scatter(scatter_input_lp)

        # Dequant tensor as sum preformed in bf16
        buffer_hp = self.dequant_scatter_output(buffer_lp)
        # Sum the received chunks (local reduction)
        sum_hp = buffer_hp.sum(dim=0)  # Shape: [-1]
        # Quant tensor as all_gather preformed in fp8
        gather_input_lp = self.quant_gather_input(sum_hp)

        gather_output_lp = self.gather(scatter_input_lp, gather_input_lp, original_shape)
        gather_output_hp = self.dequant_gather_output(gather_output_lp)

        return gather_output_hp

    def bias_add(self, output):
        assert (
            self.reduce_results or (not self.bias) or self.skip_bias_add
        ), "When not reduce the results, adding bias to the results can lead to incorrect results"
        if not self.skip_bias_add:
            output = output + self.bias if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias
        return output, output_bias


class PatchedColumnParallelLinear(PatchedLinearBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.init_linear(mod_extra_config)
        from .._core.external_func_impl import get_external_column_parallel_collective_func
        self.column_parallel_collective_func = get_external_column_parallel_collective_func()

    def forward_qdq(self, input):
        output = self.run_linear_qdq(input, None)
        output, output_bias = self.add_bias(output)
        if self.gather_output:
            output = self.column_parallel_collective_func(output)
        return output, output_bias

    def forward_quant(self, input):
        output = self.run_linear_quant(input, None)
        # TODO: enable dynamic quantization for output
        dqoutput = self.dequant_output(output)
        dqoutput, dqoutput_bias = self.add_bias(dqoutput)
        if self.gather_output:
            dqoutput = self.column_parallel_collective_func(dqoutput)
        return dqoutput, dqoutput_bias

    def forward_measure(self, input):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output = self.orig_mod.quant_method.apply(self.orig_mod, input)
        measure_output((output,), self._mod_extra_config.outputs)
        output, output_bias = self.add_bias(output)
        if self.gather_output:
            output = self.column_parallel_collective_func(output)
        return output, output_bias

    def add_bias(self, output):
        if not self.skip_bias_add:
            output = output + self.bias if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias
        return output, output_bias


class PatchedLmHeadLinearAllreduce(PatchedLinearBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.init_linear(mod_extra_config)

    def forward_qdq(self, input):
        assert (
            input.shape[-1] % self.world_size == 0
        ), "Please ensure that self.world_size is divisible by input.shape[-1]"
        input_shard = input.shape[-1] // self.world_size
        splittedInput = input[:, :, self.rank * input_shard: (self.rank + 1) * input_shard]
        output = self.run_linear_qdq(splittedInput, None)

        if self.mp_group is not None:
            from deepspeed import comm as dist

            dist.inference_all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output

    def forward_quant(self, input):
        from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list
        input_shard_size = get_shard_size(input.shape[-1], self.world_size, "lm_head")
        input_shard_offset = sum(get_shard_size_list(input.shape[-1], self.world_size, "lm_head")[0:self.rank])
        splittedInput = input[:, :, input_shard_offset:input_shard_offset + input_shard_size]
        output = self.run_linear_quant(splittedInput, None)
        # TODO: enable dynamic quantization for output
        dqoutput = self.dequant_output(output)

        if self.mp_group is not None:
            from deepspeed import comm as dist

            dist.inference_all_reduce(dqoutput, group=self.mp_group)
        if self.bias is not None:
            dqoutput += self.bias
        return dqoutput

    def forward_measure(self, input):
        from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list
        input_shard_size = get_shard_size(input.shape[-1], self.world_size, "lm_head")
        input_shard_offset = sum(get_shard_size_list(input.shape[-1], self.world_size, "lm_head")[0:self.rank])
        splittedInput = input[:, :, input_shard_offset:input_shard_offset + input_shard_size]
        measure_input((splittedInput,), observer=self._mod_extra_config.inputs)
        output = torch.matmul(splittedInput, self.weight.t())
        measure_output((output,), self._mod_extra_config.outputs)

        if self.mp_group is not None:
            from deepspeed import comm as dist

            dist.inference_all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output


class PatchedEmbeddingBag(PatchedModuleBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            if self.use_qdq:
                self.dequant_weights = self._mod_extra_config.params["weight"][1]
                if isinstance(mod_extra_config.scale.params["weight"], (torch.Tensor, float)):
                    self.register_scale("scale_weight", mod_extra_config.scale.params["weight"], self.scale_format)
                elif isinstance(mod_extra_config.scale.params["weight"], dict):
                    # PCQ weight is calculated with actual weight [0] and ones [1]
                    # only ScaleFormat.CONST is supported for per-channel scale now.
                    self.register_scale("scale_weight", mod_extra_config.scale.params["weight"][0], ScaleFormat.CONST)
            else:
                raise ValueError("EmbeddingBag is only supported QDQ mode now!")

    def forward_qdq(self, input, offsets, *args, **kwargs):
        qweight = self.dequant_weights(self.weight, )

        return torch.nn.functional.embedding_bag(
            input=input,
            offsets=offsets,
            weight=qweight,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            mode=self.mode,
            sparse=self.sparse,
            include_last_offset=self.include_last_offset,
            padding_idx=self.padding_idx,
            *args,
            **kwargs,
        )

    def forward_measure(self, input, *args, **kwargs):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output = self.orig_mod(input, *args, **kwargs)
        measure_output((output,), self._mod_extra_config.outputs)
        return output

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_weight"),
        )


# patched vllm FusedMoE module removing the bf16 weights of all experts
# measure and quant of the weights is done per expert using PatchedMoeMatmul
# therefore it is configured: ModuleInfo.should_measure_and_quant = False
class PatchedMixtralMoE(PatchedModuleBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        # remove the MoE weights that are quanted by PatchedMoeMatmul
        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            delattr(mod, "w13_weight")
            delattr(mod, "w2_weight")
            setattr(mod, "w13_weight", None)
            setattr(mod, "w2_weight", None)
            setattr(self, "w13_weight", None)
            setattr(self, "w2_weight", None)
        self.forward = self.forward_orig

    # copied from https://github.com/HabanaAI/vllm-fork/blob/93b8bad8478451349d0c76b3116d3ad863a3b48e/vllm/model_executor/layers/fused_moe/layer.py#L1429
    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        """
        The pplx combine kernel reduces across GPU ranks by default.
        """
        if (self.orig_mod.use_pplx_kernels or self.orig_mod.use_deepep_ht_kernels
                or self.orig_mod.use_deepep_ll_kernels):
            return final_hidden_states
        else:
            from .._core.external_func_impl import get_external_row_parallel_collective_func
            tensor_model_parallel_all_reduce = get_external_row_parallel_collective_func()
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self),
        )


# This patched module is called by the vllm-mixtral FusedMoE layer
# we wrap each expert weight with this module since FusedMoE has a single tensor for all experts weights
# this way we can calculate scales per expert and achive better accuracy
class PatchedMoeMatmul(PatchedLinearBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.init_linear(mod_extra_config)
        if (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            measure_input((torch.tensor(0),), observer=self._mod_extra_config.inputs)
        else:
            self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=False)

    def forward_qdq(self, input, *args, **kwargs):
        return self.run_linear_qdq(input, None)

    def forward_quant(self, input, *args, **kwargs):
        return self.run_linear_quant(input, None)

    def forward_measure(self, input, *args, **kwargs):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output = self.orig_mod(input, *args, **kwargs)
        measure_output((output,), self._mod_extra_config.outputs)
        return output


class PatchedGaudiMixtralSparseMoeBlock(PatchedModuleBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.forward = self.forward_orig
        self.ep_size = mod.ep_size
        self.experts_min = mod.experts_min
        self.experts_max = mod.experts_max
        self.experts_range = mod.experts_range
        self.num_experts = mod.num_experts

        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            self.dynamic_moe_op = get_quantized_func_wrapper(OP_TYPE.DYNAMIC_MOE, self.scale_format)
            self.quant_input = self._mod_extra_config.inputs[0]
            self.register_scale("scale_input", mod_extra_config.scale.inputs[0], self.scale_format)
            self.register_scale(
                "scale_intermediate",
                [mod_extra_config.scale.inputs[x] for x in range(self.experts_min + 1, self.experts_max + 2)],
                self.scale_format,
            )
            mod.call_dynamic_moe_op = get_call_wrapper(self, "call_dynamic_moe_quant_op")
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            mod.call_dynamic_moe_op = self.call_dynamic_moe_measure_op

    def call_dynamic_moe_quant_op(self,
                                  hidden_states,
                                  expert_routing_table,
                                  router_weights,
                                  permuted_weights=False,
                                  activation="silu"):
        w1_list = [self.experts[i].w1.weight for i in self.experts_range]
        w2_list = [self.experts[i].w2.weight for i in self.experts_range]
        w3_list = [self.experts[i].w3.weight for i in self.experts_range]
        scale_w1 = [self.experts[i].w1.scale_weight for i in self.experts_range]
        scale_w2 = [self.experts[i].w2.scale_weight for i in self.experts_range]
        scale_w3 = [self.experts[i].w3.scale_weight for i in self.experts_range]
        qinput = self.quant_input(hidden_states)
        output = self.dynamic_moe_op(
            hidden_states=qinput,
            expert_routing_table=expert_routing_table,
            router_weights=router_weights,
            w1=w1_list,
            w3=w2_list,
            w2=w3_list,
            d_scale_w1=scale_w1,
            d_scale_w2=scale_w3,
            d_scale_w3=scale_w2,
            d_scale_hidden_states=self.scale_input,
            d_scale_intermediate_hidden_states=self.scale_intermediate,
            permuted_weights=False,
            activation=activation,
            experts_min=self.experts_min,
            experts_max=self.experts_max,
        )
        return output

    def call_dynamic_moe_measure_op(self,
                                    hidden_states,
                                    expert_routing_table,
                                    router_weights,
                                    permuted_weights=True,
                                    activation="silu"):
        w1_list = [self.experts[i].w1.weight for i in self.experts_range]
        w2_list = [self.experts[i].w2.weight for i in self.experts_range]
        w3_list = [self.experts[i].w3.weight for i in self.experts_range]
        measure_input((hidden_states,), observer=self._mod_extra_config.inputs)

        output, intermidiate_amax = torch.ops.hpu.mixture_of_experts.fp8_measurement(
            hidden_states=hidden_states,
            expert_routing_table=expert_routing_table,
            router_weights=router_weights,
            w1=w1_list,
            w3=w2_list,
            w2=w3_list,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=self.experts_min,
            experts_max=self.experts_max,
            measurement_mode=True,
        )

        amax = []
        for i in range(self.num_experts):
            if i in self.experts_range:
                amax.append(intermidiate_amax[i - self.experts_min])
            else:
                amax.append(torch.tensor(0, device="hpu", dtype=intermidiate_amax[0].dtype))

        output_measure_list = [output] + amax

        measure_output(output_measure_list, self._mod_extra_config.outputs)
        return output

    def extra_repr(self) -> str:
        member_names = ["scale_input"]
        for x in range(1, self.num_experts + 1):
            member_names.append("scale_intermediate[" + str(x) + "]")
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, *member_names),
        )


class PatchedGaudiDeepseekV3MoE(PatchedGaudiMixtralSparseMoeBlock):
    """The patched module for the Optimum Habana DeepseekV3MoE BF16 and FP8 models.
    
    This MoE module differs from the Mixtral MoE by chunking the Gaudi mixture of 
    experts op for models like DeepSeek with large number of experts. The
    size of the chunks is set using SLICE_MAX_EXPERT environment variable.

    For FP8 models, we need to requantize weights as follows:
    - At measurement stage, we dequantize the FP8 weights to BF16.
    - At quantization stage, we use the same `forward_quant` method for both BF16 and FP.
    """
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)

        # Check if FP8 model
        self._is_fp8 = False
        if hasattr(mod.config, "quantization_config"):
            if hasattr(mod.config.quantization_config, "quant_method"):
                self._is_fp8 = (mod.config.quantization_config.quant_method == 'fp8')
                
    def call_dynamic_moe_quant_op(self,
                                  hidden_states,
                                  expert_routing_table,
                                  router_weights,
                                  permuted_weights=False,
                                  activation="silu"):
        
        gate_proj_list = [self.experts[i].gate_proj.weight for i in self.experts_range]
        down_proj_list = [self.experts[i].down_proj.weight for i in self.experts_range]
        up_proj_list = [self.experts[i].up_proj.weight for i in self.experts_range]
        scale_gate_proj = [self.experts[i].gate_proj.scale_weight for i in self.experts_range]
        scale_down_proj = [self.experts[i].down_proj.scale_weight for i in self.experts_range]
        scale_up_proj = [self.experts[i].up_proj.scale_weight for i in self.experts_range]
        qinput = self.quant_input(hidden_states)
        return self.dynamic_moe_op(
            hidden_states=qinput,
            expert_routing_table=expert_routing_table,
            router_weights=router_weights,
            w1=gate_proj_list,
            w2=up_proj_list,
            w3=down_proj_list,
            d_scale_w1=scale_gate_proj,
            d_scale_w2=scale_up_proj,
            d_scale_w3=scale_down_proj,
            d_scale_hidden_states=self.scale_input,
            d_scale_intermediate_hidden_states=self.scale_intermediate,
            permuted_weights=False,
            activation=activation,
            experts_min=self.experts_min,
            experts_max=self.experts_max,
        )
        
    def call_dynamic_moe_measure_op(self,
                                    hidden_states,
                                    expert_routing_table,
                                    router_weights,
                                    permuted_weights=True,
                                    activation="silu"):
        if self._is_fp8:
            gate_proj_list = [self.experts[i].gate_proj.get_dequant_weight() for i in self.experts_range]
            down_proj_list = [self.experts[i].down_proj.get_dequant_weight() for i in self.experts_range]
            up_proj_list = [self.experts[i].up_proj.get_dequant_weight() for i in self.experts_range]
        else:
            gate_proj_list = [self.experts[i].gate_proj.weight for i in self.experts_range]
            down_proj_list = [self.experts[i].down_proj.weight for i in self.experts_range]
            up_proj_list = [self.experts[i].up_proj.weight for i in self.experts_range]
        
        measure_input((hidden_states,), observer=self._mod_extra_config.inputs)
        output, intermediate_amax = torch.ops.hpu.mixture_of_experts.fp8_measurement(
            hidden_states=hidden_states,
            expert_routing_table=expert_routing_table,
            router_weights=router_weights,
            w1=gate_proj_list,
            w2=up_proj_list,
            w3=down_proj_list,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=self.experts_min,
            experts_max=self.experts_max,
            measurement_mode=True,
        )

        amax = []
        for i in range(self.num_experts):
            if i in self.experts_range:
                amax.append(intermediate_amax[i-self.experts_min])
            else:
                amax.append(torch.tensor(0, device="hpu", dtype=intermediate_amax[0].dtype))

        output_measure_list = [output] + amax

        measure_output(output_measure_list, self._mod_extra_config.outputs)
        return output

    
class PatchedVllmMixtureOfExpertsOp(PatchedModuleBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        # Get the `experts_min` and `experts_max` from the original module if they exist
        self.experts_min = self.orig_mod.experts_min if hasattr(self.orig_mod, "experts_min") else 0
        self.experts_max = self.orig_mod.experts_max if hasattr(self.orig_mod, "experts_max") else 7
        self.experts_used = self.local_num_experts if hasattr(self.orig_mod, "local_num_experts") else self.num_experts
        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:

            self.quant_input = self._mod_extra_config.inputs[0]
            self.register_scale("scale_input", mod_extra_config.scale.inputs[0], self.scale_format)
            self.register_scale(
                "scale_intermediate",
                [mod_extra_config.scale.inputs[x] for x in range(1, self.experts_used + 1)],
                self.scale_format,
            )
            self.is_dynamic_quantization = isinstance(self.quant_input, QuantDynamicInput)
            self.dynamic_moe_op = get_quantized_func_wrapper(
                OP_TYPE.DYNAMIC_MOE_FUSED_WEIGHTS, scale_format=self.scale_format, is_dynamic=self.is_dynamic_quantization
            )
            if self.is_dynamic_quantization:
                self.forward = self.forward_dynamic_quant

    def _get_extra_kwargs(self, tokens_num: int):
        kwargs = {}
        if hasattr(self.orig_mod, "_get_extra_kwargs"):
            kwargs = self.orig_mod._get_extra_kwargs(tokens_num)
        return kwargs

    def forward_quant(self,
                      hidden_states,
                      expert_routing_table,
                      router_weights,
                      permuted_weights=True,
                      activation="silu"):
        tokens_num, hidden_dim = hidden_states.shape
        extra_kwargs = self._get_extra_kwargs(tokens_num)
        experts_range = range(self.experts_used)
        w1_list = [self.w13_list[i].weight for i in experts_range]
        w2_list = [self.w2_list[i].weight for i in experts_range]
        scale_w1 = [self.w13_list[i].scale_weight for i in experts_range]
        scale_w2 = [self.w2_list[i].scale_weight for i in experts_range]
        qinput = self.quant_input(hidden_states)
        output = self.dynamic_moe_op(
            hidden_states=qinput,
            expert_routing_table=expert_routing_table,
            router_weights=router_weights,
            w12=w1_list,
            w3=w2_list,
            d_scale_w12=scale_w1,
            d_scale_w3=scale_w2,
            d_scale_hidden_states=self.scale_input,
            d_scale_intermediate_hidden_states=self.scale_intermediate,
            permuted_weights=False,
            activation=activation,
            experts_min=self.experts_min,
            experts_max=self.experts_max,
            **extra_kwargs,
        )
        return output

    def forward_dynamic_quant(
        self, hidden_states, expert_routing_table, router_weights, permuted_weights=True, layer=None, activation="silu"
    ):
        # This is the dynamic version of the forward_quant method.
        # Compared to the `forward_quant` method, the main differences are:
        #   1) The `quant_input` is of type `QuantDynamicInput`.
        #   2) There is no need to pass the `d_scale_intermediate_hidden_states` to the dynamic moe op.
        experts_range = range(self.num_experts)
        w1_list = [self.w13_list[i].weight for i in experts_range]
        w2_list = [self.w2_list[i].weight for i in experts_range]
        scale_w1 = [self.w13_list[i].scale_weight for i in experts_range]
        scale_w2 = [self.w2_list[i].scale_weight for i in experts_range]
        qinput_fp8, input_scale = self.quant_input(hidden_states)
        output = self.dynamic_moe_op(
            hidden_states=qinput_fp8,
            expert_routing_table=expert_routing_table,
            router_weights=router_weights,
            w12=w1_list,
            w3=w2_list,
            d_scale_w12=scale_w1,
            d_scale_w3=scale_w2,
            d_scale_hidden_states=input_scale,
            permuted_weights=False,
            activation=activation,
            experts_min=self.experts_min,
            experts_max=self.experts_max
        )
        return output

    def forward_measure(self,
                        hidden_states,
                        expert_routing_table,
                        router_weights,
                        permuted_weights=True,
                        activation="silu"):
        experts_range = range(self.experts_used)
        w1_list = [self.w13_list[i].weight.squeeze() for i in experts_range]
        w2_list = [self.w2_list[i].weight.squeeze() for i in experts_range]
        measure_input((hidden_states,), observer=self._mod_extra_config.inputs)
        output, intermidiate_amax = torch.ops.hpu.mixture_of_experts.fp8_measurement_fused_weights(
            hidden_states=hidden_states,
            expert_routing_table=expert_routing_table,
            router_weights=router_weights,
            w12=w1_list,
            w3=w2_list,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=self.experts_min,
            experts_max=self.experts_max,
            measurement_mode=True,
        )
        output_measure_list = [output]
        for i in range(self.num_experts):
            output_measure_list.append(intermidiate_amax[i])
        measure_output(output_measure_list, self._mod_extra_config.outputs)
        return output

    def extra_repr(self) -> str:
        member_names = ["scale_input", "scale_intermediate"]
        # for x in range(1, self.num_experts+1):
        #     member_names.append("scale_intermediate["+str(x)+"]")
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, *member_names),
        )


class PatchedVllmMixtureOfExpertsOpFP8(PatchedVllmMixtureOfExpertsOp):
    """The patched module for the VLLMMixtureOfExpertsOp module with FP8 weights.

    There are some models like Deepseek R1/V3 with FP8 weights, we need to requantize it.

    The main difference between this module and the PatchedVllmMixtureOfExpertsOp is that the weights are FP8.
    - At measurement stage, we dequantize the weights to BF16.
    - At quantization stage, we use the same `forward_quant` method as the PatchedVllmMixtureOfExpertsOp.
    """

    def forward_measure(
            self,
            x,
            topk_ids,
            topk_weights,
            permuted_weights=True,
            activation="silu"):
        hidden_states = x
        measure_input((hidden_states,), observer=self._mod_extra_config.inputs)
        min_expert = self.experts_min
        max_expert = self.experts_max
        w13_list_slice = []
        w2_list_slice = []
        for j in range(self.num_experts):
            w13_list_slice.append(self.w13_list[j].get_dequant_weight())
            w2_list_slice.append(self.w2_list[j].get_dequant_weight())

        output, intermidiate_amax = torch.ops.hpu.mixture_of_experts.fp8_measurement_fused_weights(
            hidden_states=x,
            expert_routing_table=topk_ids.to(torch.int64),
            router_weights=topk_weights.to(x.dtype),
            w12=w13_list_slice,
            w3=w2_list_slice,
            permuted_weights=permuted_weights,
            activation=activation,
            experts_min=min_expert,
            experts_max=max_expert,
            measurement_mode=True,
        )
        output_measure_list = [output]
        for i in range(self.experts_used):
            output_measure_list.append(intermidiate_amax[i])
        measure_output(output_measure_list, self._mod_extra_config.outputs)
        return output


class PatchedMoeFP8Matmul(PatchedMoeMatmul):
    """The patched module for the MoeMatmul module with FP8 weights."""

    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.get_dequant_weight = self.orig_mod.get_dequant_weight


class PatchedKVCache(PatchedModuleBase):
    # Module to patch KVCache module from llama model
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        kwargs["func_names"] = ("forward", "get_shape")
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.org_allocate = mod.allocate
        self.org_update = mod.update
        self.forward = self.forward_orig

        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            self.quant_input = self._mod_extra_config.inputs[0]
            self.dequant_output = self._mod_extra_config.outputs[0]
            if self.use_qdq:
                self.qdq_input = self._mod_extra_config.inputs[1]
                self.update = self.update_qdq
                mod.update = self.update_qdq
            else:
                self.update = self.update_quant
                mod.update = self.update_quant
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            self.update = self.update_measure
            mod.update = self.update_measure

    # overwrite allocate function of original module to force allocation in fp8
    def allocate(self, inp_seq_len, dtype, device, shape):
        dtype = torch.float8_e4m3fn if (self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]) else dtype
        return self.org_allocate(inp_seq_len, dtype, device, shape)

    # overwrite update function of original module to force quant and dequant of cache input and output
    def update_qdq(self, prev, cur, dim, idx, inp_seq_len):
        """
         Explanation:  If we want to optimize index_copy so it would run in fp8 instead of bf16
                       we need the tensors to be in fp8 before calling index_copy.
                       Also the `prev` and `curr` tensors need to be of the same dtype - and quanting them both
                       from bf16 is no help, best we can do is have prev be initialized an fp8 tensor from the start.
                       Since the initilization of `prev` is done in OHF (and that is not implemented yet) we
                       currently need to support both options until the implementation in OHF is done, then
                       can we remove the support for the bf16 `prev` option (the else here).
        """
        if prev.dtype == torch.float8_e4m3fn:
            qcurr = self.quant_input(cur)
            qoutput = self.org_update(prev, qcurr, dim, idx, inp_seq_len)
            output = self.dequant_output(qoutput)
        # TODO: remove the `else` part once the lp_dtype is implemented in OHF
        else:
            curr = self.qdq_input(cur)
            output = self.org_update(prev, curr, dim, idx, inp_seq_len)
        return output

    # overwrite update function of original module to force quant and dequant of cache input and output
    def update_quant(self, prev, cur, dim, idx, inp_seq_len):
        qinput = self.quant_input(cur)
        output = self.org_update(prev, qinput, dim, idx, inp_seq_len)
        if output.dtype == torch.float8_e4m3fn:
            return self.dequant_output(output)
        else:
            return output

    # overwrite update function of original module to force quant and dequant of cache input and output
    def update_measure(self, prev, cur, dim, idx, inp_seq_len):
        measure_input((cur,), self._mod_extra_config.inputs)
        output = self.org_update(prev, cur, dim, idx, inp_seq_len)
        measure_output((output,), self._mod_extra_config.outputs)
        return output

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self),
        )


class PatchedVLLMKVCache(PatchedModuleBase):
    # Module to patch VLLMKVCache module from llama model
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            self.quant_input = self._mod_extra_config.inputs[0]
            self.dequant_output = self._mod_extra_config.outputs[0]
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            self.fetch_from_cache = mod.fetch_from_cache

    def forward_qdq(self, input, *args, **kwargs):
        qinput = self.quant_input(input)
        output_cache = self.orig_mod(qinput, *args, **kwargs)
        return output_cache

    def forward_quant(self, input, cache, *args, **kwargs):
        if input is not None:
            qinput = self.quant_input(input)
            output_cache = self.orig_mod(qinput, cache, *args, **kwargs)
        else:
            # In cross-attention during decode stage kv cache isn't updated
            # so input is None and we don't store it
            output_cache = cache
        return self.dequant_output(output_cache)

    def forward_measure(self, input, cache, *args, **kwargs):
        # In cross-attention during decode stage kv cache isn't updated
        # so input is None and we don't measure it
        if input is None:
            return cache
        measure_input((input, ), self._mod_extra_config.inputs)
        output_cache = self.orig_mod(input, cache, *args, **kwargs)
        measure_output((output_cache, ), self._mod_extra_config.outputs)
        return output_cache

    def fetch_from_cache(self, cache, blocks):
        if cache.dtype != self.lp_dtype:
            quant_cache = self.quant_input(cache)
        else:
            quant_cache = cache
        output_cache = self.orig_mod.fetch_from_cache(quant_cache, blocks)
        return self.dequant_output(output_cache)

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self),
        )


def init_conv(instance, mod_extra_config):
    if instance.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
        instance.quant_input = instance._mod_extra_config.inputs[0]
        if instance.use_qdq:
            instance.dequant_weights = mod_extra_config.params["weight"][1]
        else:
            instance.register_scale("scale_input", mod_extra_config.scale.inputs[0], instance.scale_format)
            instance.register_scale("scale_weight", mod_extra_config.scale.params["weight"], instance.scale_format)
            instance.conv2d_fp8 = get_quantized_func_wrapper(OP_TYPE.CONV, instance.scale_format)


class PatchedConv2d(PatchedModuleBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        init_conv(self, mod_extra_config)

    def forward_qdq(self, input):
        qweight = self.dequant_weights(self.weight, )
        qinput = self.quant_input(input)
        output = torch.nn.functional.conv2d(
            qinput,
            qweight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def forward_quant(self, input):
        qinput = self.quant_input(input)
        output = self.conv2d_fp8(qinput,
                                 self.weight,
                                 self.bias,
                                 self.stride,
                                 self.padding,
                                 self.dilation,
                                 self.groups,
                                 out_dtype=self._mod_extra_config.config_params["hp_dtype"],
                                 scale_input_inv=self.scale_input,
                                 scale_other_inv=self.scale_weight)
        return output

    def forward_measure(self, input):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output = self.orig_mod(input)
        measure_output((output,), self._mod_extra_config.outputs)
        return output

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_weight") if not self.use_qdq else "",
        )


class PatchedSoftmaxBase(PatchedModuleBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            self.dequant_output = self._mod_extra_config.outputs[0]
            if self.use_qdq:
                self.quant_input = qdq(nn.Parameter(torch.Tensor([1.0])), self.lp_dtype, self.hp_dtype)
            else:
                # input scale is 1 assuming the input to SM is descaled because we are using HW supported scales
                self.register_scale("scale_input", torch.Tensor([1.0]), self.scale_format)
                self.register_scale("scale_output", torch.Tensor(
                    [1 / mod_extra_config.scale.outputs[0]]), self.scale_format)
            self.softmax_kernel_op = get_quantized_func_wrapper(
                self.get_op_type(), self.scale_format)

    @staticmethod
    @abstractmethod
    def get_op_type():
        """Forward function with Q-DQ mode for quantization stage."""
        raise NotImplementedError("get_op_type is not implemented")

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_output") if not self.use_qdq else "",
        )

class PatchedSoftmax(PatchedSoftmaxBase):

    @staticmethod
    def get_op_type():
        return OP_TYPE.SOFTMAX

    def forward_qdq(self, x, dim=None, invAttnHead=None):
        x = self.quant_input(x)
        output = torch.softmax(x, dim)
        return output

    def forward_quant(self, x, dim=None, invAttnHead=None):
        output = self.softmax_kernel_op(x, dim, self.scale_input, self.scale_output, invAttnHead)
        return self.dequant_output(output)

    def forward_measure(self, x, dim=None, invAttnHead=None):
        measure_input((x,), observer=self._mod_extra_config.inputs)
        output = self.orig_mod(x, dim)
        measure_output((output,), self._mod_extra_config.outputs)
        return output


class PatchedBlockSoftmaxConstMax(PatchedSoftmaxBase):

    @staticmethod
    def get_op_type():
        return OP_TYPE.BLOCK_SOFTMAX_CONST_MAX

    def forward_qdq(self, attn, block_bias, block_groups, batch_size, global_max):
        q_attn = self.quant_input(attn)
        output = self.softmax_kernel_op(q_attn, block_bias, block_groups, batch_size, global_max)
        return output

    def forward_quant(self, attn, block_bias, block_groups, batch_size, global_max):
        output_fp8 = self.softmax_kernel_op(attn, block_bias, block_groups, batch_size, global_max,
                                                  output_scale=self.scale_output, output_dtype=torch.float8_e4m3fn)
        output = self.dequant_output(output_fp8)
        return output

    def forward_measure(self, attn, block_bias, block_groups, batch_size, global_max):
        measure_input((attn,), observer=self._mod_extra_config.inputs)
        output = self.orig_mod(attn, block_bias, block_groups, batch_size, global_max)
        measure_output((output,), self._mod_extra_config.outputs)
        return output


class PatchedLoRACompatibleLinear(PatchedLinearBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.init_linear(mod_extra_config)

    def forward_qdq(self, input, scale: float = 1.0):
        output = self.run_linear_qdq(input, self.bias)
        if self.lora_layer is not None:
            # TODO SW-174899 support lora layer quantization
            _raise_lora_layer_error(self.class_name_org)
        return output

    def forward_quant(self, input, scale: float = 1.0):
        output = self.run_linear_quant(input, self.bias)
        if self.lora_layer is not None:
            # TODO SW-174899 support lora layer quantization
            _raise_lora_layer_error(self.class_name_org)
        return output

    def forward_measure(self, input, scale: float = 1.0):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output = self.orig_mod(input, scale)
        measure_output((output,), self._mod_extra_config.outputs)
        return output


class PatchedLoRACompatibleConv(PatchedModuleBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        init_conv(self, mod_extra_config)

    def forward_qdq(self, input, scale: float = 1.0):
        qinput = self.quant_input(input)
        if self.lora_layer is not None:
            # TODO SW-174899 support lora layer quantization
            _raise_lora_layer_error(self.class_name_org)
        else:
            qweight = self.dequant_weights(self.weight, )
            output = torch.nn.functional.conv2d(
                qinput,
                qweight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return output

    def forward_quant(self, input, scale: float = 1.0):
        qinput = self.quant_input(input)
        if self.lora_layer is not None:
            # TODO SW-174899 support lora layer quantization
            _raise_lora_layer_error(self.class_name_org)
        else:
            output = self.conv2d_fp8(qinput,
                                     self.weight,
                                     self.bias,
                                     self.stride,
                                     self.padding,
                                     self.dilation,
                                     self.groups,
                                     out_dtype=self._mod_extra_config.config_params["hp_dtype"],
                                     scale_input_inv=self.scale_input,
                                     scale_other_inv=self.scale_weight)
        return output

    def forward_measure(self, input, scale: float = 1.0):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output = self.orig_mod(input, scale)
        measure_output((output,), self._mod_extra_config.outputs)
        return output

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_weight") if not self.use_qdq else "",
        )


class PatchedModuleFusedSDPA(PatchedModuleBase):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        # fsdpa is combined out of - BMM1(Q,K) -> Softmax -> BMM2(AMAX,V)
        # during measure we receive the amax value from the cguid and apply it during quant as input
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.fp8_fused_sdpa = get_quantized_func_wrapper(OP_TYPE.FSDPA, self.scale_format)
        if self.quantization_mode in [QuantMode.QUANTIZE, QuantMode.LOAD]:
            self.quant_q = self._mod_extra_config.inputs[0]
            self.quant_k = self._mod_extra_config.inputs[1]
            self.quant_v = self._mod_extra_config.inputs[2]
            self.dequant_output = self._mod_extra_config.outputs[0]
            self.register_scale("scale_q", mod_extra_config.scale.inputs[0].type(torch.float32), self.scale_format)
            self.register_scale("scale_k", mod_extra_config.scale.inputs[1].type(torch.float32), self.scale_format)
            self.register_scale("scale_v", mod_extra_config.scale.inputs[2].type(torch.float32), self.scale_format)
            self.register_scale("descale_amax", mod_extra_config.scale.inputs[3].type(
                torch.float32), self.scale_format)
            self.register_scale("scale_output", 1 /
                                mod_extra_config.scale.outputs[0].type(torch.float32), self.scale_format)
            self.register_scale("scale_amax", 1 / self.descale_amax, self.scale_format)

    def forward_qdq(
        self,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        softmax_mode="None",
        recompute=None,
        valid_seq_len=None,
        seq_padding_type="None",
    ):

        qinput = self.quant_q(q).detach()
        kinput = self.quant_k(k).detach()
        vinput = self.quant_v(v).detach()

        results = self._hpu_kernel_fsdpa.apply(
            qinput,
            kinput,
            vinput,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            softmax_mode,
            recompute,
            valid_seq_len,
            seq_padding_type,
        )
        return results

    def forward_quant(
        self,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        softmax_mode="None",
        recompute=None,
        valid_seq_len=None,
        seq_padding_type="None",
    ):
        sm_mode = softmax_mode if softmax_mode == "fp32" else "None"
        qinput = self.quant_q(q).detach()
        kinput = self.quant_k(k).detach()
        vinput = self.quant_v(v).detach()
        results = self.fp8_fused_sdpa(
            qinput,
            kinput,
            vinput,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            softmax_mode=sm_mode,
            d_scale_q=self.scale_q,
            d_scale_k=self.scale_k,
            d_scale_v=self.scale_v,
            q_scale_s=self.scale_amax,
            q_scale_o=self.scale_output,
            d_scale_s=self.descale_amax,
            is_amax_s=False,
            valid_seq_len=valid_seq_len,
            seq_padding_type=seq_padding_type,
        )
        output = results[0]
        d_out = self.dequant_output(output)
        return d_out

    def forward_measure(
        self,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        softmax_mode="fast",
        recompute=None,
        valid_seq_len=None,
        seq_padding_type="None",
    ):
        dq = q.detach()
        dk = k.detach()
        dv = v.detach()
        measure_input((dq, dk, dv), observer=self._mod_extra_config.inputs)
        results = self.fp8_fused_sdpa(
            dq,
            dk,
            dv,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            # fp8_fused_sdpa in bf16 can use either FastSoftmax or regular
            softmax_mode="fast",
            is_amax_s=True,
            valid_seq_len=valid_seq_len,
            seq_padding_type=seq_padding_type,
        )
        output = results[0]
        amax = results[1]
        measure_output((output, amax), self._mod_extra_config.outputs)
        return output

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(
                self,
                "scale_q",
                "scale_k",
                "scale_v",
                "descale_amax",
                "scale_amax",
                "scale_output",
            ),
        )


class PatchedUnmeasuredModule(nn.Module):
    def __init__(self, name, mod, *args, **kwargs):
        super().__init__()
        if (name == "lm_head" and mod.__class__.__name__ == "ParallelLMHead"):
            # For LM_HEAD when using vLLM (as it doesn't use its forward, but a member's [quant_method] apply).
            self.__dict__.update(mod.__dict__)
            self.quant_method.apply = self.forward
        self.name = name

    def forward(self, *args, **kwargs):
        raise Exception(
            "Error - Layer '{}' was called but was not quantized because no measures were supplied.".format(self.name)
        )

    def extra_repr(self) -> str:
        return f"Dummy patch of {self.name} to raise exception as there are no measurements provided."
