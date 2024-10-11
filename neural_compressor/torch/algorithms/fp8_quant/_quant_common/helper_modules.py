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

import torch
import torch.nn as nn

from .._core.quant_dequant import QuantDequant as qdq
from .._core.scale_handler import create_scale_tensor
from .quant_config import QuantMode, ScaleFormat, get_hqt_config

try:  # backwards compatibility for 1.16
    from habana_frameworks.torch.hpex.kernels import fp8_fused_sdpa
except ImportError:
    pass


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


def matmul_fp8(
    input,
    other,
    out=None,
    out_dtype=torch.bfloat16,
    scale_input_inv=None,
    scale_other_inv=None,
):
    res = torch.ops.hpu.fp8_gemm_v2(
        input,
        False,
        other,
        False,
        out,
        out_dtype,
        scale_input_inv,
        scale_other_inv,
        None,
        False,
    )
    return res


def measure_input(input, observer):
    for i in range(len(observer)):
        observer[i].measure(input[i])


def measure_output(output, observer):
    if observer:
        for i in range(len(observer)):
            observer[i].measure(output[i])


def conv2d_fp8(
    input,
    other,
    bias,
    stride,
    padding,
    dilation,
    groups,
    out_dtype=torch.bfloat16,
    scale_input_inv=None,
    scale_other_inv=None,
):
    return torch.ops.hpu.conv2d_fp8(
        input=input,
        weight=other,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        out_dtype=out_dtype,
        scale_input=scale_input_inv,
        scale_weight=scale_other_inv,
    )


def set_attrs_from_orig_model(cls_instance, mod, mod_extra_config, *func_names):
    cls_instance.__dict__.update(mod.__dict__)
    config = get_hqt_config(cls_instance)
    cls_instance.extra_repr_org = mod.extra_repr
    cls_instance.class_name_org = mod.__class__.__name__
    cls_instance._mod_extra_config = mod_extra_config
    cls_instance.quantization_mode = config.cfg["mode"]
    cls_instance.fake_quant = config.cfg["fake_quant"]
    cls_instance.scale_format = config.cfg["scale_format"]
    # store original module in order to invoke its functions during measurements.
    # this may be omitted of torch remove the related validation from dynamo. see SW-187731.
    cls_instance.__dict__["orig_mod"] = mod
    cls_instance.forward_orig = mod.forward
    if func_names is not None:
        for func in func_names:
            setattr(cls_instance, func, getattr(mod, func))


def get_current_repr(cls_instance, *member_names):
    curr_repr = ""
    if cls_instance.quantization_mode == QuantMode.QUANTIZE:
        first_name = True
        for name in member_names:
            if not first_name:
                curr_repr += ", "
            curr_repr += f"{name} dtype={getattr(cls_instance, name).dtype}"
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


class PatchedMatmul(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        if self.quantization_mode == QuantMode.QUANTIZE:
            if not self.fake_quant:
                self.forward = self.forward_quant
                self.quant_input_0 = self._mod_extra_config.inputs[0]
                self.quant_input_1 = self._mod_extra_config.inputs[1]
                self.scale_input = create_scale_tensor(mod_extra_config.scale.inputs[0], self.scale_format)
                self.scale_other = create_scale_tensor(mod_extra_config.scale.inputs[1], self.scale_format)
            else:
                self.forward = self.forward_fakequant

                # override quantization to quant-dequant
                mec = self._mod_extra_config.inputs[0]
                self.quant_input_0 = qdq(mec.scale_inv, mec.lp_dtype, mec.hp_dtype)
                mec = self._mod_extra_config.inputs[1]
                self.quant_input_1 = qdq(mec.scale_inv, mec.lp_dtype, mec.hp_dtype)

        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            self.forward = self.forward_measure

    def forward_quant(self, input, other):
        qinput = self.quant_input_0(input)
        qother = self.quant_input_1(other)
        output = matmul_fp8(
            qinput,
            qother,
            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
            scale_input_inv=self.scale_input,
            scale_other_inv=self.scale_other,
        )
        return output

    def forward_fakequant(self, input, other):
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
            get_current_repr(self, "scale_input", "scale_other"),
        )


def init_linear(instance, mod_extra_config):
    if instance.quantization_mode == QuantMode.QUANTIZE:
        # When offloading weights to disk using device_map, the module forward is overridden.
        # __dict__.update call again overrides the PatchedLinear forward with the forward that device_map planted.
        # So need to set PatchedLinear forawrd to be the right forward.
        instance.forward = instance.forward_quant
        instance.quant_input = instance._mod_extra_config.inputs[0]
        instance.quant_output = instance._mod_extra_config.outputs[0]
        instance.weight = nn.Parameter(instance.weight.t().contiguous())
        instance.scale_input = create_scale_tensor(mod_extra_config.scale.inputs[0], instance.scale_format)
        if isinstance(mod_extra_config.scale.params["weight"], (torch.Tensor, float)):
            instance.scale_weight = create_scale_tensor(mod_extra_config.scale.params["weight"], instance.scale_format)
        elif isinstance(mod_extra_config.scale.params["weight"], dict):
            # PCQ weight is calculated with actual weight [0] and ones [1]
            instance.scale_weight = nn.Parameter(mod_extra_config.scale.params["weight"][0])
            # When offloading weights to disk using device_map, the module forward is overridden.
            # __dict__.update call again overrides the PatchedLinear forward with the forward that device_map planted.
            # So need to set PatchedLinear forawrd to be the right forward.
            instance.forward = instance.forward_quant
            instance.quant_input = instance._mod_extra_config.inputs[0]
    elif (instance.quantization_mode == QuantMode.MEASURE) or (instance.quantization_mode == QuantMode.SHAPE):
        instance.forward = instance.forward_measure


class PatchedLinear(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        init_linear(self, mod_extra_config)
        if self.fake_quant and self.quantization_mode == QuantMode.QUANTIZE:
            self.forward = self.forward_fakequant
            # override quantization to quant-dequant
            mec = self._mod_extra_config.inputs[0]
            self.quant_input = qdq(mec.scale_inv, mec.lp_dtype, mec.hp_dtype)
            mec = self._mod_extra_config.params["weight"]
            self.quant_weights = qdq(mec.scale_inv, mec.lp_dtype, mec.hp_dtype)

    def forward_fakequant(self, input):
        qweight = self.quant_weights(
            self.weight,
        )
        qinput = self.quant_input(input)
        y = torch.matmul(qinput, qweight)
        output = y + self.bias if (self.bias is not None) else y
        return output

    def forward_quant(self, input):
        qinput = self.quant_input(input)
        y = matmul_fp8(
            qinput,
            self.weight,
            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
            scale_input_inv=self.scale_input,
            scale_other_inv=self.scale_weight,
        )
        output = y + self.bias if (self.bias is not None) else y
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
            get_current_repr(self, "scale_input", "scale_weight"),
        )


# patched vllm module without measure and quant
# measure and quant of the weights is done thru PatchedMoeMatmul
class PatchedMixtralMoE(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        # remove the MoE weights that are quanted by PatchedMoeMatmul
        delattr(mod, "w13_weight")
        delattr(mod, "w2_weight")
        setattr(mod, "w13_weight", None)
        setattr(mod, "w2_weight", None)
        self.forward = mod.forward


class PatchedMoeMatmul(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        init_linear(self, mod_extra_config)
        if self.quantization_mode == QuantMode.MEASURE:
            mod.weight = mod.weight.t()

    # The calc method is called by the vllm-mixtral MoE gate layer
    # we patch it so that during quantized inference run it will use
    # our internal quantized weight member for calculating the chosen expert.
    # Therefore we ignore the expert_id and weight parameters used by the orig calc.
    def calc(self, state, expert_id, w):
        return self.forward(state)

    def forward_quant(self, input):
        qinput = self.quant_input(input)
        y = matmul_fp8(
            qinput,
            self.weight,
            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
            scale_input_inv=self.scale_input,
            scale_other_inv=self.scale_weight,
        )
        return y

    def forward_measure(self, input):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output = self.forward_orig(input)
        measure_output((output,), self._mod_extra_config.outputs)
        return output

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_weight"),
        )


class PatchedReplicatedLinear(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        init_linear(self, mod_extra_config)

    def forward_quant(self, input):
        qinput = self.quant_input(input)
        bias = self.bias if not self.skip_bias_add else None
        y = matmul_fp8(
            qinput,
            self.weight,
            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
            scale_input_inv=self.scale_input,
            scale_other_inv=self.scale_weight,
        )
        output = y + bias if (bias is not None) else y
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def forward_measure(self, input):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output, output_bias = self.forward_orig(input)
        measure_output((output,), self._mod_extra_config.outputs)
        return output, output_bias

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_weight"),
        )


class PatchedLinearAllReduce(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        init_linear(self, mod_extra_config)
        self.scoped_version = mod.__class__.__name__ == "ScopedLinearAllReduce"

    def forward_quant(self, input):
        # pre_all_reduce
        qinput = self.quant_input(input)
        output = matmul_fp8(
            qinput,
            self.weight,
            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
            scale_input_inv=self.scale_input,
            scale_other_inv=self.scale_weight,
        )
        dqoutput = self.quant_output(output)
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

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_weight"),
        )


class PatchedRowParallelLinear(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config, "resolve_input")
        init_linear(self, mod_extra_config)

    def forward_quant(self, input):
        resolved_input = self.resolve_input(input)
        qinput = self.quant_input(resolved_input)
        output = matmul_fp8(
            qinput,
            self.weight,
            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
            scale_input_inv=self.scale_input,
            scale_other_inv=self.scale_weight,
        )
        dqoutput = self.quant_output(output)
        if self.reduce_results:
            dqoutput = self.collective_func(dqoutput)
        return self.post_all_reduce(dqoutput)

    def forward_measure(self, input):
        resolved_input = self.resolve_input(input)
        measure_input((resolved_input,), observer=self._mod_extra_config.inputs)
        output = torch.matmul(resolved_input, self.weight.transpose(-1, -2))
        measure_output((output,), self._mod_extra_config.outputs)
        if self.reduce_results:
            output = self.collective_func(output)
        return self.post_all_reduce(output)

    def post_all_reduce(self, output):
        assert (
            self.reduce_results or (not self.bias) or self.skip_bias_add
        ), "When not reduce the results, adding bias to the results can lead to incorrect results"
        if not self.skip_bias_add:
            output = output + self.bias if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias
        return output, output_bias

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_weight"),
        )


class PatchedColumnParallelLinear(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        init_linear(self, mod_extra_config)

    def forward_quant(self, input):
        qinput = self.quant_input(input)
        output = matmul_fp8(
            qinput,
            self.weight,
            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
            scale_input_inv=self.scale_input,
            scale_other_inv=self.scale_weight,
        )
        dqoutput = self.quant_output(output)
        if self.gather_output:
            dqoutput = self.orig_mod.collective_func(dqoutput)
        return self.post_all_reduce(dqoutput)

    def forward_measure(self, input):
        measure_input((input,), observer=self._mod_extra_config.inputs)
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        measure_output((output,), self._mod_extra_config.outputs)
        if self.gather_output:
            output = self.orig_mod.collective_func(output)
        return self.post_all_reduce(output)

    def post_all_reduce(self, output):
        if not self.skip_bias_add:
            output = output + self.bias if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias
        return output, output_bias

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_weight"),
        )


class PatchedLmHeadLinearAllreduce(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        init_linear(self, mod_extra_config)

    def forward_quant(self, input):
        assert (
            input.shape[-1] % self.world_size == 0
        ), "Please ensure that self.world_size is divisible by input.shape[-1]"
        input_shard = input.shape[-1] // self.world_size
        splittedInput = input[:, :, self.rank * input_shard : (self.rank + 1) * input_shard]
        qinput = self.quant_input(splittedInput)
        output = matmul_fp8(
            qinput,
            self.weight,
            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
            scale_input_inv=self.scale_input,
            scale_other_inv=self.scale_weight,
        )
        dqoutput = self.quant_output(output)

        if self.mp_group is not None:
            from deepspeed import comm as dist

            dist.inference_all_reduce(dqoutput, group=self.mp_group)
        if self.bias is not None:
            dqoutput += self.bias
        return dqoutput

    def forward_measure(self, input):
        assert (
            input.shape[-1] % self.world_size == 0
        ), "Please ensure that self.world_size is divisible by input.shape[-1]"
        input_shard = input.shape[-1] // self.world_size
        splittedInput = input[:, :, self.rank * input_shard : (self.rank + 1) * input_shard]
        measure_input((splittedInput,), observer=self._mod_extra_config.inputs)
        output = torch.matmul(splittedInput, self.weight.t())
        measure_output((output,), self._mod_extra_config.outputs)

        if self.mp_group is not None:
            from deepspeed import comm as dist

            dist.inference_all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_weight"),
        )


class PatchedKVCache(nn.Module):
    # Module to patch KVCache module from llama model
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config, "forward", "get_shape")
        self.org_allocate = mod.allocate
        self.org_update = mod.update
        if self.quantization_mode == QuantMode.QUANTIZE:
            mod.update = self.update
            self.quant_input = self._mod_extra_config.inputs[0]
            self.quant_output = self._mod_extra_config.outputs[0]
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            self.update = self.update_measure
            mod.update = self.update_measure

    # overwrite allocate function of original module to force allocation in fp8
    def allocate(self, inp_seq_len, dtype, device, shape):
        dtype = torch.float8_e4m3fn if (self.quantization_mode == QuantMode.QUANTIZE) else dtype
        return self.org_allocate(inp_seq_len, dtype, device, shape)

    # overwrite update function of original module to force quant and dequant of cache input and output
    def update(self, prev, cur, dim, idx, inp_seq_len):
        qinput = self.quant_input(cur)
        output = self.org_update(prev, qinput, dim, idx, inp_seq_len)
        if output.dtype == torch.float8_e4m3fn:
            return self.quant_output(output)
        else:
            return output

    # overwrite update function of original module to force quant and dequant of cache input and output
    def update_measure(self, prev, cur, dim, idx, inp_seq_len):
        measure_input((cur,), self._mod_extra_config.inputs)
        output = self.org_update(prev, cur, dim, idx, inp_seq_len)
        measure_output((output,), self._mod_extra_config.outputs)
        return output


class PatchedVLLMKVCache(nn.Module):
    # Module to patch VLLMKVCache module from llama model
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        if self.quantization_mode == QuantMode.QUANTIZE:
            self.quant_input = self._mod_extra_config.inputs[0]
            self.quant_output = self._mod_extra_config.outputs[0]
            self.orig_fetch_from_cache = mod.fetch_from_cache
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            self.fetch_from_cache = mod.fetch_from_cache
            self.forward = self.forward_measure

    def forward(self, input, cache, num_kv_cache_passes, num_slots_available, block_indices, block_offset):
        qinput = self.quant_input(input)
        output_cache = self.forward_orig(
            qinput, cache, num_kv_cache_passes, num_slots_available, block_indices, block_offset
        )
        return self.quant_output(output_cache)

    def forward_measure(self, input, cache, num_kv_cache_passes, num_slots_available, block_indices, block_offset):
        measure_input((input), self._mod_extra_config.inputs)
        output_cache = self.forward_orig(
            input, cache, num_kv_cache_passes, num_slots_available, block_indices, block_offset
        )
        measure_output((output_cache), self._mod_extra_config.outputs)
        return output_cache

    def fetch_from_cache(self, cache, blocks, permutations=None):
        quant_cache = self.quant_input(cache)
        if permutations:
            output_cache = self.orig_fetch_from_cache(quant_cache, blocks, permutations)
            for i in range(len(output_cache)):
                output_cache[i] = self.quant_output(output_cache[i])
            return output_cache
        output_cache = self.orig_fetch_from_cache(quant_cache, blocks)
        return self.quant_output(output_cache)


class PatchedConv2d(nn.Conv2d):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        if self.quantization_mode == QuantMode.QUANTIZE:
            self.quant_input = self._mod_extra_config.inputs[0]
            self.scale_input = create_scale_tensor(mod_extra_config.scale.inputs[0], self.scale_format)
            self.scale_weight = create_scale_tensor(mod_extra_config.scale.params["weight"], self.scale_format)
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            self.forward = self.forward_measure

    def forward(self, input):
        qinput = self.quant_input(input)
        output = conv2d_fp8(
            qinput,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
            scale_input_inv=self.scale_input,
            scale_other_inv=self.scale_weight,
        )
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
            get_current_repr(self, "scale_input", "scale_weight"),
        )


class PatchedSoftmax(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        self.scale_format = ScaleFormat.CONST
        if self.quantization_mode == QuantMode.QUANTIZE:
            self.quant_output = self._mod_extra_config.outputs[0]
            self.scale_input = create_scale_tensor(torch.Tensor([1.0]), self.scale_format)
            self.scale_output = create_scale_tensor(
                torch.Tensor([1 / mod_extra_config.scale.outputs[0]]), self.scale_format
            )
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            self.forward = self.forward_measure

    def forward(self, x, dim=None, invAttnHead=None):
        output = torch.ops.hpu.softmax_fp8(x, dim, self.scale_input, self.scale_output, invAttnHead)
        return self.quant_output(output)

    def forward_measure(self, x, dim=None, invAttnHead=None):
        measure_input((x,), observer=self._mod_extra_config.inputs)
        output = self.orig_mod(x, dim, invAttnHead)
        measure_output((output,), self._mod_extra_config.outputs)
        return output

    def extra_repr(self) -> str:
        return extra_representation(
            self.extra_repr_org(),
            self.class_name_org,
            get_current_repr(self, "scale_input", "scale_output"),
        )


class PatchedLoRACompatibleLinear(nn.Linear):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        init_linear(self, mod_extra_config)

    def forward_quant(self, input, scale: float = 1.0):
        qinput = self.quant_input(input)
        y = matmul_fp8(
            qinput,
            self.weight,
            out_dtype=self._mod_extra_config.config_params["hp_dtype"],
            scale_input_inv=self.scale_input,
            scale_other_inv=self.scale_weight,
        )
        output = y + self.bias if (self.bias is not None) else y
        if self.lora_layer is not None:
            # TODO SW-174899 support lora layer quantization
            _raise_lora_layer_error(self.class_name_org)
            # output = output + (scale * self.lora_layer(input))
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
            get_current_repr(self, "scale_input", "scale_weight"),
        )


class PatchedLoRACompatibleConv(nn.Conv2d):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        if self.quantization_mode == QuantMode.QUANTIZE:
            self.quant_input = self._mod_extra_config.inputs[0]
            self.scale_input = create_scale_tensor(mod_extra_config.scale.inputs[0], self.scale_format)
            self.scale_weight = create_scale_tensor(mod_extra_config.scale.params["weight"], self.scale_format)
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            self.forward = self.forward_measure

    def forward(self, input, scale: float = 1.0):
        qinput = self.quant_input(input)
        if self.lora_layer is not None:
            # TODO SW-174899 support lora layer quantization
            _raise_lora_layer_error(self.class_name_org)
            # output = conv2d_fp8(qinput, self.weight, None, self.stride, self.padding, self.dilation, self.groups, \
            #  out_dtype=self._mod_extra_config.config_params["hp_dtype"], scale_input_inv=self.scale_input, scale_other_inv=self.scale_weight)
            # output = output + (scale * self.lora_layer(input))
            # output = output+torch.unsqueeze(torch.unsqueeze(self.bias,1), 1) if (self.bias is not None) else output
        else:
            output = conv2d_fp8(
                qinput,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                out_dtype=self._mod_extra_config.config_params["hp_dtype"],
                scale_input_inv=self.scale_input,
                scale_other_inv=self.scale_weight,
            )
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
            get_current_repr(self, "scale_input", "scale_weight"),
        )


class PatchedModuleFusedSDPA(nn.Module):
    def __init__(self, mod, mod_extra_config, *args, **kwargs):
        # fsdpa is combined out of - BMM1(Q,K) -> Softmax -> BMM2(AMAX,V)
        # during measure we receive the amax value from the cguid and apply it during quant as input
        super().__init__()
        set_attrs_from_orig_model(self, mod, mod_extra_config)
        if self.quantization_mode == QuantMode.QUANTIZE:
            self.quant_q = self._mod_extra_config.inputs[0]
            self.quant_k = self._mod_extra_config.inputs[1]
            self.quant_v = self._mod_extra_config.inputs[2]
            self.dequant_output = self._mod_extra_config.outputs[0]
            # fsdpa currently doesn't support scalar scales so scale format is always const.
            # should be fixed once SW-199793 is done
            self.scale_q = create_scale_tensor(mod_extra_config.scale.inputs[0].type(torch.float32), ScaleFormat.CONST)
            self.scale_k = create_scale_tensor(mod_extra_config.scale.inputs[1].type(torch.float32), ScaleFormat.CONST)
            self.scale_v = create_scale_tensor(mod_extra_config.scale.inputs[2].type(torch.float32), ScaleFormat.CONST)
            self.descale_amax = create_scale_tensor(
                mod_extra_config.scale.inputs[3].type(torch.float32), ScaleFormat.CONST
            )
            self.scale_output = create_scale_tensor(
                1 / mod_extra_config.scale.outputs[0].type(torch.float32), ScaleFormat.CONST
            )
            self.scale_amax = create_scale_tensor(1 / self.descale_amax, ScaleFormat.CONST)
        elif (self.quantization_mode == QuantMode.MEASURE) or (self.quantization_mode == QuantMode.SHAPE):
            self.forward = self.forward_measure

    def forward(
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
        results = fp8_fused_sdpa(
            qinput,
            kinput,
            vinput,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            # fp8_fused_sdpa in fp8 mode supports only FastSoftmax
            softmax_mode="None",
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
        results = fp8_fused_sdpa(
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
    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self.name = name

    def forward(self, *args, **kwargs):
        raise Exception(
            "Error - Layer '{}' was called but was not quantized because no measures were supplied.".format(self.name)
        )

    def extra_repr(self) -> str:
        return f"Dummy patch of {self.name} to raise exception as there are no measurements provided."
