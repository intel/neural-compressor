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

# pylint:disable=import-error

import os

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpex
import torch
import torch.nn as nn
from torch.nn import functional as F

from neural_compressor.common import logger

from .observer import calculate_qparams


##################### FP32 modules #######################
class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class BatchMatmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)


class Autocast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


##################### FP8 modules #######################
class FP8DynamicLinear(torch.nn.Module):
    def __init__(self, org_module, dtype=torch.float8_e4m3fn) -> None:
        super().__init__()
        # attributes
        self.use_amax = True
        self.dtype = dtype
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.weight_dtype = self.dtype
        self.out_dtype = org_module.weight.dtype
        # register weight, bias
        self.register_buffer(
            "weight",
            torch.empty(
                self.in_features,
                self.out_features,
                device="hpu",
                dtype=self.weight_dtype,
            ),
        )
        if org_module.bias is not None:
            self.register_buffer(
                "bias",
                torch.empty(
                    self.out_features,
                    device="hpu",
                    dtype=self.out_dtype,
                ),
            )
        else:
            self.bias = None

    def from_float(self, org_module, w_observer):
        # register scale
        if not org_module.weight.device.type == "meta":
            w_observer(org_module.weight)
            weight_scale = w_observer.calculate_qparams()
        else:
            weight_scale = torch.tensor([1.0])
        self.register_buffer(
            "weight_scale",
            torch.tensor(
                weight_scale,
                device="hpu",
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "weight_scale_inv",
            torch.tensor(
                torch.reciprocal(weight_scale),
                device="hpu",
                dtype=torch.float32,
            ),
        )
        # copy weight and bias
        if not org_module.weight.device.type == "meta":
            org_module.to("hpu")
            self.weight.data.copy_(
                torch.ops.hpu.cast_to_fp8_v2(org_module.weight.T, self.weight_scale_inv, False, False, self.dtype)[0]
            )
            if org_module.bias is not None:
                self.bias.data.copy_(org_module.bias.data.type(self.out_dtype))

    def forward(self, inp):
        assert inp.shape[-1] == self.in_features, "GEMM not possible"
        org_middle_shape = inp.shape[1:-1]
        inp = inp.view(-1, self.in_features)
        if inp.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            if self.use_amax:
                input_scale = calculate_qparams(inp.min(), inp.max(), self.dtype)
                input_scale_inv = torch.reciprocal(input_scale)
            else:
                input_scale, input_scale_inv = None, None
            inp = torch.ops.hpu.cast_to_fp8_v2(inp, input_scale_inv, False, False, self.dtype)[0]
        else:
            input_scale, input_scale_inv = None, None
        out = torch.ops.hpu.fp8_gemm_v2(
            inp,
            False,
            self.weight,
            False,
            None,
            self.out_dtype,
            input_scale,  # inv is used for recover scale
            self.weight_scale,
            self.bias,
            False,
        )
        out = out.view(-1, *org_middle_shape, out.shape[-1])
        return out

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, format={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.dtype,
        )


class FP8DynamicMatmul(torch.nn.Module):
    def __init__(self, dtype) -> None:
        super().__init__()
        self.dtype = dtype
        self.use_amax = True
        self.out_dtype = torch.float32

    def forward(self, input1, input2):
        dim1 = input1.shape[-1]
        dim2 = input2.shape[-2]
        assert dim1 == dim2, "GEMM not possible"

        # process input1
        if input1.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            self.out_dtype = input1.dtype
            if self.use_amax:
                input1_scale = calculate_qparams(input1.min(), input1.max(), self.dtype)
                input1_scale_inv = torch.reciprocal(input1_scale)
            else:
                input1_scale, input1_scale_inv = None, None
            input1 = torch.ops.hpu.cast_to_fp8_v2(input1, input1_scale_inv, False, False, self.dtype)[0]
        else:
            # skip cast for input1
            input1_scale, input1_scale_inv = None, None
        # process input2
        if input2.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            self.out_dtype = input2.dtype
            if self.use_amax:
                input2_scale = calculate_qparams(input2.min(), input2.max(), self.dtype)
                input2_scale_inv = torch.reciprocal(input2_scale)
            else:
                input2_scale, input2_scale_inv = None, None
            input2 = torch.ops.hpu.cast_to_fp8_v2(input2, input2_scale_inv, False, False, self.dtype)[0]
        else:
            # skip cast for input2
            input2_scale, input2_scale_inv = None, None
        # calculate
        out = torch.ops.hpu.fp8_gemm_v2(
            input1,
            False,
            input2,
            False,
            None,
            self.out_dtype,
            input1_scale,  # inv is used for recover scale
            input2_scale,
            None,
            False,
        )
        return out

    def extra_repr(self) -> str:
        return "format={}".format(self.dtype)


class FP8DynamicBatchMatmul(FP8DynamicMatmul):
    pass


class FP8Linear(torch.nn.Module):
    def __init__(self, org_module, dtype) -> None:
        super().__init__()
        # attributes
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.dtype = dtype
        self.weight_dtype = self.dtype
        self.out_dtype = org_module.weight.dtype
        self.register_buffer(
            "weight",
            torch.empty(
                self.in_features,
                self.out_features,
                device="hpu",
                dtype=self.weight_dtype,
            ),
        )
        if org_module.bias is not None:
            self.register_buffer(
                "bias",
                torch.empty(
                    self.out_features,
                    device="hpu",
                    dtype=self.out_dtype,
                ),
            )
        else:
            self.bias = None

    def from_float(self, org_module, w_observer):
        # register scale
        if not org_module.weight.device.type == "meta":
            w_observer(org_module.weight)
            weight_scale = w_observer.calculate_qparams()
        else:
            weight_scale = torch.tensor([1.0])
        self.register_buffer(
            "weight_scale",
            torch.tensor(
                weight_scale,
                device="hpu",
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "weight_scale_inv",
            torch.tensor(
                torch.reciprocal(weight_scale),
                device="hpu",
                dtype=torch.float32,
            ),
        )
        # copy weight and bias
        if not org_module.weight.device.type == "meta":
            org_module.to("hpu")
            self.weight.data.copy_(
                torch.ops.hpu.cast_to_fp8_v2(org_module.weight.T, self.weight_scale_inv, False, False, self.dtype)[0]
            )
            if org_module.bias is not None:
                self.bias.data.copy_(org_module.bias.data.type(self.out_dtype))
        # register input scale
        input_scale = org_module.scale if hasattr(org_module, "scale") else torch.tensor([1.0])
        self.register_buffer(
            "input_scale",
            torch.tensor(
                input_scale,
                device="hpu",
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "input_scale_inv",
            torch.tensor(
                torch.reciprocal(input_scale),
                device="hpu",
                dtype=torch.float32,
            ),
        )

    def forward(self, inp):
        assert inp.shape[-1] == self.in_features, "GEMM not possible"
        org_middle_shape = inp.shape[1:-1]
        inp = inp.view(-1, self.in_features)
        inp = torch.ops.hpu.cast_to_fp8_v2(inp, self.input_scale_inv, False, False, self.dtype)[0]
        out = torch.ops.hpu.fp8_gemm_v2(
            inp,
            False,
            self.weight,
            False,
            None,
            self.out_dtype,
            self.input_scale,  # inv is used for recover scale
            self.weight_scale,
            self.bias,
            False,
        )
        out = out.view(-1, *org_middle_shape, out.shape[-1])
        return out

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, scale={}, format={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.input_scale.tolist() if hasattr(self, "input_scale") else None,
            self.dtype,
        )


class FP8Matmul(torch.nn.Module):
    def __init__(self, org_module, dtype) -> None:
        super().__init__()
        org_module.to("hpu")
        self.dtype = dtype
        self.out_dtype = torch.float32
        scale1 = org_module.scale1 if hasattr(org_module, "scale1") else 1.0
        scale2 = org_module.scale2 if hasattr(org_module, "scale2") else 1.0
        self.register_buffer(
            "scale1",
            torch.tensor(
                scale1,
                device="hpu",
                dtype=self.out_dtype,
            ),
        )
        self.register_buffer(
            "scale2",
            torch.tensor(
                scale2,
                device="hpu",
                dtype=self.out_dtype,
            ),
        )

    def forward(self, input1, input2):
        dim1 = input1.shape[-1]
        dim2 = input2.shape[-2]
        assert dim1 == dim2, "GEMM not possible"

        if input1.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            self.out_dtype = input1.dtype
            self.scale1_inv = torch.reciprocal(self.scale1)
            input1 = torch.ops.hpu.cast_to_fp8_v2(input1, self.scale1_inv, False, False, self.dtype)[0]
        else:
            self.scale1_inv = None
        if input2.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            self.out_dtype = input2.dtype
            self.scale2_inv = torch.reciprocal(self.scale2)
            input2 = torch.ops.hpu.cast_to_fp8_v2(input2, self.scale2_inv, False, False, self.dtype)[0]
        else:
            self.scale2_inv = None
        out = torch.ops.hpu.fp8_gemm_v2(
            input1,
            False,
            input2,
            False,
            None,
            self.out_dtype,
            self.scale1,  # inv is used for recover scale
            self.scale2,
            None,
            False,
        )
        return out

    def extra_repr(self) -> str:
        return "scales={}, format={}".format(
            (self.scale1.tolist(), self.scale2.tolist()),
            self.dtype,
        )


class FP8BatchMatmul(FP8Matmul):
    pass


class FP8Cast(torch.nn.Module):
    def __init__(self, org_module=None, dtype=torch.float8_e4m3fn) -> None:
        super().__init__()
        self.dtype = dtype
        if org_module is not None:
            org_module.to("hpu")
            scale = org_module.scale if hasattr(org_module, "scale") else 1.0
            self.register_buffer(
                "scale",
                torch.tensor(
                    scale,
                    device="hpu",
                    dtype=torch.float32,
                ),
            )
            self.scale, self.scale_inv = None, None  # due to next matmul doesn't know this scale
        else:
            self.scale, self.scale_inv = None, None

    def forward(self, input):
        if input.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            out = torch.ops.hpu.cast_to_fp8_v2(input, self.scale_inv, False, False, self.dtype)[0]
        else:
            out = input
        return out

    def extra_repr(self) -> str:
        return "scales={}, format={}".format(
            self.scale,
            self.dtype,
        )


FP8LinearLayer = FP8Linear


class FP8LinearAllreduce(FP8Linear):
    def forward(self, inp):
        assert inp.shape[-1] == self.in_features, "GEMM not possible"
        inputmat = inp.view(-1, self.in_features)
        inputmat = torch.ops.hpu.cast_to_fp8_v2(inputmat, self.input_scale_inv, False, False, self.dtype)[0]
        out = torch.ops.hpu.fp8_gemm_v2(
            inputmat,
            False,
            self.weight,
            False,
            None,
            self.out_dtype,
            self.input_scale,
            self.weight_scale,
            None,
            False,
        )
        from deepspeed import comm as dist

        if self.mp_group is not None:
            dist.inference_all_reduce(out, group=self.mp_group)
        if self.bias is not None:
            out += self.bias
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])


class FP8LmHeadLinearAllreduce(FP8Linear):
    def forward(self, inp):
        # from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list
        # input_shard_size = get_shard_size(inp.shape[-1], self.world_size)
        # input_shard_offset = sum(get_shard_size_list(inp.shape[-1], self.world_size)[0:self.rank])

        # inputmat = inp[:, :, input_shard_offset:input_shard_offset + input_shard_size]
        assert (
            inp.shape[-1] % self.world_size == 0
        ), "Please ensure that self.world_size is divisible by input.shape[-1]"
        input_shard = inp.shape[-1] // self.world_size
        inp_part = inp[:, :, self.rank * input_shard : (self.rank + 1) * input_shard]
        inputmat = inp_part.view(-1, input_shard)  # dim=2 will help kernel speed
        inputmat = torch.ops.hpu.cast_to_fp8_v2(inputmat, self.input_scale_inv, False, False, self.dtype)[0]
        out = torch.ops.hpu.fp8_gemm_v2(
            inputmat,
            False,
            self.weight,
            False,
            None,
            self.out_dtype,
            self.input_scale,
            self.weight_scale,
            None,
            False,
        )
        from deepspeed import comm as dist

        if self.mp_group is not None:
            dist.inference_all_reduce(out, group=self.mp_group)
        if self.bias is not None:
            out += self.bias
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])
