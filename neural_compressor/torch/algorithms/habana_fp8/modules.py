# Copyright (c) 2023 Intel Corporation
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
from torch.nn import functional as F

from neural_compressor.common import logger

# without scale factor 0.9, the output will be abnormal.
E4M3_AMAX = torch.tensor(240 * 0.9, dtype=torch.float).to("hpu")
E5M2_AMAX = torch.tensor(57344 * 0.9, dtype=torch.float).to("hpu")


class FP8DynamicLinear(torch.nn.Module):
    def __init__(self, org_module, dtype=torch.float8_e4m3fn) -> None:
        super().__init__()
        # attributes
        org_module.to("hpu")
        self.use_amax = True
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.weight_dtype = self.dtype
        self.out_dtype = org_module.weight.dtype
        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                device="hpu",
                dtype=self.weight_dtype,
            ),
        )
        self.register_buffer(
            "bias",
            torch.empty(
                self.out_features,
                device="hpu",
                dtype=self.out_dtype,
            ),
        )
        # user configuration
        # scale = HF_max /amax
        if self.use_amax:
            self.weight_scale = self.dtype_amax / org_module.weight.data.abs().max()
            self.weight_scale_inv = torch.reciprocal(self.weight_scale)
        else:
            self.weight_scale = None
            self.weight_scale_inv = None
        self.weight = torch.ops.hpu.cast_to_fp8_v2(org_module.weight.data, self.weight_scale, False, False, self.dtype)[
            0
        ]

        if org_module.bias is not None:
            self.bias = org_module.bias.data.type(self.out_dtype)
        else:
            self.bias = None

    def forward(self, inp):
        assert inp.shape[-1] == self.in_features, "GEMM not possible"
        org_middle_shape = inp.shape[1:-1]
        inp = inp.view((-1, self.in_features))
        if inp.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            if self.use_amax:
                input_scale = self.dtype_amax / inp.abs().max()
                input_scale_inv = torch.reciprocal(input_scale)
            else:
                input_scale, input_scale_inv = None, None
            inp = torch.ops.hpu.cast_to_fp8_v2(inp, input_scale, False, False, self.dtype)[0]
        else:
            input_scale_inv = None
        out = torch.ops.hpu.fp8_gemm_v2(
            inp,
            False,
            self.weight,
            True,
            None,
            self.out_dtype,
            input_scale_inv,  # inv is used for recover scale
            self.weight_scale_inv,
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
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.out_dtype = torch.float32

    def forward(self, input1, input2):
        dim1 = input1.shape[-1]
        dim2 = input2.shape[-2]
        assert dim1 == dim2, "GEMM not possible"

        # process input1
        if input1.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            self.out_dtype = input1.dtype
            if self.use_amax:
                input1_scale = self.dtype_amax / input1.data.abs().max()
                input1_scale_inv = torch.reciprocal(input1_scale)
            else:
                input1_scale, input1_scale_inv = None, None
            input1 = torch.ops.hpu.cast_to_fp8_v2(input1, input1_scale, False, False, self.dtype)[0]
        else:
            # skip cast for input1
            input1_scale, input1_scale_inv = None, None
        # process input2
        if input2.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            self.out_dtype = input2.dtype
            if self.use_amax:
                input2_scale = self.dtype_amax / input2.data.abs().max()
                input2_scale_inv = torch.reciprocal(input2_scale)
            else:
                input2_scale, input2_scale_inv = None, None
            input2 = torch.ops.hpu.cast_to_fp8_v2(input2, input2_scale, False, False, self.dtype)[0]
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
            input1_scale_inv,  # inv is used for recover scale
            input2_scale_inv,
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
        org_module.to("hpu")
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.weight_dtype = self.dtype
        self.out_dtype = org_module.weight.dtype
        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                device="hpu",
                dtype=self.weight_dtype,
            ),
        )
        self.register_buffer(
            "bias",
            torch.empty(
                self.out_features,
                device="hpu",
                dtype=self.out_dtype,
            ),
        )
        assert hasattr(org_module, "scale"), "scale is not recorded when convert to FP8Linear."
        self.register_buffer(
            "scale",
            torch.tensor(
                org_module.scale,
                device="hpu",
                dtype=torch.float32,
            ),
        )
        self.scale_inv = torch.reciprocal(self.scale)

        self.weight_scale = self.dtype_amax / org_module.weight.data.abs().max()
        self.weight_scale_inv = torch.reciprocal(self.weight_scale)
        self.weight = torch.ops.hpu.cast_to_fp8_v2(org_module.weight.data, self.weight_scale, False, False, self.dtype)[
            0
        ]

        if org_module.bias is not None:
            self.bias = org_module.bias.data.type(self.out_dtype)
        else:
            self.bias = None

    def forward(self, inp):
        assert inp.shape[-1] == self.in_features, "GEMM not possible"
        org_middle_shape = inp.shape[1:-1]
        inp = inp.view((-1, self.in_features))
        inp = torch.ops.hpu.cast_to_fp8_v2(inp, self.scale, False, False, self.dtype)[0]
        out = torch.ops.hpu.fp8_gemm_v2(
            inp,
            False,
            self.weight,
            True,
            None,
            self.out_dtype,
            self.scale_inv,  # inv is used for recover scale
            self.weight_scale_inv,
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
            self.scale,
            self.dtype,
        )


class FP8Matmul(torch.nn.Module):
    def __init__(self, org_module, dtype) -> None:
        super().__init__()
        org_module.to("hpu")
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.out_dtype = torch.float32
        assert hasattr(org_module, "scale") and hasattr(
            org_module, "scale1"
        ), "scale is not recorded when convert to FP8Linear."
        self.register_buffer(
            "scale",
            torch.tensor(
                org_module.scale,
                device="hpu",
                dtype=self.out_dtype,
            ),
        )
        self.register_buffer(
            "scale1",
            torch.tensor(
                org_module.scale1,
                device="hpu",
                dtype=self.out_dtype,
            ),
        )
        self.input1_scale_inv = torch.reciprocal(self.scale)
        self.input2_scale_inv = torch.reciprocal(self.scale1)

    def forward(self, input1, input2):
        dim1 = input1.shape[-1]
        dim2 = input2.shape[-2]
        assert dim1 == dim2, "GEMM not possible"

        if input1.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            self.out_dtype = input1.dtype
            input1 = torch.ops.hpu.cast_to_fp8_v2(input1, self.scale, False, False, self.dtype)[0]
        else:
            self.input1_scale_inv = None
        if input2.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            self.out_dtype = input2.dtype
            input2 = torch.ops.hpu.cast_to_fp8_v2(input2, self.scale1, False, False, self.dtype)[0]
        else:
            self.input2_scale_inv = None
        out = torch.ops.hpu.fp8_gemm_v2(
            input1,
            False,
            input2,
            False,
            None,
            self.out_dtype,
            self.input1_scale_inv,  # inv is used for recover scale
            self.input2_scale_inv,
            None,
            False,
        )
        return out

    def extra_repr(self) -> str:
        return "scales={}, format={}".format(
            (self.scale, self.scale1),
            self.dtype,
        )


class FP8BatchMatmul(FP8Matmul):
    pass


class FP8Cast(torch.nn.Module):
    def __init__(self, org_module=None, dtype=torch.float8_e4m3fn) -> None:
        super().__init__()
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        if org_module is not None:
            org_module.to("hpu")
            assert hasattr(org_module, "scale"), "scale is not recorded when convert to FP8Cast."
            self.register_buffer(
                "scale",
                torch.tensor(
                    org_module.scale,
                    device="hpu",
                    dtype=torch.float32,
                ),
            )
            self.scale = None  # due to next matmul doesn't know this scale
        else:
            self.scale = None

    def forward(self, input):
        if input.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            out = torch.ops.hpu.cast_to_fp8_v2(input, self.scale, False, False, self.dtype)[0]
        else:
            out = input
        return out

    def extra_repr(self) -> str:
        return "scales={}, format={}".format(
            (self.scale),
            self.dtype,
        )


class FP8LinearLayer(torch.nn.Module):
    def __init__(self, org_module, dtype) -> None:
        super().__init__()
        # attributes
        org_module.to("hpu")
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.in_features = org_module.weight.shape[1]
        self.out_features = org_module.weight.shape[0]
        self.weight_dtype = self.dtype
        self.out_dtype = org_module.weight.dtype
        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                device="hpu",
                dtype=self.weight_dtype,
            ),
        )
        assert hasattr(org_module, "scale"), "scale is not recorded when convert to FP8Linear."
        self.register_buffer(
            "scale",
            torch.tensor(
                org_module.scale,
                device="hpu",
                dtype=torch.float32,
            ),
        )
        self.scale_inv = 1.0 / self.scale
        # user configuration
        # scale = HF_max /amax
        self.weight_scale = self.dtype_amax / org_module.weight.data.abs().max()
        self.weight_scale_inv = 1.0 / self.weight_scale
        self.weight = torch.ops.hpu.cast_to_fp8_v2(org_module.weight.data, self.weight_scale, False, False, self.dtype)[
            0
        ]
        if org_module.bias is not None:
            self.register_buffer(
                "bias",
                torch.empty(
                    self.out_features,
                    device="hpu",
                    dtype=self.out_dtype,
                ),
            )
            self.bias = org_module.bias.data.type(self.out_dtype)
        else:
            self.bias = None

    def forward(self, inp):
        assert inp.shape[-1] == self.in_features, "GEMM not possible"
        inputmat = inp.view((-1, self.in_features))
        inputmat = torch.ops.hpu.cast_to_fp8_v2(inputmat, self.scale, False, False, self.dtype)[0]
        out = torch.ops.hpu.fp8_gemm_v2(
            inputmat,
            False,
            self.weight,
            True,
            None,
            self.out_dtype,
            self.scale_inv,  # inv is used for recover scale
            self.weight_scale_inv,
            None,
            False,
        )
        if self.bias is not None:
            out += self.bias
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, scale={}, format={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.scale,
            self.dtype,
        )


class FP8LinearAllreduce(torch.nn.Module):
    def __init__(self, org_module, dtype) -> None:
        super().__init__()
        # attributes
        org_module.to("hpu")
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.in_features = org_module.weight.shape[1]
        self.out_features = org_module.weight.shape[0]
        self.weight_dtype = self.dtype
        self.out_dtype = org_module.weight.dtype
        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                device="hpu",
                dtype=self.weight_dtype,
            ),
        )
        assert hasattr(org_module, "scale"), "scale is not recorded when convert to FP8Linear."
        self.register_buffer(
            "scale",
            torch.tensor(
                org_module.scale,
                device="hpu",
                dtype=torch.float32,
            ),
        )
        self.scale_inv = 1.0 / self.scale
        # user configuration
        # scale = HF_max /amax
        self.weight_scale = self.dtype_amax / org_module.weight.data.abs().max()
        self.weight_scale_inv = 1.0 / self.weight_scale
        self.weight = torch.ops.hpu.cast_to_fp8_v2(org_module.weight.data, self.weight_scale, False, False, self.dtype)[
            0
        ]
        if org_module.bias is not None:
            self.register_buffer(
                "bias",
                torch.empty(
                    self.out_features,
                    device="hpu",
                    dtype=self.out_dtype,
                ),
            )
            self.bias = org_module.bias.data.type(self.out_dtype)
        else:
            self.bias = None
        self.mp_group = org_module.mp_group

    def forward(self, inp):
        assert inp.shape[-1] == self.in_features, "GEMM not possible"
        inputmat = inp.view((-1, self.in_features))
        inputmat = torch.ops.hpu.cast_to_fp8_v2(inputmat, self.scale, False, False, self.dtype)[0]
        out = torch.ops.hpu.fp8_gemm_v2(
            inputmat,
            False,
            self.weight,
            True,
            None,
            self.out_dtype,
            self.scale_inv,  # inv is used for recover scale
            self.weight_scale_inv,
            None,
            False,
        )
        from deepspeed import comm as dist

        if self.mp_group is not None:
            dist.inference_all_reduce(out, group=self.mp_group)
        if self.bias is not None:
            out += self.bias
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, scale={}, format={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.scale,
            self.dtype,
        )


class FP8LmHeadLinearAllreduce(torch.nn.Module):
    def __init__(self, org_module, dtype) -> None:
        super().__init__()
        # attributes
        org_module.to("hpu")
        self.dtype = dtype
        self.dtype_amax = E4M3_AMAX if self.dtype == torch.float8_e4m3fn else E5M2_AMAX
        self.in_features = org_module.weight.shape[1]
        self.out_features = org_module.weight.shape[0]
        self.weight_dtype = self.dtype
        self.out_dtype = org_module.weight.dtype
        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                device="hpu",
                dtype=self.weight_dtype,
            ),
        )
        assert hasattr(org_module, "scale"), "scale is not recorded when convert to FP8Linear."
        self.register_buffer(
            "scale",
            torch.tensor(
                org_module.scale,
                device="hpu",
                dtype=torch.float32,
            ),
        )
        self.scale_inv = 1.0 / self.scale
        # user configuration
        # scale = HF_max /amax
        self.weight_scale = self.dtype_amax / org_module.weight.data.abs().max()
        self.weight_scale_inv = 1.0 / self.weight_scale
        self.weight = torch.ops.hpu.cast_to_fp8_v2(org_module.weight.data, self.weight_scale, False, False, self.dtype)[
            0
        ]
        if org_module.bias is not None:
            self.register_buffer(
                "bias",
                torch.empty(
                    self.out_features,
                    device="hpu",
                    dtype=self.out_dtype,
                ),
            )
            self.bias = org_module.bias.data.type(self.out_dtype)
        else:
            self.bias = None
        self.mp_group = org_module.mp_group
        self.rank = org_module.rank
        self.world_size = org_module.world_size

    def forward(self, inp):
        # from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list
        # input_shard_size = get_shard_size(inp.shape[-1], self.world_size)
        # input_shard_offset = sum(get_shard_size_list(inp.shape[-1], self.world_size)[0:self.rank])

        # inputmat = inp[:, :, input_shard_offset:input_shard_offset + input_shard_size]
        assert (
            inp.shape[-1] % self.world_size == 0
        ), "Please ensure that self.world_size is divisible by input.shape[-1]"
        input_shard = inp.shape[-1] // self.world_size
        inputmat = inp[:, :, self.rank * input_shard : (self.rank + 1) * input_shard]
        inputmat = torch.ops.hpu.cast_to_fp8_v2(inputmat, self.scale, False, False, self.dtype)[0]
        out = torch.ops.hpu.fp8_gemm_v2(
            inputmat,
            False,
            self.weight,
            True,
            None,
            self.out_dtype,
            self.scale_inv,  # inv is used for recover scale
            self.weight_scale_inv,
            None,
            False,
        )
        from deepspeed import comm as dist

        if self.mp_group is not None:
            dist.inference_all_reduce(out, group=self.mp_group)
        if self.bias is not None:
            out += self.bias
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, scale={}, format={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.scale,
            self.dtype,
        )
