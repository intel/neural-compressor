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

import sys

# TODO: remove it before merge
hqq_offical_path = "/home/yliu7/workspace/hqq"
sys.path.insert(0, hqq_offical_path)


from typing import Any, Dict, Tuple

import torch
from auto_accelerator import auto_detect_accelerator
from bitpack import Packer
from config import HQQModuleConfig, QTensorConfig, default_hqq_module_config, hqq_global_option
from optimizer import optimize_weights_proximal
from qtensor import QTensor, QTensorMetaInfo
from utility import custom_print, dump_elapsed_time, get_tensor_size, inspect_function, is_divisible

__all__ = [
    "HQQTensorHandle",
    "HQQLinear",
]


class HQQTensorHandle:
    # Refactor the code from https://github.com/mobiusml/hqq.

    # Store meta-data (we invert the scale for dequantization)
    SUPPORTED_BITS = [8, 4, 3, 2]
    optimize_weights = optimize_weights_proximal

    @classmethod
    def _convert_tensor_quant_config(cls, tensor_quant_config: QTensorConfig):
        nbits = tensor_quant_config.nbits
        channel_wise = tensor_quant_config.channel_wise
        group_size = tensor_quant_config.group_size
        optimize = tensor_quant_config.optimize
        round_zero = tensor_quant_config.round_zero
        axis = 0  # *Note did not exposed to the user
        bitpack = tensor_quant_config.pack
        return nbits, channel_wise, group_size, optimize, round_zero, axis, bitpack

    @classmethod
    def _create_q_tensor_from_q_weight_and_meta(cls, weight, meta) -> "QTensor":
        scale = meta["scale"]
        zero = meta["zero"]
        meta_info = QTensorMetaInfo(
            nbits=meta["nbits"],
            group_size=meta["group_size"],
            shape=meta["shape"],
            axis=meta["axis"],
            packing=meta["packing"],
        )
        return QTensor(weight, scale, zero, meta_info)

    @classmethod
    def quantize_to_q_tensor(cls, float_tensor, tensor_quant_config: QTensorConfig = None):
        q_weight, q_tensor_meta = cls.quantize(
            tensor=float_tensor,
            tensor_quant_config=tensor_quant_config,
        )
        q_weight = cls._create_q_tensor_from_q_weight_and_meta(q_weight, q_tensor_meta)
        return q_weight

    @classmethod
    def quantize(cls, tensor, tensor_quant_config: QTensorConfig = None):
        (
            nbits,
            channel_wise,
            group_size,
            optimize,
            round_zero,
            axis,
            bitpack,
        ) = cls._convert_tensor_quant_config(tensor_quant_config)
        custom_print("start quantize ..... ")
        assert nbits in cls.SUPPORTED_BITS, "nbits=" + str(nbits) + " not supported."
        assert axis in [0, 1], "axis should be either 0 or 1, but got {}".format(axis)
        if group_size is not None:
            assert is_divisible(tensor.numel(), group_size), (
                "group_size should be divisible by the total tensor dimensions. shape: "
                + str(tensor.shape)
                + ", group_size: "
                + str(group_size)
            )

        W = tensor.float()
        shape = W.shape

        # Reshape for grouping
        if (group_size is not None) and channel_wise:
            W = W.reshape([-1, group_size]) if (axis == 1) else W.reshape([group_size, -1])

        # Get min/max values
        if not channel_wise:
            _min, _max = W.min(), W.max()
            optimize = False
        else:
            _min = W.min(axis=axis, keepdim=True)[0]
            _max = W.max(axis=axis, keepdim=True)[0]

        max_v = 2**nbits - 1
        min_v = 0
        min_max = [min_v, max_v]

        # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
        scale = (max_v / (_max - _min)).clamp(max=2e4)  # clamp to avoid half-precision problems
        zero = -_min * scale

        # Round zero as in: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py#L42C9-L42C14
        if round_zero:
            zero = torch.round(zero)

        # Fine-tune weights
        if optimize:
            scale, zero = cls.optimize_weights(tensor=W, scale=scale, zero=zero, min_max=min_max, axis=axis)

        # Quantize
        scale, zero = (
            scale.clone(),
            zero.clone(),
        )  # Necessary for fake quantization backprop
        W_q = torch.round(W * scale + zero).clamp(min_max[0], min_max[1])

        # Store meta-data (we invert the scale for dequantization)
        meta = {
            "nbits": nbits,
            "group_size": group_size,
            "shape": shape,
            "scale": 1.0 / scale,
            "zero": zero,
            "axis": axis,
            "packing": bitpack,
        }

        # Pack bits
        if bitpack:
            W_q = Packer.get_pack_fn(meta["nbits"])(W_q)
        else:
            W_q = W_q.to(tensor.dtype)
            meta["packing"] = None

        # cleanup
        del W, _min, _max
        auto_detect_accelerator().empty_cache()

        return W_q, meta

    @classmethod
    def dequantize(cls, W_q, meta):
        # Main dequantization: bit_unpacking > (W_q - z)*s > reshape
        if meta["packing"]:
            W_r = Packer.get_unpack_fn(meta["nbits"])(W_q).half()
            if (meta["group_size"] is not None) and (meta["nbits"] == 3):
                W_r = W_r[: meta["group_size"]] if (meta["axis"] == 0) else W_r[:, : meta["group_size"]]
        else:
            if hqq_global_option.use_half:
                W_r = W_q.half()
        # !!! TODO: There may cause the accuracy regression issue
        W_r = ((W_r - meta["zero"]) * meta["scale"]).reshape(meta["shape"])
        # W_r = W_r.half()  # TODO: double check the correctness, the official impl is also error...
        return W_r

    @classmethod
    def dequantize_q_tensor(cls, q_weight: "QTensor") -> torch.Tensor:
        # Dequantized the Qtensor into float tensor
        meta = q_weight.meta_info.to_dict()
        meta["zero"] = q_weight.zero
        meta["scale"] = q_weight.scale
        return cls.dequantize(q_weight.val, meta)


class HQQLinear(torch.nn.Linear):
    # The design follows the https://github.com/pytorch-labs/ao.
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        q_weight: QTensor = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.q_weight = q_weight
        self.quantized = q_weight is not None

    def get_size(self):
        # TODO: for debug only, remove it before merge
        result = 0
        weight = self.weight
        result += get_tensor_size(weight)
        result += self.q_weight.get_size()
        return result

    def quantize_weight(
        self,
        W: torch.Tensor,
        quant_config: HQQModuleConfig = default_hqq_module_config,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        weight_quant_config, scale_quant_config, zero_quant_config = (
            quant_config.weight,
            quant_config.scale,
            quant_config.zero,
        )
        need_quant_scale = scale_quant_config is not None
        need_quant_zero = zero_quant_config is not None

        self.in_features, self.out_features = W.t().shape

        # Quantize weight
        q_weight = HQQTensorHandle.quantize_to_q_tensor(float_tensor=W, tensor_quant_config=weight_quant_config)
        self.q_weight = q_weight

        # * The dequantization process only happens in the first forward pass.
        # * It will change the `q_weight` but faster.
        # * we should not save the state after doing the forward.
        if need_quant_scale:  # Quantize scale
            custom_print(message=f"need_quant_scale: {need_quant_scale}")
            q_scale_tensor = HQQTensorHandle.quantize_to_q_tensor(
                float_tensor=self.q_weight.scale, tensor_quant_config=scale_quant_config
            )
            self.q_weight.scale = q_scale_tensor
        if need_quant_zero:  # Quantize zero
            custom_print(f"need_quant_zero: {need_quant_zero}")
            q_zero_tensor = HQQTensorHandle.quantize_to_q_tensor(
                float_tensor=self.q_weight.zero,
                tensor_quant_config=zero_quant_config,
            )
            self.q_weight.zero = q_zero_tensor
        self.quantized = True

    def dequantize_weight(self):
        assert self.quantized, "model was not quantized"
        # TODO: move below logic into `HQQTensorHandle`
        if self.q_weight.is_scale_quantized():
            scale_qdq = HQQTensorHandle.dequantize_q_tensor(self.q_weight.scale)
            self.q_weight.scale = scale_qdq
            custom_print(f"scale_qdq dtype: {scale_qdq.dtype}")

        if self.q_weight.is_zero_quantized():
            zero_qdq = HQQTensorHandle.dequantize_q_tensor(self.q_weight.zero)
            self.q_weight.zero = zero_qdq
            custom_print(f"zero_qdq dtype: {zero_qdq.dtype}")

        W_qdq = HQQTensorHandle.dequantize_q_tensor(self.q_weight)
        return W_qdq

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(input, self.dequantize_weight().t())
        if self.bias is not None:
            out += self.bias
        return out

    @classmethod
    def from_float(
        cls,
        float_module: torch.nn.Linear,
        quant_config: HQQModuleConfig = default_hqq_module_config,
    ):
        # Create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features,
            fake_out_features,
            bias=float_module.bias is not None,
        )
        new_mod.requires_grad_ = False
        # Construct the q weight frpm float weight
        new_mod.quantize_weight(float_module.weight, quant_config=quant_config)
        # Update the linear module attributes
        new_mod.in_features = float_module.in_features
        new_mod.out_features = float_module.out_features
        new_mod.weight = None
        new_mod.bias = float_module.bias
        # TODO: refine it to support cuda/hpu/cpu
        device_to_use = next(float_module.parameters()).device
        new_mod.to(device_to_use)
        new_mod.q_weight.to(device_to_use)
        # !!! Delete the float explicitly to save memory
        del float_module
        return new_mod
