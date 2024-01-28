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
sys.path.insert(1, hqq_offical_path)

from collections import namedtuple
from typing import Any, Dict, Optional, Tuple, Union

import torch
from accelerator import auto_detect_accelerator

# from hqq
from hqq.core.common.optim_utils import optimize_weights_proximal
from hqq.core.common.utils import custom_print, inspect_function, is_divisible

__all__ = [
    "QuantTensorConfig",
    "HQQModuleConfig",
    "default_hqq_quant_config",
    "QTensor",
    "HQQTensorHandle",
    "HQQLinear",
]


class QuantTensorConfig:
    def __init__(
        self,
        nbits: int,
        channel_wise: bool = True,
        group_size: int = 128,
        optimize: bool = True,
        round_zero: Optional[bool] = False,
        pack: bool = False,
    ) -> None:
        self.nbits = nbits
        self.channel_wise = channel_wise
        self.group_size = group_size
        self.optimize = optimize
        self.round_zero = round_zero
        self.pack = pack

    def __repr__(self) -> str:
        return (
            f"QuantTensorConfig(nbits={self.nbits}, channel_wise={self.channel_wise}, "
            f"group_size={self.group_size}, optimize={self.optimize}, "
            f"round_zero={self.round_zero}, pack={self.pack})"
        )


default_weight_quant_config = QuantTensorConfig(
    nbits=4, channel_wise=True, group_size=128, optimize=True, round_zero=True
)
default_scale_quant_config = QuantTensorConfig(
    nbits=8, channel_wise=True, group_size=64, optimize=False, round_zero=None
)
default_zero_quant_config = QuantTensorConfig(
    nbits=8, channel_wise=False, group_size=None, optimize=False, round_zero=None
)


class HQQModuleConfig(
    namedtuple(
        "HQQModuleConfig",
        ["weight_quant_config", "scale_quant_config", "zero_quant_config"],
    )
):
    def __new__(
        cls,
        weight_quant_config=default_weight_quant_config,
        scale_quant_config=default_scale_quant_config,
        zero_quant_config=default_zero_quant_config,
    ):
        return super().__new__(cls, weight_quant_config, scale_quant_config, zero_quant_config)

    def __repr__(self) -> str:
        return (
            f"HQQModuleConfig( \nweight_quant_config={self.weight_quant_config}, \n"
            f"scale_quant_config={self.scale_quant_config}, \n"
            f"zero_quant_config={self.zero_quant_config})"
        )


default_hqq_quant_config = HQQModuleConfig(
    weight_quant_config=default_weight_quant_config,
    scale_quant_config=default_scale_quant_config,
    zero_quant_config=default_zero_quant_config,
)


class QTensorMetaInfo:
    def __init__(self, nbits, group_size, shape, axis, packing):
        self.nbits = nbits
        self.group_size = group_size
        self.shape = shape
        self.axis = axis
        self.packing = packing

    def __repr__(self) -> str:
        return (
            f"QTensorMetaInfo(nbits={self.nbits}, group_size={self.group_size}, "
            f"shape={self.shape}, axis={self.axis}, packing={self.packing})"
        )


class QTensor:
    val: torch.Tensor
    scale: Union[torch.Tensor, "QTensor"] = None
    zero: Union[torch.Tensor, "QTensor"] = None
    meta_info: QTensorMetaInfo = None

    def __init__(self, val, scale=None, zero=None, meta_info=None):
        self.val = val
        self.scale = scale
        self.zero = zero
        self.meta_info = meta_info

    def is_scale_quantized(self) -> bool:
        return isinstance(self.scale, QTensor)

    def is_zero_quantized(self) -> bool:
        return isinstance(self.zero, QTensor)

    def __repr__(self) -> str:
        return (
            f"QTensor(shape={self.val.shape}, scale_quantized={self.is_scale_quantized()}, "
            f"zero_quantized={self.is_zero_quantized()}, meta_info={self.meta_info}"
        )


class HQQTensorHandle:
    # The mostly copied from https://github.com/mobiusml/hqq.

    # Store meta-data (we invert the scale for dequantization)
    SUPPORTED_BITS = [8, 4, 3, 2]
    optimize_weights = optimize_weights_proximal
    accelerator = auto_detect_accelerator()

    @classmethod
    def _convert_tensor_quant_config(cls, tensor_quant_config: QuantTensorConfig):
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
    def quantize_to_q_tensor(cls, float_tensor, tensor_quant_config: QuantTensorConfig = None):
        q_weight, q_tensor_meta = cls.quantize(
            tensor=float_tensor,
            tensor_quant_config=tensor_quant_config,
        )
        q_weight = cls._create_q_tensor_from_q_weight_and_meta(q_weight, q_tensor_meta)
        return q_weight

    @classmethod
    @inspect_function
    def quantize(cls, tensor, tensor_quant_config: QuantTensorConfig = None):
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
            "packing": None,
        }

        # Pack bits
        if bitpack:
            raise NotImplementedError("bitpack is not implemented yet")
            # W_q = Quantizer.pack[meta["packing"]](W_q)
        else:
            W_q = W_q.to(tensor.dtype)
            meta["packing"] = None

        # cleanup
        del W, _min, _max
        cls.accelerator.empty_cache()

        return W_q, meta

    # Main dequantization: bit_unpacking > (W_q - z)*s > reshape
    @classmethod
    @inspect_function
    def dequantize(cls, W_q, meta):
        if meta["packing"]:
            raise NotImplementedError("bitpack is not implemented yet")
            # W_r = Quantizer.unpack[meta['packing']](W_q).half()
            # if((meta['group_size'] is not None) and (meta['nbits']==3)):
            #     W_r = W_r[:meta['group_size']] if (meta['axis']==0) else W_r[:,:meta['group_size']]
        else:
            W_r = W_q.half()
        # custom_print(f"W_r dtype: {W_r.dtype}, zero dtype: {meta['zero'].dtype}, scale dtype: {meta['scale'].dtype}")

        W_r = ((W_r - meta["zero"]) * meta["scale"]).reshape(meta["shape"])
        W_r = W_r.half()  # TODO: double check the correctness, the official impl is also error...
        # custom_print(f"After dq .... W_r dtype: {W_r.dtype}, zero dtype: {meta['zero'].dtype}, scale dtype: {meta['scale'].dtype}")
        return W_r

    @classmethod
    def _convert_meta_info_to_dict(cls, meta_info: "QTensorMetaInfo"):
        # TODO: to refine it
        return {
            "nbits": meta_info.nbits,
            "group_size": meta_info.group_size,
            "shape": meta_info.shape,
            "axis": meta_info.axis,
            "packing": meta_info.packing,
        }

    @classmethod
    def dequantize_q_tensor(cls, q_weight: "QTensor") -> torch.Tensor:
        # Dequantized the Qtensor into float tensor
        meta = cls._convert_meta_info_to_dict(q_weight.meta_info)
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

    def quantize_weight(
        self,
        W: torch.Tensor,
        quant_config: HQQModuleConfig = default_hqq_quant_config,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        weight_quant_config, scale_quant_config, zero_quant_config = (
            quant_config.weight_quant_config,
            quant_config.scale_quant_config,
            quant_config.zero_quant_config,
        )
        need_quant_scale = scale_quant_config is not None
        need_quant_zero = zero_quant_config is not None

        self.in_features, self.out_features = W.t().shape

        # Quantize weight
        q_weight = HQQTensorHandle.quantize_to_q_tensor(float_tensor=W, tensor_quant_config=weight_quant_config)
        self.q_weight = q_weight

        # * The dequantization process only happens in the first forward pass.
        # * It will change the `q_weight` but faster, so we should not save the state after doing the forward.
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
        # TODO: Should we delete the q_weight and q_weight_meta?
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
        quant_config: HQQModuleConfig = default_hqq_quant_config,
    ):
        # create the new module with a toy size to ensure initialization is fast
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
        # TODO: should we delete the float_module explicitly?
        return new_mod
