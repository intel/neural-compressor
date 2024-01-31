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

from typing import Any, Dict, Tuple, Union

import torch
from auto_accelerator import auto_detect_accelerator
from bitpack import BitPack
from config import HQQModuleConfig, QTensorConfig, QTensorMetaInfo, default_hqq_module_config
from optimizer import optimize_weights_proximal
from utility import custom_print, dump_elapsed_time, get_tensor_size, inspect_function, is_divisible

__all__ = [
    "HQQTensorHandle",
    "HQQLinear",
    "QTensor",
]


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

    def _get_scale_repr(self) -> str:
        if not self.is_scale_quantized():
            if self.scale is not None:
                return f"scale_shape={self.scale.shape}, scale_dtype={self.scale.dtype}, scale_device={self.scale.device}\n"
            else:
                return "scale is None\n"
        else:
            return self.scale.__repr__() + "\n"

    def _get_zero_repr(self) -> str:
        if not self.is_zero_quantized():
            if self.zero is not None:
                return f"zero_shape={self.zero.shape}, zero_dtype={self.zero.dtype}, zero_device={self.zero.device}\n"
            else:
                return "zero is None\n"
        else:
            return self.zero.__repr__() + "\n"

    def __repr__(self) -> str:
        # TODO: refine it later
        return (
            f"QTensor(\n"
            f"val_shape={self.val.shape}, val_dtype={self.val.dtype}, val_device={self.val.device}\n"
            f"scale_quantized={self.is_scale_quantized()},\n"
            f"zero_quantized={self.is_zero_quantized()},\n"
            f"zero=({self._get_zero_repr()})"
            f"scale=({self._get_scale_repr()})"
            f"meta_info={self.meta_info}\n)"
        )

    def to(self, *args, **kwargs):
        self.val = self.val.to(*args, **kwargs)
        self.scale = self.scale.to(*args, **kwargs)
        self.zero = self.zero.to(*args, **kwargs)
        return self

    def get_size(self) -> int:
        result = 0
        result += get_tensor_size(self.val)
        if isinstance(self.scale, QTensor):
            result += get_tensor_size(self.scale.val)
        else:
            result += get_tensor_size(self.scale)
        if isinstance(self.zero, QTensor):
            result += get_tensor_size(self.zero.val)
        else:
            result += get_tensor_size(self.zero)
        return result


class HQQTensorHandle:
    # Refactor the code from https://github.com/mobiusml/hqq.

    # Store meta-data (we invert the scale for dequantization)
    SUPPORTED_BITS = [8, 4, 3, 2]
    optimize_weights = optimize_weights_proximal

    # TODO: Refine the packer
    bit_to_packing = {8: "8bit_u8", 4: "4bit_u8", 3: "3bit_32", 2: "2bit_u8"}

    pack = {
        "8bit_u8": BitPack.pack_8bit_u8,
        "4bit_u8": BitPack.pack_4bit_u8,
        "3bit_32": BitPack.pack_3bit_32,
        "2bit_u8": BitPack.pack_2bit_u8,
    }

    unpack = {
        "8bit_u8": BitPack.unpack_8bit_u8,
        "4bit_u8": BitPack.unpack_4bit_u8,
        "3bit_32": BitPack.unpack_3bit_32,
        "2bit_u8": BitPack.unpack_2bit_u8,
    }

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
    @dump_elapsed_time("HQQTensorHandle.quantize_to_q_tensor")
    def quantize_to_q_tensor(cls, float_tensor, tensor_quant_config: QTensorConfig = None):
        q_weight, q_tensor_meta = cls.quantize(
            tensor=float_tensor,
            tensor_quant_config=tensor_quant_config,
        )
        q_weight = cls._create_q_tensor_from_q_weight_and_meta(q_weight, q_tensor_meta)
        return q_weight

    @classmethod
    # @inspect_function
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
            "packing": cls.bit_to_packing[nbits],
        }

        # Pack bits
        if bitpack:
            # raise NotImplementedError("bitpack is not implemented yet")
            W_q = cls.pack[meta["packing"]](W_q)
            print("packing: weight...")
        else:
            W_q = W_q.to(tensor.dtype)
            meta["packing"] = None

        # cleanup
        del W, _min, _max
        auto_detect_accelerator().empty_cache()

        return W_q, meta

    # Main dequantization: bit_unpacking > (W_q - z)*s > reshape
    @classmethod
    # @inspect_function
    def dequantize(cls, W_q, meta):
        if meta["packing"]:
            W_r = cls.unpack[meta["packing"]](W_q).half()
            if (meta["group_size"] is not None) and (meta["nbits"] == 3):
                W_r = W_r[: meta["group_size"]] if (meta["axis"] == 0) else W_r[:, : meta["group_size"]]
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
        weight, scale, zero = (
            quant_config.weight,
            quant_config.scale,
            quant_config.zero,
        )
        need_quant_scale = scale is not None
        need_quant_zero = zero is not None

        self.in_features, self.out_features = W.t().shape

        # Quantize weight
        q_weight = HQQTensorHandle.quantize_to_q_tensor(float_tensor=W, tensor_quant_config=weight)
        self.q_weight = q_weight

        # * The dequantization process only happens in the first forward pass.
        # * It will change the `q_weight` but faster, so we should not save the state after doing the forward.
        if need_quant_scale:  # Quantize scale
            custom_print(message=f"need_quant_scale: {need_quant_scale}")
            q_scale_tensor = HQQTensorHandle.quantize_to_q_tensor(
                float_tensor=self.q_weight.scale, tensor_quant_config=scale
            )
            self.q_weight.scale = q_scale_tensor
        if need_quant_zero:  # Quantize zero
            custom_print(f"need_quant_zero: {need_quant_zero}")
            q_zero_tensor = HQQTensorHandle.quantize_to_q_tensor(
                float_tensor=self.q_weight.zero,
                tensor_quant_config=zero,
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
