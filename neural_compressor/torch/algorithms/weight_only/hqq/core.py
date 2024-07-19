# Copyright (c) 2023-2024 Mobiusml and Intel Corporation

# This code is based on Mobiusml's HQQ library. It has been modified
# from its original forms to simplify and adapt it for use in
# the Intel® Neural Compressor.

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

# NOTICE: the original `Quantizer` has been modified to `HQQTensorHandle`
# and `QTensor` to decouple the data structure and the quantization logic.
"""The HQQ modules."""


from typing import Any, Dict, Mapping, Tuple

import torch

from neural_compressor.common.utils import dump_elapsed_time
from neural_compressor.torch.utils import logger
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator

from .bitpack import Packer
from .config import HQQModuleConfig, QTensorConfig, default_hqq_module_config, hqq_global_option
from .optimizer import optimize_weights_proximal
from .qtensor import QTensor, QTensorMetaInfo

__all__ = [
    "HQQTensorHandle",
    "HQQLinear",
]


class HQQTensorHandle:
    """HQQ Tensor Handle to quantize and dequantize the tensor."""

    # Refactored the code from https://github.com/mobiusml/hqq.

    # Store meta-data (we invert the scale for dequantization)
    SUPPORTED_BITS = [8, 4, 3, 2]
    optimize_weights = optimize_weights_proximal

    @classmethod
    def quantize(cls, float_tensor, tensor_quant_config: QTensorConfig = None):
        """Quantizes a given float tensor using the specified tensor quantization configuration.

        Args:
            float_tensor (torch.Tensor): The float tensor to be quantized.
            tensor_quant_config (QTensorConfig, optional): The tensor quantization configuration. Defaults to None.

        Returns:
            torch.Tensor: The quantized tensor.
        """
        q_weight, q_tensor_meta = cls._quantize(
            tensor=float_tensor,
            tensor_quant_config=tensor_quant_config,
        )
        q_weight = cls._create_q_tensor(q_weight, q_tensor_meta)
        return q_weight

    @classmethod
    def dequantize(cls, q_weight: "QTensor") -> torch.Tensor:
        """Dequantizes the QTensor into a float tensor.

        Args:
            q_weight (QTensor): The quantized weight tensor.

        Returns:
            torch.Tensor: The dequantized float tensor.
        """
        meta = q_weight.meta_info.to_dict()
        meta["zero"] = q_weight.zero
        meta["scale"] = q_weight.scale
        return cls._dequantize(q_weight.val, meta)

    @classmethod
    def _create_q_tensor(cls, weight, meta) -> "QTensor":
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
    def _quantize(cls, tensor, tensor_quant_config: QTensorConfig = None):
        nbits = tensor_quant_config.nbits
        channel_wise = tensor_quant_config.channel_wise
        group_size = tensor_quant_config.group_size if tensor_quant_config.group_size != -1 else None
        optimize = tensor_quant_config.optimize
        round_zero = tensor_quant_config.round_zero
        axis = 0  # *Note did not exposed to the user
        bitpack = tensor_quant_config.pack

        assert nbits in cls.SUPPORTED_BITS, "nbits=" + str(nbits) + " not supported."
        assert axis in [0, 1], "axis should be either 0 or 1, but got {}".format(axis)
        if group_size is not None:
            assert tensor.numel() % group_size == 0, (
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

        # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero,
        # the scale is inverted later on.
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
    def _dequantize(cls, W_q, meta):
        # Main dequantization: bit_unpacking > (W_q - z)*s > reshape
        if meta["packing"]:
            W_r = Packer.get_unpack_fn(meta["nbits"])(W_q)
            if hqq_global_option.use_half:
                W_r = W_r.half()
            if (meta["group_size"] is not None) and (meta["nbits"] == 3):
                W_r = W_r[: meta["group_size"]] if (meta["axis"] == 0) else W_r[:, : meta["group_size"]]
        else:
            if hqq_global_option.use_half:
                W_r = W_q.half()
        # TODO: double check the correctness, the official impl is also error...
        W_r = ((W_r - meta["zero"]) * meta["scale"]).reshape(meta["shape"])
        return W_r


class HQQLinear(torch.nn.Linear):
    """HQQ Linear module."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        q_weight: QTensor = None,
        device=None,
        dtype=None,
    ) -> None:
        """Init a HQQ linear."""
        super().__init__(in_features, out_features, bias, device, dtype)
        self.q_weight = q_weight
        self.quantized = q_weight is not None

    @dump_elapsed_time("Quantize linear module into HQQ module.")
    def quantize_weight(
        self,
        W: torch.Tensor,
        quant_config: HQQModuleConfig = default_hqq_module_config,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantizes the weight using HQQ.

        Args:
            W (torch.Tensor): The weight tensor to be quantized.
            quant_config (HQQModuleConfig, optional): The quantization configuration.
                Defaults to default_hqq_module_config.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: A tuple containing the quantized weight tensor
                and a dictionary of additional information.
        """
        weight_quant_config, scale_quant_config, zero_quant_config = (
            quant_config.weight,
            quant_config.scale,
            quant_config.zero,
        )
        need_quant_scale = scale_quant_config is not None
        need_quant_zero = zero_quant_config is not None

        self.in_features, self.out_features = W.t().shape

        # Quantize weight
        q_weight = HQQTensorHandle.quantize(float_tensor=W, tensor_quant_config=weight_quant_config)
        self.q_weight = q_weight

        # * The dequantization process only happens in the first forward pass.
        # * It will change the `q_weight` but faster.
        # * we should not save the state after doing the forward.
        if need_quant_scale:  # Quantize scale
            q_scale_tensor = HQQTensorHandle.quantize(
                float_tensor=self.q_weight.scale, tensor_quant_config=scale_quant_config
            )
            self.q_weight.scale = q_scale_tensor
        if need_quant_zero:  # Quantize zero
            q_zero_tensor = HQQTensorHandle.quantize(
                float_tensor=self.q_weight.zero,
                tensor_quant_config=zero_quant_config,
            )
            self.q_weight.zero = q_zero_tensor
        self.quantized = True

    def dequantize_weight(self):
        """Dequantize the weight tensor."""
        assert self.quantized, "model was not quantized"
        # TODO: move below logic into `HQQTensorHandle`
        if self.q_weight.is_scale_quantized():
            scale_qdq = HQQTensorHandle.dequantize(self.q_weight.scale)
            self.q_weight.scale = scale_qdq

        if self.q_weight.is_zero_quantized():
            zero_qdq = HQQTensorHandle.dequantize(self.q_weight.zero)
            self.q_weight.zero = zero_qdq

        W_qdq = HQQTensorHandle.dequantize(self.q_weight)
        return W_qdq

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the HQQ linear module."""
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
        """Create a new HQQModule instance from a floating-point linear.

        Args:
            float_module (torch.nn.Linear): The floating-point module to convert.
            quant_config (HQQModuleConfig, optional): The quantization configuration.
                Defaults to default_hqq_module_config.

        Returns:
            HQQModule: The converted HQQModule instance.
        """
        # Create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features,
            fake_out_features,
            bias=float_module.bias is not None,
        )
        new_mod.requires_grad_ = False
        # Construct the q weight from float weight
        new_mod.quantize_weight(float_module.weight, quant_config=quant_config)
        # Update the linear module attributes
        new_mod.in_features = float_module.in_features
        new_mod.out_features = float_module.out_features
        new_mod.weight = None
        new_mod.bias = float_module.bias
        if hqq_global_option.use_half and new_mod.bias is not None:
            new_mod.bias = torch.nn.Parameter(float_module.bias.half())
        # TODO: refine it to support cuda/hpu/cpu
        device_to_use = next(float_module.parameters()).device
        if hqq_global_option.use_half:
            new_mod.q_weight = new_mod.q_weight.half()
        new_mod.to(device_to_use)
        new_mod.q_weight.to(device_to_use)
        # !!! Delete the float explicitly to save memory
        del float_module
        return new_mod

    def state_dict(self, *args, **kwargs):  # nn.Module override compatible
        """Returns a dictionary containing the state of the module.

        The state dictionary contains the weights of the `q_weight` attribute.
        If the `bias` attribute is not None, it is also included in the state dictionary.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: A dictionary containing the state of the module.
        """
        state_dict = self.q_weight.to_state_dict()
        if self.bias is not None:
            state_dict["bias"] = self.bias
        if "destination" in kwargs and "prefix" in kwargs:
            for key, value in state_dict.items():
                kwargs["destination"][kwargs["prefix"] + key] = value
        return state_dict

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        all_expected_keys = ["val", "scale_quantized", "zero_quantized", "meta_info"]
        if self.bias is not None:
            all_expected_keys.append("bias")

        for key in all_expected_keys:
            if prefix + key not in state_dict:
                missing_keys.append(key)
        if missing_keys:
            return  # Can't load weights if either weight or meta is missing

        cur_state_dict = {}
        for key in all_expected_keys:
            cur_state_dict[key] = state_dict.pop(prefix + key)

        unexpected_keys += state_dict.keys()
        self._assign_state_dict(cur_state_dict, strict)

    def _assign_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        _scale_quantized = state_dict["scale_quantized"]
        _zero_quantized = state_dict["zero_quantized"]
        scale_state = state_dict["meta_info"]["scale"]
        zero_state = state_dict["meta_info"]["zero"]
        if _scale_quantized:
            scale = HQQTensorHandle._create_q_tensor(scale_state["val"], scale_state["meta_info"])
        else:
            scale = state_dict["meta_info"]["scale"]
        if _zero_quantized:
            zero = HQQTensorHandle._create_q_tensor(zero_state["val"], zero_state["meta_info"])
        else:
            zero = state_dict["meta_info"]["zero"]
        meta = state_dict["meta_info"]
        meta["scale"] = scale
        meta["zero"] = zero
        self.q_weight = HQQTensorHandle._create_q_tensor(state_dict["val"], meta)
        if self.bias is not None:
            self.bias = state_dict["bias"]
        self.quantized = True
        return self
