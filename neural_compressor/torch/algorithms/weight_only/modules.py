#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch.nn.Module Class Definition."""
# Note: Do not import this file unless you have already imported torch,
# since the model classes inherit torch.nn.Module.
import math
from abc import abstractmethod

import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F

from neural_compressor.torch.utils import accelerator, can_pack_with_numba, logger

from .utility import quant_tensor


class QDQLayer(torch.nn.Module):
    """Quantized and dequantized layer."""

    def __init__(self, module, input_scale=None) -> None:
        """Init the QDQLayer object."""
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.module = module
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.input_scale = input_scale

    def forward(self, X):
        """Forward function."""
        if self.input_scale is not None:
            X = torch.mul(X, self.input_scale)
        X = self.quant(X)
        X = self.module(X)
        X = self.dequant(X)
        return X


class UnpackedWeightOnlyLinearParams(dict):
    """Contains all unpacked weight values."""

    def __init__(self, unpack_weight, scales, unpack_zp, **kwargs):
        """Create dict."""
        super().__init__(int_weight=unpack_weight, scales=scales, zp=unpack_zp, **kwargs)

    def to(self, device):
        """Change device for all values."""
        for key, value in self.items():
            if isinstance(value, torch.Tensor) and value is not None:
                self[key] = value.to(device)
        return self


class WeightOnlyLinear(torch.nn.Module):
    """Base Weight Only Linear."""

    def __init__(
        self,
        in_features,
        out_features,
        dtype,
        bits,
        group_size,
        device,
    ):
        """Initialization."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.bits = bits
        self.group_size = group_size if group_size != -1 else in_features
        self.device = device

    @abstractmethod
    def pack(self, *args, **kwargs):
        """Abstract method."""
        raise NotImplementedError("{} doesn't implement `pack` function. ".format(self.__class__.__name__))

    @abstractmethod
    def unpack(self, *args, **kwargs):
        """Abstract method."""
        raise NotImplementedError("{} doesn't implement `unpack` function. ".format(self.__class__.__name__))

    @abstractmethod
    def forward(self, input):
        """Abstract method."""
        raise NotImplementedError("{} doesn't implement `forward` function. ".format(self.__class__.__name__))

    def extra_repr(self) -> str:
        """Show message about WeighOnlyLinear."""
        tmp_str = "in_features={}, out_features={}, bits={}, group_size={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.bits,
            self.group_size,
            self.bias is not None,
        )
        return tmp_str


class INCWeightOnlyLinear(WeightOnlyLinear):
    """INC Weight Only Linear."""

    def __init__(
        self,
        in_features,
        out_features,
        dtype="int",
        bits=4,
        group_size=32,
        zp=False,
        bias=False,
        scale_dtype=torch.float32,
        compression_dtype=torch.int32,
        compression_dim=1,
        g_idx=False,
        device="cpu",
        use_optimum_format=True,
        **kwargs,
    ):
        """Init the WeightOnlyLinear object.

        Args:
            in_features (int): input features.
            out_features (int): out features.
            dtype (str, optional):  the data type of the quantized model. Defaults to "int".
            bits (int, optional): number of bits for quantization. Defaults to 4.
            group_size (int, optional): size of the quantization group. Defaults to 32.
            zp (bool, optional): zero point. Defaults to False.
            bias (bool, optional): module bias. Defaults to False.
            scale_dtype (torch.Tensor, optional): the data type of quantization scale to be used.
                                                  Defaults to torch.float32.
            compression_dtype (torch.Tensor, optional): the target dtype after comoression.
                                                        Defaults to torch.int32.
            compression_dim (int, optional): select from [0, 1], 0 is output channel, 1 is input channel.
                                             Defaults to 1.
            g_idx (bool, optional): for recording the channel order.
            device (str, optional): choose device for compression. Defaults to cpu.
            use_optimum_format (bool, optional): use the popular huggingface compression format.
                1: compression_dim: weight = 1, zeros = 0 and both are transposed.
                2: zeros -= 1 before compression.
                3: g_idx: use same number for one group instead of recording the channel order.
                4. parameter name changed, such as 'packed_weight' -> 'qweight'.
                5. zeros is always needed even for sym.
        """
        super(INCWeightOnlyLinear, self).__init__(
            in_features,
            out_features,
            dtype,
            bits,
            group_size,
            device,
        )
        self.use_optimum_format = use_optimum_format
        if "int" not in self.dtype:  # for nf4, fp4
            from neural_compressor.torch.algorithms.weight_only.utility import FLOAT_MAPPING, INT_MAPPING

            self.use_optimum_format = False  # optimum_format doesn't suit for symmetric nf4 fp4.
            float_list = FLOAT_MAPPING[self.dtype]
            int_list = INT_MAPPING[self.dtype]
            self.int2float_mapping = {}
            for k, v in zip(int_list, float_list):
                self.int2float_mapping[k] = v
        self.compression_dim = compression_dim
        assert compression_dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ], f"Only support torch.int8|16|32|64 as compressed dtype. but got {compression_dtype}"
        dtype_bits_mapping = {torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64}
        self.compress_bits = dtype_bits_mapping[compression_dtype]
        self.n_pack = self.compress_bits // self.bits
        # K is input channel, N is output channel
        assert compression_dim in [0, 1], (
            "Only support 0 or 1 as compression dimension, " + "0 is output channel, 1 is input channel."
        )
        if self.use_optimum_format:
            self.float_type = torch.float16
            self.compression_dtype = torch.int32
            self.register_buffer(
                "scales",
                torch.zeros(
                    (math.ceil(in_features / self.group_size), out_features),
                    dtype=self.float_type,
                ).to(device),
            )
            self.register_buffer(
                "qweight",
                torch.zeros(
                    (math.ceil(in_features / self.n_pack), out_features),
                    dtype=self.compression_dtype,
                ).to(device),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    (math.ceil(self.in_features / self.group_size), math.ceil(self.out_features / self.n_pack)),
                    dtype=self.compression_dtype,
                ).to(device),
            )
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=self.float_type).to(device))
        else:
            self.compression_dtype = compression_dtype
            self.float_type = scale_dtype
            self.register_buffer(
                "scales",
                torch.zeros(
                    (out_features, math.ceil(in_features / self.group_size)),
                    dtype=self.float_type,
                ).to(device),
            )
            if compression_dim == 1:
                self.register_buffer(
                    "qweight",
                    torch.zeros(
                        (out_features, math.ceil(in_features / self.n_pack)),
                        dtype=self.compression_dtype,
                    ).to(device),
                )
                if zp:
                    self.register_buffer(
                        "qzeros",
                        torch.zeros(
                            (self.out_features, math.ceil(self.in_features / self.group_size / self.n_pack)),
                            dtype=self.compression_dtype,
                        ).to(device),
                    )
            else:
                self.register_buffer(
                    "qweight",
                    torch.zeros(
                        (math.ceil(out_features / self.n_pack), in_features),
                        dtype=self.compression_dtype,
                    ).to(device),
                )
                if zp:
                    self.register_buffer(
                        "qzeros",
                        torch.zeros(
                            (math.ceil(self.out_features / self.n_pack), math.ceil(self.in_features / self.group_size)),
                            dtype=self.compression_dtype,
                        ).to(device),
                    )
            if bias:
                self.register_buffer("bias", torch.zeros(self.out_features, dtype=self.float_type).to(device))
            else:
                self.bias = None
        if g_idx:
            self.register_buffer("g_idx", torch.zeros(in_features, dtype=torch.int32).to(device))
        else:
            self.g_idx = None

    def pack(self, int_weight, scales, zp, bias=None, g_idx=None, **kwargs):
        """Pack int weight."""
        if self.use_optimum_format:
            self.scales = self.scales.T.contiguous()
            self.qweight = self.qweight.T.contiguous()
            self.qzeros = self.qzeros.T.contiguous()
        if int_weight.device.type != "meta":
            int_weight = int_weight.to(self.device)
        if self.use_optimum_format and zp is None:
            # to avoid overflow
            int_weight = int_weight.type(torch.int32)
            shift_bias = 2 ** (self.bits - 1)
            int_weight += shift_bias
            zp = torch.zeros_like(scales, dtype=torch.uint8) + shift_bias
        if bias is not None:
            assert hasattr(self, "bias"), "bias is not set when initializing."
            self.bias = bias.type(self.float_type).to(self.device)
        if g_idx is not None:
            assert hasattr(self, "g_idx"), "g_idx is not set when initializing."
            self.g_idx = g_idx.type(torch.int32).to(self.device)
            if self.use_optimum_format or "int" not in self.dtype:
                invperm = torch.argsort(self.g_idx)
                self.g_idx = invperm // self.group_size
                self.g_idx = self.g_idx.type(torch.int32).to(self.device)
        assert scales.shape == self.scales.shape, f"{scales.shape} != {self.scales.shape} Scale shape is mismatched."
        self.scales = scales.type(self.float_type).to(self.device)
        if not self.use_optimum_format and self.compression_dim == 0:
            int_weight = int_weight.T.contiguous()
            self.qweight = self.qweight.T.contiguous()
        origin_shape = int_weight.shape
        target_shape = self.qweight.shape
        assert origin_shape[0] == target_shape[0], "output channels mismatch, please check."

        # pack weight
        self.qweight.copy_(self.pack_tensor(int_weight))
        if not self.use_optimum_format and self.compression_dim == 0:
            self.qweight = self.qweight.T.contiguous()

        if zp is not None:
            zp = zp.to(self.device)
            if self.use_optimum_format:
                zp -= 1
            if self.use_optimum_format or self.compression_dim == 0:
                zp = zp.T.contiguous()
                self.qzeros = self.qzeros.T.contiguous()
            assert hasattr(self, "qzeros"), "zp is not set when initializing."
            self.qzeros.copy_(self.pack_tensor(zp))
            if self.use_optimum_format or self.compression_dim == 0:
                self.qzeros = self.qzeros.T.contiguous()
        if self.use_optimum_format:
            self.scales = self.scales.T.contiguous()
            self.qweight = self.qweight.T.contiguous()
            self.qzeros = self.qzeros.T.contiguous()

    def unpack(self):
        """Unpack weight and zero point."""
        scales = self.scales.T.contiguous() if self.use_optimum_format else self.scales
        qweight = self.qweight.T.contiguous() if self.use_optimum_format else self.qweight

        device = scales.device
        if self.g_idx is None:
            # used for recovering fp32_weight
            self.g_idx = torch.tensor([i // self.group_size for i in range(self.in_features)], dtype=torch.int32).to(
                device
            )
        # unpack weight
        if not self.use_optimum_format and self.compression_dim == 0:
            qweight = qweight.T.contiguous()
        weight = self.unpack_tensor(qweight)
        if not self.use_optimum_format and self.compression_dim == 0:
            weight = weight.T.contiguous()
        weight = weight[: self.out_features, : self.in_features]  # avoid oversize
        if "int" not in self.dtype:
            new_weight = torch.zeros(self.out_features, self.in_features).to(device)
            for k, v in self.int2float_mapping.items():
                new_weight += torch.where(weight == k, v, 0)
            weight = new_weight

        # unpack zero_point
        zp = None
        if hasattr(self, "qzeros"):
            qzeros = self.qzeros.T.contiguous() if self.use_optimum_format else self.qzeros
            if self.use_optimum_format or self.compression_dim == 0:
                qzeros = qzeros.T.contiguous()
            zp = self.unpack_tensor(qzeros)
            if self.use_optimum_format or self.compression_dim == 0:
                zp = zp.T.contiguous()
            zp = zp[: scales.shape[0], : scales.shape[1]]  # avoid oversize
            if self.use_optimum_format:
                # zp -= 1 may cause zp == -1, after recover it becomes 2**self.bits - 1
                zp += 1
                zp = torch.where(zp > (2**self.bits - 1), 0, zp)
        return UnpackedWeightOnlyLinearParams(weight, scales, zp, g_idx=self.g_idx, bias=self.bias)

    def recover(self):
        """Recover fp32 weight from packed weight."""
        logger.debug(f"Recovering {self} weight")
        unpack_params_dict = self.unpack()
        weight = unpack_params_dict.get("int_weight")
        scales = unpack_params_dict.get("scales")
        zp = unpack_params_dict.get("zp")

        device = scales.device

        fp32_weight = torch.zeros(self.out_features, self.in_features, dtype=self.float_type).to(device)

        # recover fp32 weight
        if zp is not None:
            # recover fp32 weight with int_weight, scale, and zero_point
            for idx in range(self.in_features):
                fp32_weight[:, idx] = (torch.subtract(weight[:, idx], zp[:, self.g_idx[idx]]).to(torch.int8)) * scales[
                    :, self.g_idx[idx]
                ]
        else:
            # recover fp32 weight with int_weight, scale
            for idx in range(self.in_features):
                fp32_weight[:, idx] = weight[:, idx] * scales[:, self.g_idx[idx]]

        return fp32_weight.to(scales.device)

    def pack_tensor_with_torch(self, raw_tensor):
        """Pack the tensor with torch.

        Args:
            raw_tensor (tensor): raw tensor.

        Returns:
            tensor: packed tensor.
        """
        target_len = math.ceil(raw_tensor.shape[1] / self.n_pack)
        packed_tensor = torch.zeros(raw_tensor.shape[0], target_len, dtype=self.compression_dtype).to(raw_tensor.device)
        mask = torch.tensor(2**self.bits - 1, dtype=self.compression_dtype).to(raw_tensor.device)
        for j in range(packed_tensor.shape[1]):
            start = self.n_pack * j
            end = self.n_pack * (j + 1)
            tmp = raw_tensor[:, start:end].type(self.compression_dtype)
            tmp &= mask
            for e in range(tmp.shape[1]):
                tmp[:, e] = tmp[:, e] << (self.bits * e)
                packed_tensor[:, j] |= tmp[:, e]
                accelerator.synchronize()
        return packed_tensor

    def unpack_tensor_with_torch(self, packed_tensor):
        """Unpack the tensor with torch.

        Args:
            packed_tensor (tensor): packed tensor.

        Returns:
            tensor: unpacked tensor.
        """
        target_dtype = torch.int16
        target_len = packed_tensor.shape[1] * self.n_pack
        unpacked_tensor = torch.zeros(packed_tensor.shape[0], target_len, dtype=target_dtype).to(packed_tensor.device)
        mask = torch.tensor(2**self.bits - 1, dtype=self.compression_dtype).to(packed_tensor.device)
        for j in range(packed_tensor.shape[1]):
            for e in range(self.n_pack):
                index = j * self.n_pack + e
                tmp = packed_tensor[:, j]
                tmp = tmp << (self.compress_bits - self.bits * (e + 1))
                tmp = tmp >> self.compress_bits - self.bits
                if hasattr(self, "qzeros"):
                    tmp &= mask  # remove sign bit
                unpacked_tensor[:, index].copy_(tmp.type(target_dtype))
                accelerator.synchronize()
        return unpacked_tensor

    def pack_array_with_numba(
        self, raw_array: np.ndarray, n_pack: int, bits: int, compress_bits: int, compression_dtype=np.int32
    ) -> np.ndarray:
        """Packs the input array by combining elements into a specified bit-width format using NumPy.

        Args:
            raw_array (np.ndarray): The array to be packed. Shape: [out_features, in_features] or [1, in_features].
            n_pack (int): The number of elements to be packed together.
            bits (int): The number of bits for each element.
            compress_bits (int): The number of bits for each element of the compressed array, supported 2, 4, 8.
            compression_dtype (np.dtype, optional): The data type of the compressed array. Defaults to np.int32.

        Returns:
            np.ndarray: The packed array.
        """
        # Try to pack with numba to accelerate the packing process.
        # If numba is not availabll or the packing method is not supported,
        # fallback to the torch implementation.
        if not can_pack_with_numba():
            return self.pack_tensor_with_torch(torch.from_numpy(raw_array)).cpu().numpy()
        from neural_compressor.torch.utils.bit_packer import bit_packers

        pack_func_name = (bits, compress_bits)
        if pack_func_name not in bit_packers:
            logger.warning(
                f"Unsupported packing with bits: {bits}, compress_bits: {compress_bits} using numba, fallback to torch implementation."
            )
            return self.pack_tensor_with_torch(torch.from_numpy(raw_array)).cpu().numpy()
        out_features, in_features = raw_array.shape
        new_in_features = (in_features + n_pack - 1) // n_pack
        packed_array = np.zeros((out_features, new_in_features), dtype=compression_dtype)
        raw_array = raw_array.astype(compression_dtype)
        pack_method = bit_packers[pack_func_name]
        return pack_method(raw_array, packed_array, n_pack, new_in_features)

    def pack_tensor_with_numpy_impl(self, raw_tensor):
        """The implement of packing tensor with numpy."""
        raw_array = raw_tensor.cpu().numpy()
        target_len = np.ceil(raw_array.shape[1] / self.n_pack).astype(int)
        target_dtype = torch.tensor(0, dtype=self.compression_dtype).numpy().dtype
        packed_array = np.zeros((raw_array.shape[0], target_len), dtype=target_dtype)
        mask = np.uint8(2**self.bits - 1)
        for j in range(packed_array.shape[1]):
            start = self.n_pack * j
            end = self.n_pack * (j + 1)
            tmp = raw_array[:, start:end].astype(target_dtype)
            tmp &= mask
            for e in range(tmp.shape[1]):
                tmp[:, e] = np.left_shift(tmp[:, e], self.bits * e)
                packed_array[:, j] |= tmp[:, e]
        packed_tensor = torch.from_numpy(packed_array).to(device=raw_tensor.device)
        return packed_tensor

    def pack_tensor_with_numpy(self, raw_tensor):
        """Pack the tensor with numpy."""
        if self.bits == 8 and self.compression_dtype == torch.int8:
            return raw_tensor
        if self.bits not in [2, 4, 8]:
            return self.pack_tensor_with_numpy_impl(raw_tensor)
        compression_dtype = torch.tensor(0, dtype=self.compression_dtype).numpy().dtype
        packed_array = self.pack_array_with_numba(
            raw_tensor.cpu().numpy(), self.n_pack, self.bits, self.compress_bits, compression_dtype
        )
        return torch.from_numpy(packed_array).to(device=raw_tensor.device)

    def unpack_tensor_with_numpy(self, packed_tensor):
        """Unpack the packed tensor with numpy."""
        packed_array = packed_tensor.cpu().numpy()
        target_dtype = np.int16
        if self.bits == 8 and self.compression_dtype == torch.int8 and hasattr(self, "qzeros"):
            # special case for unpacking uint8 date from int8 compression_dtype
            target_dtype = np.uint8
        target_len = packed_array.shape[1] * self.n_pack
        unpacked_array = np.zeros((packed_array.shape[0], target_len), dtype=target_dtype)
        mask = np.uint8(2**self.bits - 1)
        for j in range(packed_array.shape[1]):
            for e in range(self.n_pack):
                index = j * self.n_pack + e
                tmp = packed_array[:, j]
                tmp = np.left_shift(tmp, self.compress_bits - self.bits * (e + 1))
                tmp = np.right_shift(tmp, self.compress_bits - self.bits)
                if hasattr(self, "qzeros"):
                    tmp &= mask
                unpacked_array[:, index] = tmp.astype(target_dtype)
        unpacked_tensor = torch.from_numpy(unpacked_array).to(device=packed_tensor.device)
        return unpacked_tensor

    def pack_tensor(self, raw_tensor):
        """Pack tensor."""
        if "cuda" in raw_tensor.device.type:
            return self.pack_tensor_with_torch(raw_tensor)
        else:
            return self.pack_tensor_with_numpy(raw_tensor)

    def unpack_tensor(self, packed_tensor):
        """Unpack tensor."""
        if "cuda" in packed_tensor.device.type:
            return self.unpack_tensor_with_torch(packed_tensor)
        else:
            return self.unpack_tensor_with_numpy(packed_tensor)

    def forward(self, input):
        """Forward function."""
        if not hasattr(self, "weight"):
            weight = self.recover()
            device = self.scales.device
            if weight.dtype == torch.float16 and device.type == "cpu":
                weight = weight.float()
                self.bias = self.bias.float() if self.bias is not None else None
        if True:  # keep reusing self.weight due to recover is too slow.
            if not hasattr(self, "weight"):
                self.weight = weight
            input = input.type(self.weight.dtype)
            logger.debug(f"Calculating {self}")
            return F.linear(input, self.weight, self.bias)
        else:
            input = input.type(weight.dtype)
            return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        """Extract the configuration string.

        Returns:
            str: the configuration string.
        """
        tmp_str = "in_features={}, out_features={}, bits={}, group_size={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.bits,
            self.group_size,
            self.bias is not None,
        )
        if self.use_optimum_format:
            tmp_str += ", use_optimum_format=True"
        return tmp_str


class HPUWeightOnlyLinear(WeightOnlyLinear):
    """Weight Only Linear for HPU device."""

    def __init__(
        self,
        in_features,
        out_features,
        dtype="int",
        bits=4,
        group_size=32,
        zp=False,
        bias=False,
        scale_dtype=torch.bfloat16,
        compression_dtype=torch.int32,
        compression_dim=1,
        g_idx=False,
        device="hpu",
        use_optimum_format=True,
        **kwargs,
    ):
        """Init the WeightOnlyLinear object.

        Args:
            in_features (int): input features.
            out_features (int): out features.
            dtype (str, optional):  the data type of the quantized model. Defaults to "int".
            bits (int, optional): number of bits for quantization. Defaults to 4.
            group_size (int, optional): size of the quantization group. Defaults to 32.
            zp (bool, optional): zero point. Defaults to False.
            bias (bool, optional): module bias. Defaults to False.
            scale_dtype (torch.Tensor, optional): the data type of quantization scale to be used.
                                                  Defaults to torch.float32.
            compression_dtype (torch.Tensor, optional): the target dtype after comoression.
                                                        Defaults to torch.int32.
            compression_dim (int, optional): select from [0, 1], 0 is output channel, 1 is input channel.
                                             Defaults to 1.
            g_idx (bool, optional): for recording the channel order.
            device (str, optional): choose device for compression. Defaults to cpu.
            use_optimum_format (bool, optional): use the popular huggingface compression format.
                1: compression_dim: weight = 1, zeros = 0 and both are transposed.
                2: zeros -= 1 before compression.
                3: g_idx: use same number for one group instead of recording the channel order.
                4. parameter name changed, such as 'packed_weight' -> 'qweight'.
                5. zeros is always needed even for sym.
        """
        super(HPUWeightOnlyLinear, self).__init__(
            in_features,
            out_features,
            dtype,
            bits,
            group_size,
            device,
        )
        self.float_type = torch.bfloat16
        self.compression_dim = compression_dim
        self.compression_dtype = compression_dtype

        if bits != 4:
            raise NotImplementedError("Only 4 bits are supported.")
        self.maxq = 2**self.bits - 1

        if bias:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=self.float_type).to(self.device))
        else:
            self.bias = None

        self.register_buffer(
            "qweight",
            torch.zeros((in_features, out_features // 32 * self.bits), dtype=self.compression_dtype).to(self.device),
        )

        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(in_features / self.group_size),
                    out_features // 32 * self.bits,
                ),
                dtype=self.compression_dtype,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(in_features / self.group_size), out_features),
                dtype=self.float_type,
            ),
        )

        if g_idx:
            self.register_buffer(
                "g_idx",
                torch.tensor([i // self.group_size for i in range(in_features)], dtype=torch.int32),
            )
        else:
            self.g_idx = None

        self.half_indim = self.in_features // 2

        self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)

    def forward(self, input):
        """The forward function of HPUWeighOnlyLinear."""
        input_dtype = input.dtype
        output_shape = input.shape[:-1] + (self.out_features,)
        scales = self.scales
        qweight = self.qweight
        zeros = self.qzeros
        weight = torch.ops.hpu.convert_from_uint4(qweight, scales, zeros, input_dtype)
        output = torch.matmul(input, weight)
        output = output.to(dtype=input_dtype).reshape(
            output_shape
        )  # A cast is needed here as for some reason the vecquant2matmul_faster_old still allocate a float32 output.
        output = output + self.bias if self.bias is not None else output
        return output

    def pack(self, int_weight, scales, zp, bias=None, g_idx=None):
        """Pack weight and zero point."""
        logger.debug("Packing for HPU")

        scales = scales.T.contiguous()
        qzeros = zp.T.contiguous()
        qweight = int_weight.T.contiguous()

        self.scales = scales.to(dtype=torch.bfloat16)

        # weights and zp are on device from unpack, need to load to cpu for packing
        self.qweight = qweight.cpu()
        new_qweight = self.pack_tensor(self.qweight)
        self.qweight = new_qweight.to("hpu")

        self.qzeros = qzeros.cpu()
        new_qzeros = self.pack_tensor(self.qzeros)
        self.qzeros = new_qzeros.to("hpu")

        if bias is not None:
            self.bias = bias.to("hpu").to(torch.bfloat16)

    def unpack(self):
        """Unpack weight and zero point."""
        logger.debug("Unpacking from HPU")
        self.qweight = self.qweight.cpu()
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        weight = weight.reshape((weight.shape[0] * weight.shape[1], weight.shape[2]))
        self.qweight = self.qweight.to(self.device)

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)

        zeros = torch.bitwise_and(zeros, (2**self.bits) - 1).to(
            self.scales.dtype
        )  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.
        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
        return weight, zeros

    def pack_tensor(self, input, bits=4):
        """Pack tensor."""
        normal = input.to(torch.int32)
        q = torch.zeros((normal.shape[0], normal.shape[1] // 32 * bits), dtype=torch.int32)
        i = 0
        col = 0
        while col < q.shape[1]:
            for j in range(i, i + (32 // bits)):
                q[:, col] |= normal[:, j] << (bits * (j - i))
            i += 32 // bits
            col += 1
        q = q.to(torch.int32)
        return q


class FakeAffineTensorQuantFunction(Function):
    """Fake version of affine quantization."""

    @staticmethod
    def forward(ctx, inputs, num_bits=4, group_size=1024, scheme="asym"):
        """As it will be only applied on activation with per tensor granularity, broadcast is not needed.

        Args:
            ctx: Pytorch convention.
            inputs: A Tensor of type float32.
            min_range: A float.
            max_range: A float.
            num_bits: An integer

        Returns:
            outputs: A Tensor of type output_dtype
        """
        return quant_tensor(inputs, num_bits, group_size, scheme)

    @staticmethod
    def backward(ctx, grad_outputs):
        """Backward function.

        Args:
            ctx: Pytorch convention.
            grad_output: A tensor of gradient of outputs

        Returns:
            grad_inputs: A tensor of gradient
        """
        return grad_outputs, None, None, None


class TEQLinearFakeQuant(torch.nn.Module):
    """Wrapper quantization linear."""

    def __init__(self, orig_layer, alpha=None, num_bits=4, group_size=-1, scheme="asym"):
        """A forward hook to linear module.

        Args:
            orig_layer: the original module
            alpha: trainable alpha/scale
            num_bits: quantization level
            group_size: for fine-grained quantization.
            scheme: symmetric quantization or asymmetric quantization.
        """
        super(TEQLinearFakeQuant, self).__init__()
        self.orig_layer = orig_layer
        self.alpha = alpha

        self.num_bits = num_bits
        self.group_size = group_size
        self.scheme = scheme

    def forward(self, x):
        """Forward function."""
        alpha = torch.clip(self.alpha, 1e-5)
        shape_len = len(x.shape) - 1
        shape = (1,) * shape_len + (-1,)
        x = x / alpha.view(shape)
        weight = self.orig_layer.weight
        weight = weight * alpha.unsqueeze(dim=0)
        weight_q = FakeAffineTensorQuantFunction().apply(weight, self.num_bits, self.group_size, self.scheme)
        return F.linear(x, weight_q, self.orig_layer.bias)


class MulLinear(torch.nn.Module):
    """Linear wrapper to apply scale to input."""

    def __init__(self, module, input_scale=None):
        """A forward hook to save input max of a module.

        Args:
            module: the linear module.
            input_scale: scale for input.
        """
        super().__init__()
        if input_scale is None:
            input_scale = torch.empty(module.in_features)
        self.register_buffer("input_scale", input_scale)
        self.add_module("linear", module)

    @property
    def weight(self):
        """Property weight."""
        return self.linear.weight

    @weight.setter
    def weight(self, weight):
        """Property weight setter."""
        self.linear.weight = weight

    def forward(self, X):
        """Forward function."""
        X = torch.mul(X, self.input_scale)
        X = self.linear(X)
        return X

    def _update_linear(self):
        # update linear weight with input_scale
        scale = self.input_scale.view(1, self.input_scale.shape[0])
        with torch.no_grad():
            self.linear.weight /= scale

    def _recover_linear(self):
        # remove mul and reset sq_linear for ipex inference
        scale = self.input_scale.view(1, self.input_scale.shape[0])
        with torch.no_grad():
            self.linear.weight *= scale
