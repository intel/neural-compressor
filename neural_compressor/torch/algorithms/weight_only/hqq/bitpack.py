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

# Notice: Copied from from https://github.com/mobiusml/hqq
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
"""Bit packing logic for HQQ."""


import numpy as np
import torch

__all__ = ["Packer"]


# Bit packing logic. format: pack/unpack_nBits_target-<uint8 or int32>
class BitPack:
    """Packing and unpacking tensors into different bit representations."""

    # 8-bit
    ################################################
    @staticmethod
    def pack_8bit_u8(W_q):
        """Packs the given tensor into 8-bit unsigned integers.

        Args:
            W_q (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The packed tensor.
        """
        return W_q.to(torch.uint8)

    @staticmethod
    def unpack_8bit_u8(W_q):
        """Unpacks the given 8-bit tensor into 8-bit unsigned integer tensor.

        Args:
            W_q (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The unpacked tensor.
        """
        return W_q

    # 4-bit
    ################################################
    @staticmethod
    def pack_4bit_u8(W_q):  # uint8 > uint8/2
        """Packs the given 4-bit tensor into 8-bit unsigned integers.

        Args:
            W_q (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The packed tensor.
        """
        W_q = W_q.to(torch.uint8)
        _step = int(len(W_q) / 2)
        return (W_q[:_step] << 4) | W_q[_step:]

    # A bit faster than the _cat version
    @staticmethod
    def unpack_4bit_u8(W_q):  # uint8/2 > uint8
        """Unpacks the given 4-bit tensor into 8-bit unsigned integers.

        Args:
            W_q (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The unpacked tensor.
        """
        _step = W_q.shape[0]
        tmp = torch.empty([2 * _step, W_q.shape[1]], dtype=torch.uint8, device=W_q.device)
        tmp[:_step] = (W_q & 0b11110000) >> 4
        tmp[_step:] = W_q & 0b00001111
        return tmp

    # 2-bit
    ################################################
    @staticmethod
    def pack_2bit_u8(W_q):  # uint8 > uint8/4
        """Packs the given 2-bit tensor into 8-bit unsigned integers.

        Args:
            W_q (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The packed tensor.
        """
        W_q = W_q.to(torch.uint8)
        _step = int(len(W_q) / 4)
        return W_q[:_step] << 6 | W_q[_step : 2 * _step] << 4 | W_q[2 * _step : 3 * _step] << 2 | W_q[3 * _step :]

    # A bit faster than the _cat version
    @staticmethod
    def unpack_2bit_u8(W_q):
        """Unpacks the tensor packed by `pack_2bit_u8`.

        Args:
            W_q (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The unpacked tensor.
        """
        _step = W_q.shape[0]
        tmp = torch.empty([4 * _step, W_q.shape[1]], dtype=torch.uint8, device=W_q.device)
        tmp[:_step] = (W_q & 0b11000000) >> 6
        tmp[_step : 2 * _step] = (W_q & 0b00110000) >> 4
        tmp[2 * _step : 3 * _step] = (W_q & 0b00001100) >> 2
        tmp[3 * _step :] = W_q & 0b00000011
        return tmp

    # 3bit
    ################################################
    @staticmethod
    def pack_3bit_32(W_q_in):
        """Packs the given 3-bit tensor into 32-bit signed integers.

        Args:
            W_q_in (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The packed tensor.
        """
        W_q = torch.zeros(
            [int(10 * np.ceil(W_q_in.shape[0] / 10.0)), W_q_in.shape[1]], device=W_q_in.device, dtype=torch.int32
        )
        W_q[: len(W_q_in)] = W_q_in
        _step = int(len(W_q) / 10)
        W_q = (
            (W_q[:_step] << 27)
            | (W_q[_step : _step * 2] << 24)
            | (W_q[_step * 2 : _step * 3] << 21)
            | (W_q[_step * 3 : _step * 4] << 18)
            | (W_q[_step * 4 : _step * 5] << 15)
            | (W_q[_step * 5 : _step * 6] << 12)
            | (W_q[_step * 6 : _step * 7] << 9)
            | (W_q[7 * _step : _step * 8] << 6)
            | (W_q[_step * 8 : _step * 9] << 3)
            | (W_q[_step * 9 :])
        )
        return W_q

    # A bit faster than _cat version
    @staticmethod
    def unpack_3bit_32(W_q):
        """Unpacks the tensor packed by `pack_3bit_32`.

        Args:
            W_q (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The unpacked tensor.
        """
        _step = W_q.shape[0]
        tmp = torch.empty([10 * _step, W_q.shape[1]], dtype=torch.uint8, device=W_q.device)
        tmp[:_step] = (W_q & 0b00111000000000000000000000000000) >> 27
        tmp[1 * _step : 2 * _step] = (W_q & 0b00000111000000000000000000000000) >> 24
        tmp[2 * _step : 3 * _step] = (W_q & 0b00000000111000000000000000000000) >> 21
        tmp[3 * _step : 4 * _step] = (W_q & 0b00000000000111000000000000000000) >> 18
        tmp[4 * _step : 5 * _step] = (W_q & 0b00000000000000111000000000000000) >> 15
        tmp[5 * _step : 6 * _step] = (W_q & 0b00000000000000000111000000000000) >> 12
        tmp[6 * _step : 7 * _step] = (W_q & 0b00000000000000000000111000000000) >> 9
        tmp[7 * _step : 8 * _step] = (W_q & 0b00000000000000000000000111000000) >> 6
        tmp[8 * _step : 9 * _step] = (W_q & 0b00000000000000000000000000111000) >> 3
        tmp[9 * _step :] = W_q & 0b00000000000000000000000000000111
        return tmp


class Packer:
    """Pack/unpack functions collection."""

    bit_to_packing = {8: "8bit_u8", 4: "4bit_u8", 3: "3bit_32", 2: "2bit_u8"}

    pack_fn_mapping = {
        "8bit_u8": BitPack.pack_8bit_u8,
        "4bit_u8": BitPack.pack_4bit_u8,
        "3bit_32": BitPack.pack_3bit_32,
        "2bit_u8": BitPack.pack_2bit_u8,
    }

    unpack_fn_mapping = {
        "8bit_u8": BitPack.unpack_8bit_u8,
        "4bit_u8": BitPack.unpack_4bit_u8,
        "3bit_32": BitPack.unpack_3bit_32,
        "2bit_u8": BitPack.unpack_2bit_u8,
    }

    @staticmethod
    def get_pack_fn(nbits: int):
        """Get the pack function for the specified number of bits.

        Args:
            nbits (int): The number of bits.

        Returns:
            function: The pack function for the specified number of bits.
        """
        return Packer.pack_fn_mapping[Packer.bit_to_packing[nbits]]

    @staticmethod
    def get_unpack_fn(nbits: int):
        """Get the unpack function for the specified number of bits.

        Args:
            nbits (int): The number of bits.

        Returns:
            function: The unpack function for the specified number of bits.
        """
        return Packer.unpack_fn_mapping[Packer.bit_to_packing[nbits]]
