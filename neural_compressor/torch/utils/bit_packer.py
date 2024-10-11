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
"""Utility functions for bit packing."""


from typing import Callable, Dict, Tuple

import numba
import numpy as np

#  key: (bits, compress_bits), value: pack function
bit_packers: Dict[Tuple[int, int], Callable] = {}


def register_pack_func(orig_bits: int, compress_bits: int):
    """Register the pack function."""

    def decorator(func):
        bit_packers[(orig_bits, compress_bits)] = func
        return func

    return decorator


@register_pack_func(4, 32)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b4_c32(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=4 and compress_bits=32."""
    for i in range(new_in_features):
        packed_array[:, i] = (
            ((raw_array[:, i * n_pack + 7] & 0b1111) << 28)
            | ((raw_array[:, i * n_pack + 6] & 0b1111) << 24)
            | ((raw_array[:, i * n_pack + 5] & 0b1111) << 20)
            | ((raw_array[:, i * n_pack + 4] & 0b1111) << 16)
            | ((raw_array[:, i * n_pack + 3] & 0b1111) << 12)
            | ((raw_array[:, i * n_pack + 2] & 0b1111) << 8)
            | ((raw_array[:, i * n_pack + 1] & 0b1111) << 4)
            | (raw_array[:, i * n_pack] & 0b1111)
        )
    return packed_array


@register_pack_func(4, 16)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b4_c16(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=4 and compress_bits=16."""
    for i in range(new_in_features):
        packed_array[:, i] = (
            ((raw_array[:, i * n_pack + 3] & 0b1111) << 12)
            | ((raw_array[:, i * n_pack + 2] & 0b1111) << 8)
            | ((raw_array[:, i * n_pack + 1] & 0b1111) << 4)
            | (raw_array[:, i * n_pack] & 0b1111)
        )
    return packed_array


@register_pack_func(4, 8)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b4_c8(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=4 and compress_bits=8."""
    for i in range(new_in_features):
        packed_array[:, i] = ((raw_array[:, i * n_pack + 1] & 0b1111) << 4) | (raw_array[:, i * n_pack] & 0b1111)
    return packed_array


@register_pack_func(4, 64)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b4_c64(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=4 and compress_bits=64."""
    for i in range(new_in_features):
        packed_array[:, i] = (
            ((raw_array[:, i * n_pack + 15] & 0b1111) << 60)
            | ((raw_array[:, i * n_pack + 14] & 0b1111) << 56)
            | ((raw_array[:, i * n_pack + 13] & 0b1111) << 52)
            | ((raw_array[:, i * n_pack + 12] & 0b1111) << 48)
            | ((raw_array[:, i * n_pack + 11] & 0b1111) << 44)
            | ((raw_array[:, i * n_pack + 10] & 0b1111) << 40)
            | ((raw_array[:, i * n_pack + 9] & 0b1111) << 36)
            | ((raw_array[:, i * n_pack + 8] & 0b1111) << 32)
            | ((raw_array[:, i * n_pack + 7] & 0b1111) << 28)
            | ((raw_array[:, i * n_pack + 6] & 0b1111) << 24)
            | ((raw_array[:, i * n_pack + 5] & 0b1111) << 20)
            | ((raw_array[:, i * n_pack + 4] & 0b1111) << 16)
            | ((raw_array[:, i * n_pack + 3] & 0b1111) << 12)
            | ((raw_array[:, i * n_pack + 2] & 0b1111) << 8)
            | ((raw_array[:, i * n_pack + 1] & 0b1111) << 4)
            | (raw_array[:, i * n_pack] & 0b1111)
        )
    return packed_array


@register_pack_func(8, 32)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b8_c32(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=8 and compress_bits=32."""
    for i in range(new_in_features):
        packed_array[:, i] = (
            ((raw_array[:, i * n_pack + 3] & 0b11111111) << 24)
            | ((raw_array[:, i * n_pack + 2] & 0b11111111) << 16)
            | ((raw_array[:, i * n_pack + 1] & 0b11111111) << 8)
            | (raw_array[:, i * n_pack] & 0b11111111)
        )
    return packed_array


@register_pack_func(8, 16)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b8_c16(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=8 and compress_bits=16."""
    for i in range(new_in_features):
        packed_array[:, i] = (
            ((raw_array[:, i * n_pack + 3] & 0b11111111) << 24)
            | ((raw_array[:, i * n_pack + 2] & 0b11111111) << 16)
            | ((raw_array[:, i * n_pack + 1] & 0b11111111) << 8)
            | (raw_array[:, i * n_pack] & 0b11111111)
        )
    return packed_array


@register_pack_func(8, 8)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b8_c8(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=8 and compress_bits=8."""
    for i in range(new_in_features):
        packed_array[:, i] = raw_array[:, i * n_pack] & 0b11111111
    return packed_array


@register_pack_func(8, 64)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b8_c64(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=8 and compress_bits=64."""
    for i in range(new_in_features):
        packed_array[:, i] = (
            ((raw_array[:, i * n_pack + 7] & 0b11111111) << 56)
            | ((raw_array[:, i * n_pack + 6] & 0b11111111) << 48)
            | ((raw_array[:, i * n_pack + 5] & 0b11111111) << 40)
            | ((raw_array[:, i * n_pack + 4] & 0b11111111) << 32)
            | ((raw_array[:, i * n_pack + 3] & 0b11111111) << 24)
            | ((raw_array[:, i * n_pack + 2] & 0b11111111) << 16)
            | ((raw_array[:, i * n_pack + 1] & 0b11111111) << 8)
            | (raw_array[:, i * n_pack] & 0b11111111)
        )
    return packed_array


@register_pack_func(2, 32)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b2_c32(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=2 and compress_bits=32."""
    for i in range(new_in_features):
        packed_array[:, i] = (
            ((raw_array[:, i * n_pack + 15] & 0b11) << 30)
            | ((raw_array[:, i * n_pack + 14] & 0b11) << 28)
            | ((raw_array[:, i * n_pack + 13] & 0b11) << 26)
            | ((raw_array[:, i * n_pack + 12] & 0b11) << 24)
            | ((raw_array[:, i * n_pack + 11] & 0b11) << 22)
            | ((raw_array[:, i * n_pack + 10] & 0b11) << 20)
            | ((raw_array[:, i * n_pack + 9] & 0b11) << 18)
            | ((raw_array[:, i * n_pack + 8] & 0b11) << 16)
            | ((raw_array[:, i * n_pack + 7] & 0b11) << 14)
            | ((raw_array[:, i * n_pack + 6] & 0b11) << 12)
            | ((raw_array[:, i * n_pack + 5] & 0b11) << 10)
            | ((raw_array[:, i * n_pack + 4] & 0b11) << 8)
            | ((raw_array[:, i * n_pack + 3] & 0b11) << 6)
            | ((raw_array[:, i * n_pack + 2] & 0b11) << 4)
            | ((raw_array[:, i * n_pack + 1] & 0b11) << 2)
            | (raw_array[:, i * n_pack] & 0b11)
        )
    return packed_array


@register_pack_func(2, 16)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b2_c16(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=2 and compress_bits=16."""
    for i in range(new_in_features):
        packed_array[:, i] = (
            ((raw_array[:, i * n_pack + 7] & 0b11) << 14)
            | ((raw_array[:, i * n_pack + 6] & 0b11) << 12)
            | ((raw_array[:, i * n_pack + 5] & 0b11) << 10)
            | ((raw_array[:, i * n_pack + 4] & 0b11) << 8)
            | ((raw_array[:, i * n_pack + 3] & 0b11) << 6)
            | ((raw_array[:, i * n_pack + 2] & 0b11) << 4)
            | ((raw_array[:, i * n_pack + 1] & 0b11) << 2)
            | (raw_array[:, i * n_pack] & 0b11)
        )
    return packed_array


@register_pack_func(2, 8)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b2_c8(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=2 and compress_bits=8."""
    for i in range(new_in_features):
        packed_array[:, i] = (
            ((raw_array[:, i * n_pack + 3] & 0b11) << 6)
            | ((raw_array[:, i * n_pack + 2] & 0b11) << 4)
            | ((raw_array[:, i * n_pack + 1] & 0b11) << 2)
            | (raw_array[:, i * n_pack] & 0b11)
        )
    return packed_array


@register_pack_func(2, 64)
@numba.jit(nopython=True, parallel=True)
def pack_array_with_numba_b2_c64(
    raw_array: np.ndarray, packed_array: np.ndarray, n_pack: int, new_in_features: int
) -> np.ndarray:
    """Pack the array with numba when bits=2 and compress_bits=64."""
    for i in range(new_in_features):
        packed_array[:, i] = (
            ((raw_array[:, i * n_pack + 31] & 0b11) << 62)
            | ((raw_array[:, i * n_pack + 30] & 0b11) << 60)
            | ((raw_array[:, i * n_pack + 29] & 0b11) << 58)
            | ((raw_array[:, i * n_pack + 28] & 0b11) << 56)
            | ((raw_array[:, i * n_pack + 27] & 0b11) << 54)
            | ((raw_array[:, i * n_pack + 26] & 0b11) << 52)
            | ((raw_array[:, i * n_pack + 25] & 0b11) << 50)
            | ((raw_array[:, i * n_pack + 24] & 0b11) << 48)
            | ((raw_array[:, i * n_pack + 23] & 0b11) << 46)
            | ((raw_array[:, i * n_pack + 22] & 0b11) << 44)
            | ((raw_array[:, i * n_pack + 21] & 0b11) << 42)
            | ((raw_array[:, i * n_pack + 20] & 0b11) << 40)
            | ((raw_array[:, i * n_pack + 19] & 0b11) << 38)
            | ((raw_array[:, i * n_pack + 18] & 0b11) << 36)
            | ((raw_array[:, i * n_pack + 17] & 0b11) << 34)
            | ((raw_array[:, i * n_pack + 16] & 0b11) << 32)
            | ((raw_array[:, i * n_pack + 15] & 0b11) << 30)
            | ((raw_array[:, i * n_pack + 14] & 0b11) << 28)
            | ((raw_array[:, i * n_pack + 13] & 0b11) << 26)
            | ((raw_array[:, i * n_pack + 12] & 0b11) << 24)
            | ((raw_array[:, i * n_pack + 11] & 0b11) << 22)
            | ((raw_array[:, i * n_pack + 10] & 0b11) << 20)
            | ((raw_array[:, i * n_pack + 9] & 0b11) << 18)
            | ((raw_array[:, i * n_pack + 8] & 0b11) << 16)
            | ((raw_array[:, i * n_pack + 7] & 0b11) << 14)
            | ((raw_array[:, i * n_pack + 6] & 0b11) << 12)
            | ((raw_array[:, i * n_pack + 5] & 0b11) << 10)
            | ((raw_array[:, i * n_pack + 4] & 0b11) << 8)
            | ((raw_array[:, i * n_pack + 3] & 0b11) << 6)
            | ((raw_array[:, i * n_pack + 2] & 0b11) << 4)
            | ((raw_array[:, i * n_pack + 1] & 0b11) << 2)
            | (raw_array[:, i * n_pack] & 0b11)
        )
    return packed_array
