# Copyright (c) 2025 Intel Corporation
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
"""Tensor-level QDQ hot-path kernels and cold-path export packing utilities."""

from .pack import (
    PackedWeightTensor,
    dequantize_packed_weight,
    pack_nibbles,
    pack_weight,
    unpack_nibbles,
)
from .qdq import (
    QuantBackendName,
    StraightThroughFakeQuant,
    fake_quant_dynamic,
    fake_quant_fp8_static_amax,
    get_quant_backend,
    register_quant_backend,
    set_quant_backend,
    ste_fake_quant,
)

__all__ = [
    "PackedWeightTensor",
    "QuantBackendName",
    "StraightThroughFakeQuant",
    "dequantize_packed_weight",
    "fake_quant_dynamic",
    "fake_quant_fp8_static_amax",
    "get_quant_backend",
    "pack_nibbles",
    "pack_weight",
    "register_quant_backend",
    "set_quant_backend",
    "ste_fake_quant",
    "unpack_nibbles",
]
