import copy

import pytest
import torch

from neural_compressor.torch.algorithms.weight_only.modules import INCWeightOnlyLinear
from neural_compressor.torch.algorithms.weight_only.utility import quant_tensor


class TestWeightOnlyLinear:
    @pytest.mark.parametrize(
        "bits, compression_dtype",
        [
            (8, torch.int8),
            (8, torch.int16),
            (8, torch.int32),
            (8, torch.int64),
            (4, torch.int8),
            (4, torch.int16),
            (4, torch.int32),
            (4, torch.int64),
            (2, torch.int8),
            (2, torch.int16),
            (2, torch.int32),
            (2, torch.int64),
        ],
    )
    def test_pack_with_numba(self, bits, compression_dtype):
        m = torch.nn.Linear(64, 32)
        dtype = "int"
        weight = m.weight.detach()
        int_weight, scale, zp = quant_tensor(
            weight,
            dtype=dtype,
            bits=bits,
            return_int=True,
            group_size=32,
        )
        new_module = INCWeightOnlyLinear(
            m.in_features,
            m.out_features,
            dtype=dtype,
            bits=bits,
            group_size=32,
            zp=zp is not None,
            bias=m.bias is not None,
            use_optimum_format=False,
            compression_dtype=compression_dtype,
        )
        new_module.pack(int_weight, scale, zp, m.bias)
        unpacked_int_weight = new_module.unpack_tensor(new_module.qweight)
        assert torch.equal(unpacked_int_weight, int_weight)
