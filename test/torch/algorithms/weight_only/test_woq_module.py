import copy
from unittest.mock import patch

import numpy as np
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

    @pytest.mark.parametrize(
        "bits, compression_dtype",
        [
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
    @pytest.mark.parametrize("in_features", [65, 70])
    def test_pack_with_numba_unaligned_in_features(self, bits, compression_dtype, in_features):
        # When in_features is not a multiple of n_pack, the last packed column is only partially filled
        # and must be zero-padded, matching the torch implementation, without reading past the end of a row.
        pytest.importorskip("numba")
        import neural_compressor.torch.algorithms.weight_only.modules as woq_modules
        from neural_compressor.torch.utils import bit_packer

        module = INCWeightOnlyLinear(
            in_features,
            16,
            bits=bits,
            group_size=32,
            use_optimum_format=False,
            compression_dtype=compression_dtype,
        )
        np_dtype = torch.tensor(0, dtype=compression_dtype).numpy().dtype
        raw_array = np.random.default_rng(0).integers(0, 2**bits, size=(16, in_features)).astype(np_dtype)
        expected = module.pack_tensor_with_torch(torch.from_numpy(raw_array.copy())).numpy()

        # Run the pure-Python versions of the numba kernels, so out-of-bounds reads raise instead of
        # silently reading neighboring memory.
        py_packers = {key: func.py_func for key, func in bit_packer.bit_packers.items()}
        with patch.object(woq_modules, "can_pack_with_numba", lambda: True), patch.dict(
            bit_packer.bit_packers, py_packers
        ):
            packed = module.pack_array_with_numba(raw_array, module.n_pack, bits, module.compress_bits, np_dtype)
        assert np.array_equal(packed, expected)
