import pytest
import torch

from neural_compressor.torch.algorithms.mx_quant import utils


def test_mx_quant_utility():
    tensor = torch.rand((1, 30))
    assert torch.equal(tensor, utils.quantize_mx_op(tensor, None, "nearest", 32))
    assert torch.equal(tensor, utils._quantize_fp(tensor))
    assert torch.equal(tensor, utils._quantize_bfloat(tensor, 0))
    assert torch.equal(tensor, utils._quantize_mx(tensor, 8, None))

    assert not torch.equal(utils._shared_exponents(tensor, "none"), utils._shared_exponents(tensor))
    with pytest.raises(Exception):
        utils._shared_exponents(tensor, None)
    with pytest.raises(Exception):
        utils._reshape_to_blocks(tensor, None, 32)
    with pytest.raises(Exception):
        utils.quantize_elemwise_op(tensor, "test")
    with pytest.raises(Exception):
        utils._round_mantissa(tensor, 3, "test")
