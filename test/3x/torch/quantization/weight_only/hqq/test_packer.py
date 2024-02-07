import pytest
import torch

from neural_compressor.torch.algorithms.weight_only.hqq.bitpack import Packer


@pytest.mark.parametrize("nbits", [2, 4, 8])
def test_packer(nbits):
    # TODO:　add test for 3 bits
    range_max = 2**nbits
    dims = 16
    W = torch.randint(0, range_max, (dims, dims)).to(torch.uint8)
    W_pack = Packer.get_pack_fn(nbits)(W)
    W_pack_unpack = Packer.get_unpack_fn(nbits)(W_pack)
    assert torch.allclose(W, W_pack_unpack)
    print("Packer test passed!")
