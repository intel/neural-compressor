import pytest
import torch


@pytest.mark.parametrize("shape", [1024, 512, 300])
def test_quant_tensor_id(shape):
    from neural_compressor.torch.algorithms.weight_only.utility import quant_tensor

    input = torch.randn(shape, shape)
    id1 = id(input)
    output = quant_tensor(input)
    id2 = id(output)
    assert id1 == id2, "quant_tensor function is an in-place operator"
