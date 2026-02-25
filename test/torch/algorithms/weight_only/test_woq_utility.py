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


def test_convert_dtype_str2torch():
    from neural_compressor.torch.algorithms.weight_only.utility import convert_dtype_str2torch

    # Test for supported dtypes
    assert convert_dtype_str2torch("int8") == torch.int8
    assert convert_dtype_str2torch("fp32") == torch.float
    assert convert_dtype_str2torch("float32") == torch.float
    assert convert_dtype_str2torch("auto") == torch.float
    assert convert_dtype_str2torch("fp16") == torch.float16
    assert convert_dtype_str2torch("float16") == torch.float16
    assert convert_dtype_str2torch("bf16") == torch.bfloat16
    assert convert_dtype_str2torch("bfloat16") == torch.bfloat16

    # Test for unsupported dtypes
    with pytest.raises(AssertionError):
        convert_dtype_str2torch("int16")
