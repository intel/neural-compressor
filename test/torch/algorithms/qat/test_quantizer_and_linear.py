import math
import types

import pytest
import torch
import torch.nn as nn

# Skip the whole module if auto_round (needed for get_quant_func inside TensorQuantizer) is not available
auto_round = pytest.importorskip("auto_round")

from neural_compressor.torch.algorithms.qat.quant_linear import QuantLinear
from neural_compressor.torch.algorithms.qat.tensor_quantizer import TensorQuantizer


def make_quant_cfg(
    *,
    data_type="mx_fp8",
    bits=8,
    group_size=32,
    sym=True,
    act_data_type="mx_fp8",
    act_bits=8,
    act_group_size=32,
    act_sym=True,
):
    """Build a lightweight namespace mimicking the attributes QuantLinear._setup expects."""
    return types.SimpleNamespace(
        data_type=data_type,
        bits=bits,
        group_size=group_size,
        sym=sym,
        act_data_type=act_data_type,
        act_bits=act_bits,
        act_group_size=act_group_size,
        act_sym=act_sym,
    )


def build_quant_linear(in_features=32, out_features=16, bias=True, quant_cfg=None, device="cpu", dtype=torch.float32):
    """Manually construct a QuantLinear since the class does not define an __init__.

    Steps:
      1. Instantiate the module
      2. Register parameter tensors (weight, bias)
      3. Add metadata attributes used by extra_repr / repr
      4. Call internal _setup with provided quant config
    """
    if quant_cfg is None:
        quant_cfg = make_quant_cfg(group_size=32, act_group_size=32)

    ql = QuantLinear()
    ql.in_features = in_features
    ql.out_features = out_features

    weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
    ql.register_parameter("weight", nn.Parameter(weight))

    if bias:
        b = torch.randn(out_features, device=device, dtype=dtype)
        ql.register_parameter("bias", nn.Parameter(b))
    else:
        ql.bias = None  # make sure attribute exists

    ql._setup(quant_cfg)
    return ql


@pytest.mark.parametrize("bias", [True, False])
def test_quant_linear_forward_and_backward(bias):
    torch.manual_seed(42)

    in_features = 32
    out_features = 16
    batch_size = 3

    ql = build_quant_linear(in_features=in_features, out_features=out_features, bias=bias)

    # Create a deliberately non-contiguous input (transpose trick)
    base = torch.randn(in_features, batch_size)
    x = base.t()  # shape (batch_size, in_features) but non-contiguous
    assert not x.is_contiguous()

    x.requires_grad_(True)
    out = ql(x)

    # Shape & dtype checks
    assert out.shape == (batch_size, out_features)
    assert out.dtype == x.dtype

    # Backward pass
    loss = out.sum()
    loss.backward()

    assert ql.weight.grad is not None, "Weight should receive gradient through fake quant path"
    if bias:
        assert ql.bias.grad is not None, "Bias should receive gradient"
    else:
        assert ql.bias is None

    # Ensure original weight dtype tracked
    assert ql.original_weight_dtype == ql.weight.dtype

    # Output quantizer is explicitly disabled in _setup
    assert "TensorQuantizer(disabled)" in repr(ql.output_quantizer)

    # Input/weight quantizers should be enabled (not containing 'disabled')
    assert "disabled" not in repr(ql.input_quantizer)
    assert "disabled" not in repr(ql.weight_quantizer)


def test_quant_linear_repr_and_extra_repr():
    ql = build_quant_linear(in_features=8, out_features=4, bias=True)
    r = repr(ql)
    # Basic structural checks
    assert "QuantLinear(" in r
    assert "(input_quantizer):" in r
    assert "(weight_quantizer):" in r
    assert "(output_quantizer):" in r
    # extra_repr path
    er = ql.extra_repr()
    assert "in_features=8" in er
    assert "out_features=4" in er
    assert "bias=True" in er


def test_tensor_quantizer_disable_and_no_quant_path():
    tq = TensorQuantizer(if_quant=False)  # constructed with quantization turned off
    x = torch.randn(5, 7)
    out = tq(x)
    # When disabled (not quant) it should return the identical object (same memory)
    assert out.data_ptr() == x.data_ptr()
    assert repr(tq) == "TensorQuantizer(disabled)"


def test_tensor_quantizer_enable_disable_cycle():
    tq = TensorQuantizer()
    x = torch.randn(4, 32)  # group size default 32, matches last dim
    y1 = tq(x)
    assert y1.shape == x.shape
    # Disable and ensure passthrough (pointer equality)
    tq.disable()
    y2 = tq(x)
    assert y2.data_ptr() == x.data_ptr()
    assert "disabled" in repr(tq)
    # Re-enable
    tq.enable()
    y3 = tq(x)
    assert y3.shape == x.shape
    assert "disabled" not in repr(tq)


def test_tensor_quantizer_scale_persistence():
    # Provide scale_shape so internal buffer is registered & updated
    tq = TensorQuantizer(scale_shape=(4, 32), block_size=32)
    x = torch.randn(4, 32)
    # Use internal fake quant function to generate scale
    q, shared_exp = tq._fake_quantize(x)
    # scale buffer should have been updated (shape (4, 1))
    assert hasattr(tq, "scale")
    assert tq.scale.shape == (4, 1)
    # We cannot be certain of values, but at least ensure it is uint8 and not all zeros (likely)
    assert tq.scale.dtype == torch.uint8
    # Heuristic: at least one non-zero (if all zero it may still be valid, but improbable)
    assert (tq.scale != 0).any() or (shared_exp == 0).all()


def test_weight_pack():
    # Provide scale_shape so internal buffer is registered & updated
    tq = TensorQuantizer(scale_shape=(4, 32), block_size=32)
    x = torch.randn(4, 32)
    # Use internal fake quant function to generate scale
    q, shared_exp = tq._fake_quantize(x)

    q_packed, scale = tq.weight_pack(q, shared_exp)

    assert q_packed.dtype == torch.float8_e4m3fn

    tq = TensorQuantizer(data_type="mx_fp4", bits=4, scale_shape=(4, 32), block_size=32)
    x = torch.randn(4, 32)
    # Use internal fake quant function to generate scale
    q, shared_exp = tq._fake_quantize(x)

    q_packed, scale = tq.weight_pack(q, shared_exp)

    assert q_packed.dtype == torch.uint8
