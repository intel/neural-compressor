import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, MoEActivation
from vllm_qdq_plugin.qdq.nvfp4 import nvfp4_qdq
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import nvfp4_e5m3_qdq
from vllm_qdq_plugin.quantization.inc_nvfp4_e5m3_moe import INCNvfp4UE5M3MoEMethod
from vllm_qdq_plugin.quantization.inc_nvfp4_moe import INCNvfp4QDQMoEMethod


def _decode_weight(packed: torch.Tensor, scale_bits: torch.Tensor, group_size: int) -> torch.Tensor:
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device)
    low = packed & 0x0F
    high = packed >> 4
    low_value = values[(low & 0x07).long()] * torch.where(low & 0x08 != 0, -1.0, 1.0)
    high_value = values[(high & 0x07).long()] * torch.where(high & 0x08 != 0, -1.0, 1.0)
    unpacked = torch.stack((low_value, high_value), dim=-1).flatten(-2)
    scales = _decode_ue5m3(scale_bits).repeat_interleave(group_size, dim=-1)
    return unpacked * scales


def _decode_standard_weight(packed: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device)
    low = packed & 0x0F
    high = packed >> 4
    low_value = values[(low & 0x07).long()] * torch.where(low & 0x08 != 0, -1.0, 1.0)
    high_value = values[(high & 0x07).long()] * torch.where(high & 0x08 != 0, -1.0, 1.0)
    unpacked = torch.stack((low_value, high_value), dim=-1).flatten(-2)
    return unpacked * scales.float().repeat_interleave(group_size, dim=-1)


def _decode_ue5m3(scale_bits: torch.Tensor) -> torch.Tensor:
    exponent = scale_bits.to(torch.int32) >> 3
    mantissa = scale_bits.to(torch.int32) & 0x7
    subnormal = mantissa.float() * 2.0**-17
    normal = torch.ldexp(1.0 + mantissa.float() * 0.125, exponent - 15)
    return torch.where(exponent == 0, subnormal, normal)


def _reference(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    output = torch.zeros_like(x)
    x = nvfp4_e5m3_qdq(x, group_size)
    decoded_w13 = _decode_weight(w13, w13_scale, group_size).to(x.dtype)
    decoded_w2 = _decode_weight(w2, w2_scale, group_size).to(x.dtype)
    for expert_id in range(w13.shape[0]):
        token_ids, slots = torch.where(topk_ids == expert_id)
        gate_up = x[token_ids] @ decoded_w13[expert_id].T
        intermediate_size = gate_up.shape[-1] // 2
        activated = F.silu(gate_up[:, :intermediate_size]) * gate_up[:, intermediate_size:]
        activated = nvfp4_e5m3_qdq(activated, group_size)
        expert_output = activated @ decoded_w2[expert_id].T
        routed_output = (expert_output * topk_weights[token_ids, slots, None]).to(output.dtype)
        output.index_add_(0, token_ids, routed_output)
    return output


def test_marlin_moe_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(1)
    device = torch.device("cuda")
    num_experts, num_tokens, top_k = 4, 5, 2
    hidden_size, intermediate_size = 128, 64
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    w13 = torch.randint(
        0, 256, (num_experts, intermediate_size * 2, hidden_size // 2), device=device, dtype=torch.uint8
    )
    w2 = torch.randint(0, 256, (num_experts, hidden_size, intermediate_size // 2), device=device, dtype=torch.uint8)
    w13_scale = torch.randint(
        0x40, 0x79, (num_experts, intermediate_size * 2, hidden_size // 16), device=device, dtype=torch.uint8
    )
    w2_scale = torch.randint(
        0x40, 0x79, (num_experts, hidden_size, intermediate_size // 16), device=device, dtype=torch.uint8
    )
    topk_ids = torch.tensor([[0, 1], [2, 3], [1, 3], [0, 2], [3, 1]], device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
    expected = _reference(x, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids, 16)

    layer = torch.nn.Module()
    layer.num_experts = num_experts
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = intermediate_size
    layer.params_dtype = torch.bfloat16
    layer.w13_weight_packed = torch.nn.Parameter(w13, requires_grad=False)
    layer.w2_weight_packed = torch.nn.Parameter(w2, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.global_num_experts = num_experts
    layer.expert_map = None

    method = object.__new__(INCNvfp4UE5M3MoEMethod)
    method.group_size = 16
    method.moe = type(
        "MoeConfig",
        (),
        {
            "is_act_and_mul": True,
            "swiglu_limit": None,
            "swiglu_alpha": None,
            "swiglu_beta": None,
            "activation_situ_beta": None,
            "activation_situ_linear_beta": None,
        },
    )()
    method.process_weights_after_loading(layer)
    actual = method.apply(layer, x, topk_weights, topk_ids, None, None)

    assert F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) > 0.9999
    relative_l2_error = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
        expected.float()
    )
    assert relative_l2_error < 0.01


@pytest.mark.parametrize("uniform_scales", [False, True])
def test_standard_nvfp4_qdq_moe_matches_reference(uniform_scales: bool):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(2)
    device = torch.device("cuda")
    num_experts, num_tokens, top_k = 4, 5, 2
    hidden_size, intermediate_size = 128, 64
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    w13 = torch.randint(
        0, 256, (num_experts, intermediate_size * 2, hidden_size // 2), device=device, dtype=torch.uint8
    )
    w2 = torch.randint(0, 256, (num_experts, hidden_size, intermediate_size // 2), device=device, dtype=torch.uint8)
    w13_scale = torch.rand(num_experts, intermediate_size * 2, hidden_size // 16, device=device).to(torch.float8_e4m3fn)
    w2_scale = torch.rand(num_experts, hidden_size, intermediate_size // 16, device=device).to(torch.float8_e4m3fn)
    w13_global_scale = torch.full((num_experts, 2), 2.0, device=device)
    w2_global_scale = torch.full((num_experts,), 2.0, device=device)
    input_global_scale = torch.tensor([0.25, 0.125, 0.5, 0.0625], device=device)[:, None].repeat(1, 2)
    w2_input_global_scale = torch.tensor([0.5, 0.25, 0.125, 1.0], device=device)
    if uniform_scales:
        input_global_scale = torch.full_like(input_global_scale, 0.25)
        w2_input_global_scale = torch.full_like(w2_input_global_scale, 0.5)
    topk_ids = torch.tensor([[0, 1], [2, 3], [1, 3], [0, 2], [3, 1]], device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)

    decoded_w13 = _decode_standard_weight(w13, w13_scale, 16).to(x.dtype) / 2.0
    decoded_w2 = _decode_standard_weight(w2, w2_scale, 16).to(x.dtype) / 2.0
    expected = torch.zeros_like(x)
    for expert_id in range(num_experts):
        token_ids, slots = torch.where(topk_ids == expert_id)
        quantized_x = nvfp4_qdq(x[token_ids], input_global_scale[expert_id, 0], 16)
        gate_up = quantized_x @ decoded_w13[expert_id].T
        activated = F.silu(gate_up[:, :intermediate_size]) * gate_up[:, intermediate_size:]
        activated = nvfp4_qdq(activated, w2_input_global_scale[expert_id], 16)
        expert_output = activated @ decoded_w2[expert_id].T
        expected.index_add_(0, token_ids, (expert_output * topk_weights[token_ids, slots, None]).to(expected.dtype))

    layer = torch.nn.Module()
    layer.num_experts = num_experts
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = intermediate_size
    layer.params_dtype = torch.bfloat16
    layer.w13_weight_packed = torch.nn.Parameter(w13, requires_grad=False)
    layer.w2_weight_packed = torch.nn.Parameter(w2, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
    layer.w13_weight_global_scale = torch.nn.Parameter(w13_global_scale, requires_grad=False)
    layer.w2_weight_global_scale = torch.nn.Parameter(w2_global_scale, requires_grad=False)
    layer.w13_input_global_scale = torch.nn.Parameter(input_global_scale, requires_grad=False)
    layer.w2_input_global_scale = torch.nn.Parameter(w2_input_global_scale, requires_grad=False)
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.global_num_experts = num_experts
    layer.expert_map = None

    method = object.__new__(INCNvfp4QDQMoEMethod)
    method.group_size = 16
    method.moe = type(
        "MoeConfig",
        (),
        {
            "is_act_and_mul": True,
            "w13_num_shards": 2,
            "swiglu_limit": None,
            "swiglu_alpha": None,
            "swiglu_beta": None,
            "activation_situ_beta": None,
            "activation_situ_linear_beta": None,
        },
    )()
    method.process_weights_after_loading(layer)
    if uniform_scales:
        assert layer.w13_input_scale.ndim == 0 and layer.w2_input_scale.ndim == 0
    else:
        torch.testing.assert_close(layer.w13_input_scale, input_global_scale[:, 0])
        torch.testing.assert_close(layer.w2_input_scale, w2_input_global_scale)
    actual = method.apply(layer, x, topk_weights, topk_ids, None, None)

    assert F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) > 0.9999
    relative_l2_error = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
        expected.float()
    )
    assert relative_l2_error < 0.01


def test_marlin_moe_with_global_scale_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(3)
    device = torch.device("cuda")
    num_experts, num_tokens, top_k = 4, 5, 2
    hidden_size, intermediate_size = 128, 64
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    w13 = torch.randint(
        0, 256, (num_experts, intermediate_size * 2, hidden_size // 2), device=device, dtype=torch.uint8
    )
    w2 = torch.randint(0, 256, (num_experts, hidden_size, intermediate_size // 2), device=device, dtype=torch.uint8)
    # High UE5M3 codes (up to 114688) exercise the Marlin range shift.
    w13_scale = torch.randint(
        0xB0, 0xFF, (num_experts, intermediate_size * 2, hidden_size // 16), device=device, dtype=torch.uint8
    )
    w2_scale = torch.randint(
        0xB0, 0xFF, (num_experts, hidden_size, intermediate_size // 16), device=device, dtype=torch.uint8
    )
    w13_global_scale = torch.tensor([4.0e5, 6.0e5, 8.0e5, 1.0e6], device=device)[:, None].repeat(1, 2)
    w2_global_scale = torch.tensor([3.0e5, 5.0e5, 7.0e5, 9.0e5], device=device)
    w13_input_global_scale = (
        torch.tensor([3.0e4, 4.0e4, 5.0e4, 6.0e4], device=device)[:, None].expand(-1, 2).contiguous()
    )
    w2_input_global_scale = torch.tensor([5.0e4, 6.0e4, 7.0e4, 8.0e4], device=device)
    topk_ids = torch.tensor([[0, 1], [2, 3], [1, 3], [0, 2], [3, 1]], device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)

    decoded_w13 = (_decode_weight(w13, w13_scale, 16) / w13_global_scale[:, :1, None]).to(x.dtype)
    decoded_w2 = (_decode_weight(w2, w2_scale, 16) / w2_global_scale[:, None, None]).to(x.dtype)
    routed_x = x[:, None, :].expand(-1, top_k, -1).reshape(-1, hidden_size)
    route_ids = topk_ids.reshape(-1)
    quantized_x = nvfp4_e5m3_qdq(routed_x, 16, input_global_scale=w13_input_global_scale[route_ids, 0])
    expected = torch.zeros_like(x)
    for expert_id in range(num_experts):
        token_ids, slots = torch.where(topk_ids == expert_id)
        route_rows = token_ids * top_k + slots
        gate_up = quantized_x[route_rows] @ decoded_w13[expert_id].T
        activated = F.silu(gate_up[:, :intermediate_size]) * gate_up[:, intermediate_size:]
        activated = nvfp4_e5m3_qdq(
            activated,
            16,
            input_global_scale=w2_input_global_scale[expert_id].expand(token_ids.numel()),
        )
        expert_output = activated @ decoded_w2[expert_id].T
        expected.index_add_(0, token_ids, (expert_output * topk_weights[token_ids, slots, None]).to(expected.dtype))

    layer = torch.nn.Module()
    layer.num_experts = num_experts
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = intermediate_size
    layer.params_dtype = torch.bfloat16
    layer.w13_weight_packed = torch.nn.Parameter(w13, requires_grad=False)
    layer.w2_weight_packed = torch.nn.Parameter(w2, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
    layer.w13_weight_global_scale = torch.nn.Parameter(w13_global_scale, requires_grad=False)
    layer.w2_weight_global_scale = torch.nn.Parameter(w2_global_scale, requires_grad=False)
    layer.w13_input_global_scale = torch.nn.Parameter(w13_input_global_scale, requires_grad=False)
    layer.w2_input_global_scale = torch.nn.Parameter(w2_input_global_scale, requires_grad=False)
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.global_num_experts = num_experts
    layer.expert_map = None

    method = object.__new__(INCNvfp4UE5M3MoEMethod)
    method.group_size = 16
    method.with_global_scale = True
    method.moe = type(
        "MoeConfig",
        (),
        {
            "is_act_and_mul": True,
            "w13_num_shards": 2,
            "swiglu_limit": None,
            "swiglu_alpha": None,
            "swiglu_beta": None,
            "activation_situ_beta": None,
            "activation_situ_linear_beta": None,
        },
    )()
    method.process_weights_after_loading(layer)
    actual = method.apply(layer, x, topk_weights, topk_ids, None, None)

    assert F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) > 0.9999
    relative_l2_error = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
        expected.float()
    )
    assert relative_l2_error < 0.01


@pytest.mark.parametrize(
    "scale_format, group_size, per_row",
    [("UE5M3", 16, False), ("UE5M3", 32, True), ("E4M3", 16, True), ("E4M3", 16, False)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_silu_and_mul_qdq_matches_separate_ops(scale_format, group_size, per_row, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation
    from vllm_qdq_plugin.qdq.cute import cute_qdq_status
    from vllm_qdq_plugin.qdq.nvfp4_e5m3_cute import try_fused_silu_and_mul_qdq

    torch.manual_seed(13)
    rows, width = 37, 256
    activation_input = (
        torch.randn(rows, 2 * width, device="cuda") * torch.logspace(-3, 1.5, 2 * width, device="cuda")
    ).to(dtype)
    if not cute_qdq_status(activation_input)[0]:
        pytest.skip("CuTe DSL is not available")
    if per_row:
        scale = torch.rand(rows, device="cuda") * (1.0e4 if scale_format == "UE5M3" else 100.0)
    else:
        scale = None if scale_format == "UE5M3" else torch.tensor([50.0], device="cuda")
    expected = torch.empty(rows, width, device="cuda", dtype=dtype)
    apply_moe_activation(MoEActivation.SILU, expected, activation_input)
    if scale_format == "UE5M3":
        nvfp4_e5m3_qdq(expected, group_size, input_global_scale=scale, inplace=True)
    else:
        nvfp4_qdq(expected, scale, group_size, inplace=True)

    actual = torch.empty_like(expected)
    fused = try_fused_silu_and_mul_qdq(
        MoEActivation.SILU,
        actual,
        activation_input,
        ApplyMoEActivationConfig(),
        group_size,
        scale_format,
        scale,
        format_name="TEST",
        trace_op_name="test",
    )

    assert fused
    assert torch.equal(actual, expected)
    clamped = ApplyMoEActivationConfig(clamp_limit=7.0)
    assert not try_fused_silu_and_mul_qdq(
        MoEActivation.SILU,
        actual,
        activation_input,
        clamped,
        group_size,
        scale_format,
        scale,
        format_name="TEST",
        trace_op_name="test",
    )


def test_e5m3_moe_global_scale_keeps_shared_scale_fast_paths():
    prepare_scale = INCNvfp4UE5M3MoEMethod._prepare_input_global_scale

    assert prepare_scale(torch.ones(4, 2), "w13 input global scale") is None
    shared = prepare_scale(torch.full((4,), 5.0), "w2 input global scale")
    assert shared is not None and shared.ndim == 0 and shared.item() == 5.0
    per_expert = prepare_scale(torch.tensor([2.0, 3.0, 2.0, 4.0]), "w2 input global scale")
    torch.testing.assert_close(per_expert, torch.tensor([2.0, 3.0, 2.0, 4.0]))
