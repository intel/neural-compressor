import pytest
import torch
import torch.nn.functional as F
from nvfp4_hw.fused_moe_ue5m3 import fused_moe_ue5m3
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, MoEActivation
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import nvfp4_e5m3_qdq


def _decode_weight(packed: torch.Tensor, scale_bits: torch.Tensor, group_size: int) -> torch.Tensor:
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device)
    low = packed & 0x0F
    high = packed >> 4
    low_value = values[(low & 0x07).long()] * torch.where(low & 0x08 != 0, -1.0, 1.0)
    high_value = values[(high & 0x07).long()] * torch.where(high & 0x08 != 0, -1.0, 1.0)
    unpacked = torch.stack((low_value, high_value), dim=-1).flatten(-2)
    scales = torch.ops.vllm.nvfp4_ue5m3_to_fp32(scale_bits).repeat_interleave(group_size, dim=-1)
    return unpacked * scales


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


@pytest.mark.parametrize("group_size", [16, 32])
def test_fused_moe_matches_reference(group_size: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(1)
    device = torch.device("cuda")
    num_experts, num_tokens, top_k = 4, 5, 2
    hidden_size = intermediate_size = 64
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    w13 = torch.randint(
        0, 256, (num_experts, intermediate_size * 2, hidden_size // 2), device=device, dtype=torch.uint8
    )
    w2 = torch.randint(0, 256, (num_experts, hidden_size, intermediate_size // 2), device=device, dtype=torch.uint8)
    w13_scale = torch.ops.vllm.nvfp4_fp32_to_ue5m3(
        torch.rand(num_experts, intermediate_size * 2, hidden_size // group_size, device=device)
    )
    w2_scale = torch.ops.vllm.nvfp4_fp32_to_ue5m3(
        torch.rand(num_experts, hidden_size, intermediate_size // group_size, device=device)
    )
    topk_ids = torch.tensor([[0, 1], [2, 3], [1, 3], [0, 2], [3, 1]], device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)

    actual = fused_moe_ue5m3(
        x,
        w13,
        w2,
        w13_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        ApplyMoEActivationConfig(),
        group_size,
    )
    expected = _reference(x, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids, group_size)
    assert F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) > 0.9999
    relative_l2_error = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
        expected.float()
    )
    assert relative_l2_error < 0.01


def test_fused_moe_supports_dynamo_fullgraph_capture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = torch.device("cuda")
    x = torch.randn(2, 64, device=device, dtype=torch.bfloat16)
    w13 = torch.randint(0, 256, (2, 128, 32), device=device, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (2, 64, 32), device=device, dtype=torch.uint8)
    w13_scale = torch.ops.vllm.nvfp4_fp32_to_ue5m3(torch.rand(2, 128, 4, device=device))
    w2_scale = torch.ops.vllm.nvfp4_fp32_to_ue5m3(torch.rand(2, 64, 4, device=device))
    topk_ids = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.int32)
    topk_weights = torch.full((2, 2), 0.5, device=device)
    activation_config = ApplyMoEActivationConfig()

    def apply(value: torch.Tensor) -> torch.Tensor:
        return fused_moe_ue5m3(
            value,
            w13,
            w2,
            w13_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            MoEActivation.SILU,
            activation_config,
            16,
        )

    expected = apply(x)
    compiled = torch.compile(apply, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)


def test_fused_moe_supports_cuda_graph_capture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = torch.device("cuda")
    x = torch.randn(2, 64, device=device, dtype=torch.bfloat16)
    w13 = torch.randint(0, 256, (2, 128, 32), device=device, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (2, 64, 32), device=device, dtype=torch.uint8)
    w13_scale = torch.ops.vllm.nvfp4_fp32_to_ue5m3(torch.rand(2, 128, 4, device=device))
    w2_scale = torch.ops.vllm.nvfp4_fp32_to_ue5m3(torch.rand(2, 64, 4, device=device))
    topk_ids = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.int32)
    topk_weights = torch.full((2, 2), 0.5, device=device)

    def apply() -> torch.Tensor:
        return fused_moe_ue5m3(
            x,
            w13,
            w2,
            w13_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            MoEActivation.SILU,
            ApplyMoEActivationConfig(),
            16,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            apply()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = apply()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(captured_output).all()
