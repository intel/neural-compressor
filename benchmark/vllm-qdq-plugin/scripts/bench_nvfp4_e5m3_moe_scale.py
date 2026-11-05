"""Benchmark NVFP4_E5M3_GS MoE activation QDQ scale strategies."""

import argparse
import os
import sys
import time
from collections.abc import Callable

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from vllm_qdq_plugin.qdq.cute import cute_qdq_status
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import _nvfp4_e5m3_qdq_reference, nvfp4_e5m3_qdq
from vllm_qdq_plugin.qdq.nvfp4_e5m3_cute import nvfp4_e5m3_gs_qdq_cute


def measure(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def benchmark_case(
    *,
    experts: int,
    tokens: int,
    top_k: int,
    hidden_size: int,
    warmup: int,
    iterations: int,
    shared_scale: bool,
) -> None:
    device = torch.device("cuda")
    rows = tokens * top_k
    x = torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
    expert_ids = torch.arange(rows, device=device) % experts
    scales = torch.full((experts,), 2.0, device=device, dtype=torch.float32)
    if not shared_scale:
        scales = 3.0e4 + (torch.arange(experts, device=device, dtype=torch.float32) % 17) * 1024.0
    row_scales = scales[expert_ids]
    reference = _nvfp4_e5m3_qdq_reference(x, 16, row_scales)

    strategies: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("batched-reference", lambda: nvfp4_e5m3_qdq(x, 16, input_global_scale=row_scales)),
    ]
    if not shared_scale:
        from vllm_qdq_plugin.qdq.nvfp4_e5m3_cute import nvfp4_e5m3_gs_qdq_per_row_cute

        expert_rows = [torch.where(expert_ids == expert)[0] for expert in range(experts)]

        strategies.append(("batched-row-cute", lambda: nvfp4_e5m3_gs_qdq_per_row_cute(x, row_scales)))

        def expert_loop_cute() -> torch.Tensor:
            output = torch.empty_like(x)
            for expert, indices in enumerate(expert_rows):
                if indices.numel():
                    output.index_copy_(
                        0,
                        indices,
                        nvfp4_e5m3_gs_qdq_cute(x.index_select(0, indices).contiguous(), scales[expert]),
                    )
            return output

        strategies.append(("expert-loop-cute", expert_loop_cute))
    else:
        strategies.append(
            ("shared-scalar-cute", lambda: nvfp4_e5m3_gs_qdq_cute(x, scales[0]))
        )

    print(
        f"case: experts={experts}, tokens={tokens}, top_k={top_k}, rows={rows}, "
        f"hidden={hidden_size}, scales={'shared' if shared_scale else 'per-expert'}"
    )
    results: dict[str, float] = {}
    for name, fn in strategies:
        actual = fn()
        torch.cuda.synchronize()
        exact = torch.equal(actual, reference)
        max_error = (actual.float() - reference.float()).abs().max().item()
        ms = measure(fn, warmup, iterations)
        results[name] = ms
        print(f"  {name}: {ms:.3f} ms, exact={exact}, max_abs_error={max_error:.6g}")
    baseline = results["batched-reference"]
    for name, ms in results.items():
        if name != "batched-reference":
            print(f"  speedup vs batched-reference ({name}): {baseline / ms:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--tokens", nargs="+", type=int, default=[16, 128])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.experts <= 0 or args.top_k <= 0 or args.hidden_size % 16:
        raise ValueError("experts/top-k must be positive and hidden-size divisible by 16")
    probe = torch.empty((1, args.hidden_size), device="cuda", dtype=torch.bfloat16)
    available, reason = cute_qdq_status(probe)
    print(f"device: {torch.cuda.get_device_name()} | CuTe status: {reason}")
    if not available:
        raise RuntimeError(reason)

    for token_count in args.tokens:
        for shared in (False, True):
            benchmark_case(
                experts=args.experts,
                tokens=token_count,
                top_k=args.top_k,
                hidden_size=args.hidden_size,
                warmup=args.warmup,
                iterations=args.iterations,
                shared_scale=shared,
            )


if __name__ == "__main__":
    main()
