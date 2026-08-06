"""Compare reference and CuTe QDQ accuracy and latency on SM80+ GPUs."""

import argparse
import os
import sys
from collections.abc import Callable

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from vllm_qdq_plugin.qdq.cute import cute_qdq_status
from vllm_qdq_plugin.qdq.mxfp4 import _mxfp4_qdq_reference, mxfp4_qdq
from vllm_qdq_plugin.qdq.mxfp8 import _mxfp8_qdq_reference, mxfp8_qdq


def benchmark(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", nargs=2, type=int, default=(1024, 4096), metavar=("M", "K"))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    x = torch.randn(*args.shape, device="cuda", dtype=dtype)
    available, reason = cute_qdq_status(x)
    print(f"device: {torch.cuda.get_device_name()} | CuTe status: {reason}")
    if not available:
        raise RuntimeError(reason)

    previous_flag = os.environ.get("VLLM_QDQ_CUTE")
    os.environ["VLLM_QDQ_CUTE"] = "1"
    try:
        for name, reference, cute in (
            ("MXFP4", _mxfp4_qdq_reference, mxfp4_qdq),
            ("MXFP8", _mxfp8_qdq_reference, mxfp8_qdq),
        ):
            reference_out = reference(x)
            cute_out = cute(x)
            max_abs_error = (reference_out.float() - cute_out.float()).abs().max().item()
            equal = torch.equal(reference_out, cute_out)
            reference_ms = benchmark(lambda: reference(x), args.warmup, args.iterations)
            cute_ms = benchmark(lambda: cute(x), args.warmup, args.iterations)
            print(
                f"{name}: exact={equal} max_abs_error={max_abs_error:.6g} "
                f"reference={reference_ms:.3f} ms cute={cute_ms:.3f} ms speedup={reference_ms / cute_ms:.2f}x"
            )
    finally:
        if previous_flag is None:
            os.environ.pop("VLLM_QDQ_CUTE", None)
        else:
            os.environ["VLLM_QDQ_CUTE"] = previous_flag


if __name__ == "__main__":
    main()