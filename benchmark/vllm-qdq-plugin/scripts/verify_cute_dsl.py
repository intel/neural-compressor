"""Verify that the installed CuTe DSL can compile and execute on the active GPU."""

import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def copy_kernel(source: cute.Tensor, destination: cute.Tensor, count: cute.Int32):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 256 + thread
    if index < count:
        destination[index] = source[index]


@cute.jit
def copy_host(source: cute.Tensor, destination: cute.Tensor, count: cute.Int32):
    copy_kernel(source, destination, count).launch(
        grid=[(count + 255) // 256, 1, 1],
        block=[256, 1, 1],
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CuTe DSL verification requires CUDA")
    if torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError("CuTe QDQ requires SM80 or newer")

    source = torch.arange(1024, device="cuda", dtype=torch.float32)
    destination = torch.empty_like(source)
    cute_source = from_dlpack(source)
    cute_destination = from_dlpack(destination)
    compiled = cute.compile(copy_host, cute_source, cute_destination, source.numel())
    compiled(cute_source, cute_destination, source.numel())
    torch.cuda.synchronize()
    torch.testing.assert_close(destination, source)
    print(f"CuTe DSL SM80 smoke test passed on {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
