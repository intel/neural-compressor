import os
import unittest
from unittest import mock

import torch
from vllm_qdq_plugin.qdq.mxfp4 import _mxfp4_qdq_reference, mxfp4_qdq
from vllm_qdq_plugin.qdq.mxfp8 import _mxfp8_qdq_reference, mxfp8_qdq


class CuteQDQTests(unittest.TestCase):
    def test_capture_cache_miss_does_not_attempt_jit(self) -> None:
        from vllm_qdq_plugin.qdq.cute_kernels import _get_compiled_kernel

        with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Run one eager QDQ warmup"):
                _get_compiled_kernel("MXFP4", torch.bfloat16, 123456, torch.device("cuda", 0))

    def test_cpu_input_uses_exact_reference_fallback(self) -> None:
        x = torch.randn(3, 37, dtype=torch.bfloat16)
        with mock.patch.dict(os.environ, {"VLLM_QDQ_CUTE": "1"}, clear=False):
            with self.assertWarnsRegex(RuntimeWarning, "input is not a CUDA tensor"):
                actual_mxfp4 = mxfp4_qdq(x)
            actual_mxfp8 = mxfp8_qdq(x)

        self.assertTrue(torch.equal(actual_mxfp4, _mxfp4_qdq_reference(x)))
        self.assertTrue(torch.equal(actual_mxfp8, _mxfp8_qdq_reference(x)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cute_kernels_match_reference(self) -> None:
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("CuTe QDQ requires SM80 or newer")

        with mock.patch.dict(os.environ, {"VLLM_QDQ_CUTE": "1"}, clear=False):
            for dtype in (torch.float16, torch.bfloat16):
                x = torch.randn(17, 96, device="cuda", dtype=dtype)
                self.assertTrue(torch.equal(mxfp4_qdq(x), _mxfp4_qdq_reference(x)))
                self.assertTrue(torch.equal(mxfp8_qdq(x), _mxfp8_qdq_reference(x)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cute_kernels_support_cuda_graph_replay(self) -> None:
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("CuTe QDQ requires SM80 or newer")

        with mock.patch.dict(os.environ, {"VLLM_QDQ_CUTE": "1"}, clear=False):
            for reference, qdq in (
                (_mxfp4_qdq_reference, mxfp4_qdq),
                (_mxfp8_qdq_reference, mxfp8_qdq),
            ):
                static_x = torch.randn(17, 96, device="cuda", dtype=torch.bfloat16)
                qdq(static_x)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = qdq(static_x)

                for _ in range(2):
                    static_x.copy_(torch.randn_like(static_x))
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(static_output, reference(static_x)))
