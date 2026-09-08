import os
import unittest
from unittest import mock

import torch
from vllm_qdq_plugin.qdq.mxfp4 import _mxfp4_qdq_reference, mxfp4_qdq
from vllm_qdq_plugin.qdq.mxfp8 import _mxfp8_qdq_reference, mxfp8_qdq
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import _nvfp4_e5m3_qdq_reference, nvfp4_e5m3_qdq


class CuteQDQTests(unittest.TestCase):
    def test_missing_cutlass_dsl_warns_in_auto_mode_on_supported_gpu(self) -> None:
        x = mock.Mock(is_cuda=True, device=torch.device("cuda", 0))
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            torch.version, "cuda", "12.8"
        ), mock.patch("torch.cuda.get_device_capability", return_value=(8, 0)), mock.patch(
            "importlib.util.find_spec", return_value=None
        ):
            from vllm_qdq_plugin.qdq.cute import cute_qdq_status, warn_reference_fallback

            available, reason = cute_qdq_status(x)
            self.assertFalse(available)
            with self.assertWarnsRegex(RuntimeWarning, "pip install.*nvidia-cutlass-dsl"):
                warn_reference_fallback("TEST_MXFP4", reason)

    def test_missing_cutlass_dsl_errors_when_explicitly_requested(self) -> None:
        x = mock.Mock(is_cuda=True, device=torch.device("cuda", 0))
        with mock.patch.dict(os.environ, {"VLLM_QDQ_CUTE": "1"}, clear=True), mock.patch.object(
            torch.version, "cuda", "12.8"
        ), mock.patch("torch.cuda.get_device_capability", return_value=(8, 0)), mock.patch(
            "importlib.util.find_spec", return_value=None
        ):
            from vllm_qdq_plugin.qdq.cute import cute_qdq_status

            with self.assertRaisesRegex(RuntimeError, "pip install.*nvidia-cutlass-dsl"):
                cute_qdq_status(x)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cute_ops_support_dynamo_fullgraph_capture(self) -> None:
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("CuTe QDQ requires SM80 or newer")

        with mock.patch.dict(os.environ, {"VLLM_QDQ_CUTE": "1"}, clear=False):
            for reference, qdq in (
                (_mxfp4_qdq_reference, mxfp4_qdq),
                (_mxfp8_qdq_reference, mxfp8_qdq),
            ):
                x = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
                qdq(x)
                compiled = torch.compile(qdq, backend="eager", fullgraph=True)
                self.assertTrue(torch.equal(compiled(x), reference(x)))

    def test_capture_cache_miss_does_not_attempt_jit(self) -> None:
        from vllm_qdq_plugin.qdq.cute_kernels import _COMPILED_KERNELS, _get_compiled_kernel

        with mock.patch.dict(_COMPILED_KERNELS, {}, clear=True), mock.patch(
            "torch.cuda.is_current_stream_capturing", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "Run one eager QDQ warmup"):
                _get_compiled_kernel("MXFP4", torch.bfloat16, 123456, torch.device("cuda", 0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cute_kernel_compilation_is_reused_across_shapes(self) -> None:
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("CuTe QDQ requires SM80 or newer")

        from vllm_qdq_plugin.qdq.cute_kernels import _COMPILED_KERNELS

        with mock.patch.dict(os.environ, {"VLLM_QDQ_CUTE": "1"}, clear=False), mock.patch.dict(
            _COMPILED_KERNELS, {}, clear=True
        ):
            first = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
            second = torch.randn(200, 2048, device="cuda", dtype=torch.bfloat16)
            self.assertTrue(torch.equal(mxfp4_qdq(first), _mxfp4_qdq_reference(first)))
            self.assertTrue(torch.equal(mxfp4_qdq(second), _mxfp4_qdq_reference(second)))
            self.assertEqual(len(_COMPILED_KERNELS), 1)

    def test_cpu_input_uses_exact_reference_fallback(self) -> None:
        x = torch.randn(3, 37, dtype=torch.bfloat16)
        with mock.patch.dict(os.environ, {"VLLM_QDQ_CUTE": "1"}, clear=False):
            with self.assertWarnsRegex(RuntimeWarning, "input is not a CUDA tensor"):
                actual_mxfp4 = mxfp4_qdq(x)
            actual_mxfp8 = mxfp8_qdq(x)

        self.assertTrue(torch.equal(actual_mxfp4, _mxfp4_qdq_reference(x)))
        self.assertTrue(torch.equal(actual_mxfp8, _mxfp8_qdq_reference(x)))

    def test_non_contiguous_input_is_materialized_for_cute(self) -> None:
        from vllm_qdq_plugin.qdq import cute

        x = torch.randn(32, 3, dtype=torch.bfloat16).t()
        self.assertFalse(x.is_contiguous())
        with mock.patch.object(cute, "cute_qdq_status", return_value=(True, "CuTe DSL is available")), mock.patch.object(
            cute, "_mxfp8_qdq_cute_op", side_effect=lambda value: value
        ) as cute_op, self.assertWarnsRegex(RuntimeWarning, r"shape=.*stride=.*Materializing a contiguous copy"):
            actual = cute.mxfp8_qdq_cute(x)

        cute_input = cute_op.call_args.args[0]
        self.assertTrue(cute_input.is_contiguous())
        self.assertTrue(torch.equal(actual, x))

    def test_nvfp4_defaults_to_auto_reference_fallback(self) -> None:
        x = torch.randn(3, 32, dtype=torch.bfloat16)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertWarnsRegex(RuntimeWarning, "NVFP4_E5M3.*input is not a CUDA tensor.*performance"):
                actual = nvfp4_e5m3_qdq(x, 16)

        self.assertTrue(torch.equal(actual, _nvfp4_e5m3_qdq_reference(x, 16)))

    def test_nvfp4_explicit_disable_uses_reference_without_warning(self) -> None:
        x = torch.randn(3, 32, dtype=torch.bfloat16)
        with mock.patch.dict(os.environ, {"VLLM_QDQ_CUTE": "0"}, clear=False), mock.patch("warnings.warn") as warn:
            actual = nvfp4_e5m3_qdq(x, 16)

        warn.assert_not_called()
        self.assertTrue(torch.equal(actual, _nvfp4_e5m3_qdq_reference(x, 16)))

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
