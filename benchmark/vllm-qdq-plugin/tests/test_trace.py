import unittest
from unittest import mock

import torch

from vllm_qdq_plugin import trace
from vllm_qdq_plugin.qdq.mxfp8 import mxfp8_qdq
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import nvfp4_e5m3_qdq


class TraceQDQTests(unittest.TestCase):
    def setUp(self) -> None:
        trace._call_count = 0

    def test_trace_includes_backend_and_format(self) -> None:
        with mock.patch.object(trace.envs, "VLLM_QDQ_TRACE", True), mock.patch("builtins.print") as print_mock:
            trace.trace_qdq(
                "marlin_gemm",
                torch.Size((2, 32)),
                torch.bfloat16,
                backend="CuTe",
                format_name="MXFP8",
            )

        print_mock.assert_called_once_with(
            "[QDQ] backend=CuTe format=MXFP8 op=marlin_gemm shape=torch.Size([2, 32]) dtype=torch.bfloat16"
        )

    def test_trace_is_silent_when_disabled(self) -> None:
        with mock.patch.object(trace.envs, "VLLM_QDQ_TRACE", False), mock.patch("builtins.print") as print_mock:
            trace.trace_qdq(
                "nvfp4_e5m3_qdq",
                torch.Size((2, 16)),
                torch.float16,
                backend="Reference",
                format_name="NVFP4_E5M3",
            )

        print_mock.assert_not_called()

    def test_reference_route_reports_actual_backend_and_format(self) -> None:
        input_tensor = torch.zeros((2, 32), dtype=torch.bfloat16)
        with mock.patch.object(trace.envs, "VLLM_QDQ_TRACE", True), mock.patch.object(
            trace.envs, "VLLM_QDQ_CUTE", False
        ), mock.patch("builtins.print") as print_mock:
            mxfp8_qdq(input_tensor, trace_op_name="marlin_gemm")

        message = print_mock.call_args.args[0]
        self.assertIn("backend=Reference format=MXFP8 op=marlin_gemm", message)

    def test_cute_route_reports_actual_backend_and_format(self) -> None:
        from vllm_qdq_plugin.qdq import cute

        input_tensor = torch.zeros((2, 32), dtype=torch.bfloat16)
        with mock.patch.object(trace.envs, "VLLM_QDQ_TRACE", True), mock.patch.object(
            cute, "cute_qdq_status", return_value=(True, "CuTe DSL is available")
        ), mock.patch.object(cute, "_mxfp8_qdq_cute_op", side_effect=lambda value: value), mock.patch(
            "builtins.print"
        ) as print_mock:
            cute.mxfp8_qdq_cute(input_tensor, trace_op_name="marlin_gemm")

        message = print_mock.call_args.args[0]
        self.assertIn("backend=CuTe format=MXFP8 op=marlin_gemm", message)

    def test_nvfp4_reference_route_reports_format(self) -> None:
        input_tensor = torch.zeros((2, 16), dtype=torch.float16)
        with mock.patch.object(trace.envs, "VLLM_QDQ_TRACE", True), mock.patch.object(
            trace.envs, "VLLM_QDQ_CUTE", False
        ), mock.patch("builtins.print") as print_mock:
            nvfp4_e5m3_qdq(input_tensor)

        message = print_mock.call_args.args[0]
        self.assertIn("backend=Reference format=NVFP4_E5M3 op=nvfp4_e5m3_qdq", message)


if __name__ == "__main__":
    unittest.main()