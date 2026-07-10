import os
import types
import unittest
from unittest import mock

import torch

from vllm_qdq_plugin.patch import _patch_moe_marlin_gemm


def _call_moe_marlin_gemm(fn, input_tensor: torch.Tensor, b_q_type: object):
    return fn(
        input_tensor,
        None,
        torch.empty(1),
        None,
        torch.empty(1),
        None,
        None,
        None,
        None,
        None,
        torch.empty(1),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
        torch.empty((1, 1), dtype=torch.float32),
        16,
        1,
        False,
        b_q_type,
        1,
        1,
        input_tensor.shape[-1],
        True,
        False,
        False,
        False,
    )


class PatchMoeMarlinGemmTests(unittest.TestCase):
    def test_force_mxfp4_mode_applies_qdq_case_insensitively(self) -> None:
        scalar_types = types.SimpleNamespace(
            float4_e2m1f=object(),
            float8_e4m3fn=object(),
        )
        qdq_calls: list[int] = []
        trace_calls: list[tuple[str, torch.Size, torch.dtype]] = []

        def orig(*args, **kwargs):
            return args[0]

        ops = types.SimpleNamespace(moe_wna16_marlin_gemm=orig)

        def mxfp4_qdq(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
            qdq_calls.append(group_size)
            return x + 1

        with mock.patch.dict(
            os.environ,
            {"VLLM_MARLIN_MOE_QDQ_MODE": "force_mxfp4"},
            clear=False,
        ):
            _patch_moe_marlin_gemm(
                ops,
                scalar_types,
                mxfp4_qdq,
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("unexpected MXFP8 QDQ")
                ),
                lambda op_name, shape, dtype: trace_calls.append(
                    (op_name, shape, dtype)
                ),
            )
            input_tensor = torch.zeros((2, 64), dtype=torch.float16)
            output = _call_moe_marlin_gemm(
                ops.moe_wna16_marlin_gemm,
                input_tensor,
                object(),
            )

        self.assertEqual(qdq_calls, [32])
        self.assertEqual(
            trace_calls,
            [("moe_wna16_marlin_gemm", input_tensor.shape, input_tensor.dtype)],
        )
        self.assertTrue(torch.equal(output, input_tensor + 1))

    def test_force_mxfp4_mode_skips_non_2d_inputs(self) -> None:
        scalar_types = types.SimpleNamespace(
            float4_e2m1f=object(),
            float8_e4m3fn=object(),
        )
        qdq_calls: list[int] = []

        def orig(*args, **kwargs):
            return args[0]

        ops = types.SimpleNamespace(moe_wna16_marlin_gemm=orig)

        with mock.patch.dict(
            os.environ,
            {"VLLM_MARLIN_MOE_QDQ_MODE": "FORCE_MXFP4"},
            clear=False,
        ):
            _patch_moe_marlin_gemm(
                ops,
                scalar_types,
                lambda x, group_size=32: qdq_calls.append(group_size) or (x + 1),
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("unexpected MXFP8 QDQ")
                ),
                lambda *args, **kwargs: None,
            )
            input_tensor = torch.zeros((1, 2, 64), dtype=torch.float16)
            output = _call_moe_marlin_gemm(
                ops.moe_wna16_marlin_gemm,
                input_tensor,
                object(),
            )

        self.assertEqual(qdq_calls, [])
        self.assertTrue(torch.equal(output, input_tensor))
