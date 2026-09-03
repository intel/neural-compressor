import os
import unittest
from unittest import mock

from vllm_qdq_plugin import envs


class EnvTests(unittest.TestCase):
    def test_weight_dequant_mode_defaults_to_once(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(envs.VLLM_NVFP4_E5M3_WEIGHT_DEQUANT_MODE, "ONCE")

    def test_weight_dequant_mode_is_case_insensitive(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VLLM_NVFP4_E5M3_WEIGHT_DEQUANT_MODE": "per_call"},
            clear=False,
        ):
            self.assertEqual(envs.VLLM_NVFP4_E5M3_WEIGHT_DEQUANT_MODE, "PER_CALL")

    def test_invalid_weight_dequant_mode_raises(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VLLM_NVFP4_E5M3_WEIGHT_DEQUANT_MODE": "bad_mode"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "Invalid value 'bad_mode'"):
                _ = envs.VLLM_NVFP4_E5M3_WEIGHT_DEQUANT_MODE

    def test_case_insensitive_choice_returns_canonical_value(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VLLM_MARLIN_MOE_QDQ_MODE": "force_mxfp4"},
            clear=False,
        ):
            self.assertEqual(envs.VLLM_MARLIN_MOE_QDQ_MODE, "FORCE_MXFP4")

    def test_invalid_choice_raises(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VLLM_MARLIN_MOE_QDQ_MODE": "bad_mode"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "Invalid value 'bad_mode'"):
                _ = envs.VLLM_MARLIN_MOE_QDQ_MODE
