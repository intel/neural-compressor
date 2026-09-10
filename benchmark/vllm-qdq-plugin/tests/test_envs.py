import os
import unittest
from unittest import mock

from vllm_qdq_plugin import envs


class EnvTests(unittest.TestCase):
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
