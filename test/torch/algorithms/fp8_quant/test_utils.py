import re
import unittest
from typing import List

from neural_compressor.torch.algorithms.fp8_quant._core.utils import is_re_match


class TestUtils(unittest.TestCase):
    def test_is_re_match_found(self):
        substr_list = ["lm_head", "mlp\\.gate\\b"]
        target = "layer.1.mlp.gate"
        self.assertTrue(is_re_match(substr_list, target))
        target2 = "model.lm_head"
        self.assertTrue(is_re_match(substr_list, target2))

    def test_is_re_match_not_found(self):
        substr_list = ["lm_head", "mlp\\.gate\\b"]
        target = "layer.1.mlp.gate_up_proj"
        self.assertFalse(is_re_match(substr_list, target))
