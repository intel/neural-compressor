import unittest
from typing import List

from neural_compressor.common.tuning_param import TuningParam


class TestTuningParam(unittest.TestCase):
    def test_is_tunable_same_type(self):
        # Test when tunable_type has the same type as the default value
        param = TuningParam("param_name", [1, 2, 3], List[int])
        self.assertTrue(param.is_tunable([4, 5, 6]))
        self.assertFalse(param.is_tunable(["not_an_int"]))

    def test_is_tunable_recursive(self):
        # Test recursive type checking for iterables
        param = TuningParam("param_name", [[1, 2], [3, 4]], List[List[int]])
        self.assertTrue(param.is_tunable([[5, 6], [7, 8]]))
        # TODO: double check if this is the expected behavior
        self.assertTrue(param.is_tunable([[5, 6], [7, "8"]]))


if __name__ == "__main__":
    unittest.main()
