import unittest

from neural_coder.utils import common

class TestCommon(unittest.TestCase):
    def test_move_element_to_front(self):
        f = common.move_element_to_front
        self.assertEqual(f([1, 2, 3, 4], 0), [1, 2, 3, 4])
        self.assertEqual(f([1, 2, 3, 4], 1), [1, 2, 3, 4])
        self.assertEqual(f([1, 2, 3, 4], 2), [2, 1, 3, 4])
        self.assertEqual(f([1, 2, 3, 4], 3), [3, 1, 2, 4])
        self.assertEqual(f([1, 2, 3, 4], 4), [4, 1, 2, 3])
        self.assertEqual(f([1, 2, 3, 4], "a"), [1, 2, 3, 4])
        self.assertEqual(f(["a", "b", "c", "d"], "d"), ["d", "a", "b", "c"])
        self.assertEqual(f(["ab", "a", "ac", "ad"], "a"), ["a", "ab", "ac", "ad"])

if __name__ == '__main__':
    unittest.main()
