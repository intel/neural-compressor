"""Tests for strategy utility."""

import unittest

from neural_compressor.strategy.utils.utility import build_slave_faker_model


class TestUtils(unittest.TestCase):
    def test_build_slave_faker_model(self):
        faker_model = build_slave_faker_model()
        faker_model.some_method(0, a=1)
        faker_model.some_attr
        faker_model.some_attr.another_attr[0].some_method()


if __name__ == "__main__":
    unittest.main()
