"""Tests for logger components."""

import unittest

from onnx_neural_compressor import logger

log_msg_lst = [
    "call logger log function.",
    {"msg": "call logger log function."},
    ["call logger warning function", "done"],
    ("call logger warning function", "done"),
    # the following log will be prettified
    {"msg": "call logger warning function"},
    {"msg": {("bert", "embedding"): {"weight": {"dtype": ["unint8", "int8"]}}}},
    {"msg": {("bert", "embedding"): {"op": ("a", "b")}}},
    # the following log will not be prettified
    [{"msg": "call logger warning function"}, {"msg2": "done"}],
    ({"msg": "call logger warning function"}, {"msg2": "done"}),
    ({"msg": [{"sub_msg": "call logger"}, {"sub_msg2": "call warning function"}]}, {"msg2": "done"}),
]


class TestLogger(unittest.TestCase):

    def test_logger(self):

        for msg in log_msg_lst:
            logger.log(0, msg)
            logger.log(1, msg)
            logger.debug(msg)
            logger.error(msg)
            logger.fatal(msg)
            logger.info(msg)
            logger.warning(msg)


if __name__ == "__main__":
    unittest.main()
