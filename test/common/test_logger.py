"""Tests for common logger.

!!! Please do not import any framework-specific modules in this file. !!!
* Note, we may need to add some auto check mechanisms to ensure this.

These tests aim to assess the fundamental functionalities of common components and enhance code coverage.
All tests will be included for each framework CI.
"""

import unittest
from unittest.mock import patch

from neural_compressor.common.utils import Logger

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
        from neural_compressor.common import logger

        for msg in log_msg_lst:
            logger.log(0, msg)
            logger.log(1, msg)
            logger.debug(msg)
            logger.error(msg)
            logger.fatal(msg)
            logger.info(msg)
            logger.warning(msg)
        # logger.log(0, "call logger log function.")
        # logger.log(1, {"msg": "call logger log function."})
        # logger.debug("call logger debug function.")
        # logger.debug({"msg": "call logger debug function."})
        # logger.error("call logger error function.")
        # logger.error({"msg": "call logger error function."})
        # logger.fatal("call logger fatal function")
        # logger.fatal({"msg": "call logger fatal function"})
        # logger.info("call logger info function")
        # logger.info({"msg": "call logger info function."})
        # logger.warning("call logger warning function")
        # logger.warning({"msg": "call logger warning function"})
        # logger.warning(["call logger warning function", "done"])
        # logger.warning(("call logger warning function", "done"))
        # logger.warning({"msg": {("bert", "embedding"): {"weight": {"dtype": ["unint8", "int8"]}}}})
        # logger.warning({"msg": {("bert", "embedding"): {"op": ("a", "b")}}})
        # # the following log will not be prettified
        # logger.warning([{"msg": "call logger warning function"}, {"msg2": "done"}])
        # logger.warning(({"msg": "call logger warning function"}, {"msg2": "done"}))
        # logger.warning(({"msg": [{"sub_msg": "call logger"}, {"sub_msg2": "call warning function"}]}, {"msg2": "done"}))

    # def test_logger_func_and_pretty_dict(self):
    #     from neural_compressor.common.utils import debug, error, fatal, info, log, warning

    #     for msg in log_msg_lst:
    #         log(0, msg)
    #         log(1, msg)
    #         debug(msg)
    #         error(msg)
    #         fatal(msg)
    #         info(msg)
    #         warning(msg)

    @patch.object(Logger, "warning")
    def test_warning_once(self, mock_method):

        warning_message = "test warning message"
        # First call
        Logger.warning_once(warning_message)
        mock_method.assert_called_with(warning_message)
        # Second call
        Logger.warning_once(warning_message)
        Logger.warning_once(warning_message)
        # Call `warning_once` 3 times, but `warning` should only be called twice,
        # one for help message and one for warning message.
        assert mock_method.call_count == 2, "Expected warning to be called twice."


if __name__ == "__main__":
    unittest.main()
