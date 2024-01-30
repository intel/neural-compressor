"""Tests for logging utilities."""
import unittest

from neural_compressor.common.utils import logger

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
        logger.log(0, "call logger log function.")
        logger.log(1, {"msg": "call logger log function."})
        logger.debug("call logger debug function.")
        logger.debug({"msg": "call logger debug function."})
        logger.error("call logger error function.")
        logger.error({"msg": "call logger error function."})
        logger.fatal("call logger fatal function")
        logger.fatal({"msg": "call logger fatal function"})
        logger.info("call logger info function")
        logger.info({"msg": "call logger info function."})
        logger.warning("call logger warning function")
        logger.warning({"msg": "call logger warning function"})
        logger.warning(["call logger warning function", "done"])
        logger.warning(("call logger warning function", "done"))
        logger.warning({"msg": {("bert", "embedding"): {"weight": {"dtype": ["unint8", "int8"]}}}})
        logger.warning({"msg": {("bert", "embedding"): {"op": ("a", "b")}}})
        # the following log will not be prettified
        logger.warning([{"msg": "call logger warning function"}, {"msg2": "done"}])
        logger.warning(({"msg": "call logger warning function"}, {"msg2": "done"}))
        logger.warning(({"msg": [{"sub_msg": "call logger"}, {"sub_msg2": "call warning function"}]}, {"msg2": "done"}))

    def test_logger_func_and_pretty_dict(self):
        from neural_compressor.common.utils import debug, error, fatal, info, log, warning

        for msg in log_msg_lst:
            log(0, msg)
            log(1, msg)
            debug(msg)
            error(msg)
            fatal(msg)
            info(msg)
            warning(msg)


if __name__ == "__main__":
    unittest.main()
