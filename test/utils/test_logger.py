"""Tests for logging utilities."""
from neural_compressor.utils import logger
import unittest

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
        logger.warn("call logger warn function")
        logger.warn({"msg": "call logger warn function"})
        logger.warning("call logger warning function")
        logger.warning({"msg": "call logger warning function"})


if __name__ == "__main__":
    unittest.main()