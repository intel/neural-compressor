"""Tests for utils components."""

import unittest
from neural_compressor_ort import utility

log_msg_lst = [
    "call logger log function.",
    {
        "msg": "call logger log function."
    },
    ["call logger warning function", "done"],
    ("call logger warning function", "done"),
    # the following log will be prettified
    {
        "msg": "call logger warning function"
    },
    {
        "msg": {
            ("bert", "embedding"): {
                "weight": {
                    "dtype": ["unint8", "int8"]
                }
            }
        }
    },
    {
        "msg": {
            ("bert", "embedding"): {
                "op": ("a", "b")
            }
        }
    },
    # the following log will not be prettified
    [{
        "msg": "call logger warning function"
    }, {
        "msg2": "done"
    }],
    ({
        "msg": "call logger warning function"
    }, {
        "msg2": "done"
    }),
    ({
        "msg": [{
            "sub_msg": "call logger"
        }, {
            "sub_msg2": "call warning function"
        }]
    }, {
        "msg2": "done"
    }),
]


class TestLogger(unittest.TestCase):

    def test_logger(self):

        for msg in log_msg_lst:
            utility.logger.log(0, msg)
            utility.logger.log(1, msg)
            utility.logger.debug(msg)
            utility.logger.error(msg)
            utility.logger.fatal(msg)
            utility.logger.info(msg)
            utility.logger.warning(msg)

if __name__ == "__main__":
    unittest.main()
