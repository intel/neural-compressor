import copy
import os
import shutil
import unittest

try:
    import habana_frameworks.torch.hpex

    USE_HPEX = True
except:
    USE_HPEX = False
import torch

from neural_compressor.common import logger
from neural_compressor.torch.amp import autocast


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 10)

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(x1)
        x3 = torch.matmul(inp.T, x2)
        x3 = x3.unsqueeze(0)
        x3 = torch.bmm(x3, x3)
        return x3


@unittest.skipIf(not USE_HPEX, "HPEX is required for HPU inference")
class TestPytorchFP8Adaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = M().to("hpu")
        self.inp = torch.randn(1, 10).to("hpu")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("./.graph_dumps", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_autocast(self):
        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        with autocast("hpu", dtype=torch.bfloat16):
            bf16_out = m(inp)
            print("BF16 MSE:", (bf16_out - fp32_out).pow(2).sum())

        with autocast("hpu", dtype=torch.float8_e5m2):
            e5m2_out = m(inp)
            print("FP8_E5M2 MSE:", (e5m2_out - fp32_out).pow(2).sum())

        with autocast("hpu", dtype=torch.float8_e4m3fn):
            e4m3_out = m(inp)
            print("FP8_E4M3 MSE:", (e4m3_out - fp32_out).pow(2).sum())

    def test_autocast_use_amax(self):
        os.environ["PT_USE_FP8_AMAX"] = str(1)
        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        with autocast("hpu", dtype=torch.float8_e5m2):
            e5m2_out = m(inp)
            print("FP8_E5M2 using amax MSE:", (e5m2_out - fp32_out).pow(2).sum())

        with autocast("hpu", dtype=torch.float8_e4m3fn):
            e4m3_out = m(inp)
            print("FP8_E4M3 using amax MSE:", (e4m3_out - fp32_out).pow(2).sum())
        os.environ.pop("PT_USE_FP8_AMAX", None)


if __name__ == "__main__":
    unittest.main()
