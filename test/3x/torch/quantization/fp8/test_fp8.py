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
from neural_compressor.torch.quantization import BatchMatmul, Matmul, get_fp8_e4m3_qconfig, get_fp8_e5m2_qconfig
from neural_compressor.torch.quantization.fp8 import quantize, quantize_dynamic

tmp = torch.ops.hpu.cast_to_fp8_v2(torch.tensor(500).to("hpu"), torch.tensor(1).to("hpu"), False, False)[0]

tmp = torch.ops.hpu.cast_from_fp8(tmp, torch.tensor(1).to("hpu"), torch.float32)
logger.info(f"max value: {tmp}")


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 10)
        self.mm = Matmul()
        self.bmm = BatchMatmul()

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(x1)
        x3 = self.mm(inp.T, x2)
        x3 = x3.unsqueeze(0)
        x4 = self.mm(inp.T, x2)
        x4 = x4.unsqueeze(0)
        x5 = self.bmm(x3, x4)
        x6 = self.bmm(x3, x4)
        out = x5 + x6
        return out


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

    def test_dynamic(self):
        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        m = quantize_dynamic(m, dtype=torch.float8_e5m2)
        print(m)
        fp8_out = m(inp)
        print("Dynamic quantization FP8_E5M2 MSE:", (fp32_out - fp8_out).pow(2).sum())

        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        m = quantize_dynamic(m, dtype=torch.float8_e4m3fn)
        print(m)
        fp8_out = m(inp)
        print("Dynamic quantization FP8_E4M3 MSE:", (fp32_out - fp8_out).pow(2).sum())

    def test_static(self):
        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        qconfig = get_fp8_e5m2_qconfig()

        def calib_func(model):
            model(inp)

        m = quantize(m, qconfig, calib_func=calib_func)
        print(m)
        fp8_out = m(inp)
        print("Static quantization FP8_E5M2 MSE:", (fp32_out - fp8_out).pow(2).sum())

        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        qconfig = get_fp8_e4m3_qconfig()

        def calib_func(model):
            model(inp)

        m = quantize(m, qconfig, calib_func=calib_func)
        print(m)
        fp8_out = m(inp)
        print("Static quantization FP8_E4M3 MSE:", (fp32_out - fp8_out).pow(2).sum())


if __name__ == "__main__":
    unittest.main()
