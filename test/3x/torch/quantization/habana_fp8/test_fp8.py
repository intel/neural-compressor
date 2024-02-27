import copy
import shutil

import pytest
import torch

from neural_compressor.torch.utils import is_hpex_available

if is_hpex_available():
    from neural_compressor.torch.algorithms.habana_fp8 import quantize_dynamic
    from neural_compressor.torch.algorithms.habana_fp8.modules import (
        BatchMatmul,
        FP8BatchMatmul,
        FP8DynamicBatchMatmul,
        FP8DynamicLinear,
        FP8DynamicMatmul,
        FP8Linear,
        FP8Matmul,
        Matmul,
    )
    from neural_compressor.torch.quantization import quantize
    from neural_compressor.torch.quantization.config import FP8Config, get_default_fp8_config

    torch.set_grad_enabled(False)


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


@pytest.mark.skipif(not is_hpex_available(), reason="no hpex in environment here.")
class TestPytorchFP8Adaptor:
    def setup_class(self):
        self.model = M().to("hpu")
        self.inp = torch.randn(1, 10).to("hpu")

    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("./.graph_dumps", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_dynamic_accu(self):
        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        m = quantize_dynamic(m, dtype="fp8_e5m2", inplace=True)
        assert isinstance(m.fc1, FP8DynamicLinear), "Unexpected result. Please double check."
        assert isinstance(m.mm, FP8DynamicMatmul), "Unexpected result. Please double check."
        assert isinstance(m.bmm, FP8DynamicBatchMatmul), "Unexpected result. Please double check."
        print(m)
        fp8_out = m(inp)
        print("Dynamic quantization FP8_E5M2 MSE:", (fp32_out - fp8_out).pow(2).sum())

        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        m = quantize_dynamic(m, dtype="fp8_e4m3", inplace=True)
        assert isinstance(m.fc1, FP8DynamicLinear), "Unexpected result. Please double check."
        assert isinstance(m.mm, FP8DynamicMatmul), "Unexpected result. Please double check."
        assert isinstance(m.bmm, FP8DynamicBatchMatmul), "Unexpected result. Please double check."
        print(m)
        fp8_out = m(inp)
        print("Dynamic quantization FP8_E4M3 MSE:", (fp32_out - fp8_out).pow(2).sum())

        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        qconfig = FP8Config(approach="dynamic")
        m = quantize(m, qconfig, inplace=True)
        assert isinstance(m.fc1, FP8DynamicLinear), "Unexpected result. Please double check."
        assert isinstance(m.mm, FP8DynamicMatmul), "Unexpected result. Please double check."
        assert isinstance(m.bmm, FP8DynamicBatchMatmul), "Unexpected result. Please double check."
        print(m)
        fp8_out = m(inp)
        print("Dynamic quantization FP8_E4M3 MSE:", (fp32_out - fp8_out).pow(2).sum())

    def test_static_accu(self):
        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        qconfig = FP8Config(w_dtype="fp8_e5m2", act_dtype="fp8_e5m2", approach="static")

        def calib_func(model):
            model(inp)

        m = quantize(m, qconfig, run_fn=calib_func, inplace=True)
        assert isinstance(m.fc1, FP8Linear), "Unexpected result. Please double check."
        assert isinstance(m.mm, FP8Matmul), "Unexpected result. Please double check."
        assert isinstance(m.bmm, FP8BatchMatmul), "Unexpected result. Please double check."
        print(m)
        fp8_out = m(inp)
        print("Static quantization FP8_E5M2 MSE:", (fp32_out - fp8_out).pow(2).sum())

        m = copy.deepcopy(self.model)
        inp = self.inp
        fp32_out = m(inp)
        qconfig = get_default_fp8_config()

        def calib_func(model):
            model(inp)

        m = quantize(m, qconfig, run_fn=calib_func, inplace=True)
        assert isinstance(m.fc1, FP8Linear), "Unexpected result. Please double check."
        assert isinstance(m.mm, FP8Matmul), "Unexpected result. Please double check."
        assert isinstance(m.bmm, FP8BatchMatmul), "Unexpected result. Please double check."
        print(m)
        fp8_out = m(inp)
        print("Static quantization FP8_E4M3 MSE:", (fp32_out - fp8_out).pow(2).sum())

    def test_convert(self):
        # Temporary implementation of fp8 tensor saving and loading
        # Will remove after Habana torch applies below patch:
        # https://github.com/pytorch/pytorch/pull/114662
        # e4m3
        fp8_inp = torch.ops.hpu.cast_to_fp8_v2(self.inp, 500, dtype=torch.float8_e4m3fn)[0].to("cpu")
        import fp8_convert

        int8_inp = fp8_convert.to_u8(fp8_inp)
        torch.save(int8_inp, "tmp.pt")
        saved_int8_inp = torch.load("tmp.pt")
        recovered_inp = fp8_convert.from_u8(saved_int8_inp, 1)
        assert (fp8_inp == recovered_inp).all(), "Unexpected result. Please double check."
        # e5m2
        fp8_inp = torch.ops.hpu.cast_to_fp8_v2(self.inp, 500, dtype=torch.float8_e5m2)[0].to("cpu")
        int8_inp = fp8_convert.to_u8(fp8_inp)
        recovered_inp = fp8_convert.from_u8(int8_inp, 0)
        assert (fp8_inp == recovered_inp).all(), "Unexpected result. Please double check."

    def test_save_load(self):
        m = copy.deepcopy(self.model)
        inp = self.inp
        qconfig = get_default_fp8_config()

        def calib_func(model):
            model(inp)

        m = quantize(m, qconfig, run_fn=calib_func, inplace=True)
        fp8_out = m(inp)
        m.save("saved_results")

        from neural_compressor.torch.quantization import load

        m = copy.deepcopy(self.model)
        m = load(m, "saved_results")
        recovered_out = m(inp)
        assert (recovered_out == fp8_out).all(), "Unexpected result. Please double check."
        assert isinstance(m.fc1, FP8Linear), "Unexpected result. Please double check."
        assert isinstance(m.mm, FP8Matmul), "Unexpected result. Please double check."
        assert isinstance(m.bmm, FP8BatchMatmul), "Unexpected result. Please double check."
