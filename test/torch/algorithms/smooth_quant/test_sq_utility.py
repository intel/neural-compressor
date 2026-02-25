import copy

import pytest
import torch
from packaging.version import Version


class Model(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


model = Model()

@pytest.mark.skipif(not Version(torch.__version__) < Version("2.9.0"), reason="only for torch<2.9.0 [ipex]")
def test_captured_dataloader():
    from neural_compressor.torch.algorithms.smooth_quant import build_captured_dataloader

    fp32_model = copy.deepcopy(model)

    def run_fn(model):
        for i in range(10):
            example_inputs = torch.randn([1, 3])
            model(example_inputs)

    tmp_model, dataloader = build_captured_dataloader(fp32_model, run_fn, calib_num=32)
    assert tmp_model == fp32_model, "Model should be same after building dataloader. Please check."
    assert isinstance(dataloader.args_list[0][0], torch.Tensor), "Args list should contain tensors. Please check."
    assert not dataloader.kwargs_list[0], "Kwargs list should be empty. Please check."
