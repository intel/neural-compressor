import torch
from quantization.config import Fp8cfg
from quantization.quantize import prepare, convert, save_calib

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 10)

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(x1)
        return x2


# model = M().to("hpu")
# input = torch.randn(1, 10).to("hpu")
model = M()
input = torch.randn(1, 10)

quant_config = Fp8cfg()
prepared_model = prepare(model, quant_config)

#
prepared_model(input)

# save_calib(prepared_model)
# q_model = convert(prepared_model)
