import torch

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 10)

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(x1)
        return x2

model = M().to("hpu")

from neural_compressor.torch import FP8QuantConfig, prepare, convert, save_calib
quant_config = FP8QuantConfig()

# prepare the model for quantization if needed
model = prepare(model, quant_config)

# use user's eval func to do calibration
user_func(model)

# save calibration results to local file if needed
save_calib(model, quant_config)

# convert the origin model to a quantized model
model = convert(model, quant_config)

# ? have to call user's eval func again to evaluate quantized model
