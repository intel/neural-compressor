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


def eval_func(model):
    # user's eval func
    pass


from neural_compressor.torch import FP8QuantConfig, convert, prepare, save_calibration_result

quant_config = FP8QuantConfig()

# prepare the model for quantization if needed
model = prepare(model, quant_config)

# reuse user's eval func to do calibration
eval_func(model)

# save calibration results to local file if needed
save_calibration_result(model, quant_config)

# convert the origin model to a quantized model
model = convert(model, quant_config)

# ? have to call user's eval func again to evaluate quantized model
