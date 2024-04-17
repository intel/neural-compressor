import argparse

import habana_frameworks.torch.core as htcore
import torch

htcore.hpu_set_env()

from neural_compressor.torch import FP8QuantConfig, convert, finalize_calibration, prepare

torch.manual_seed(1)


# 1. python sample_one_step.py --quant_config=quant_config.json


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 10)

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(x1)
        return x2


def eval_func(model):
    # user's eval func
    input = torch.randn(1, 10)
    model(input.to("hpu"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habana FP8 sample code.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--quant_config", type=str, help="json file of quantization config")
    args = parser.parse_args()

    model = M().eval().to("hpu")

    config = FP8QuantConfig.from_json_file(args.quant_config)
    model = prepare(model, config)
    eval_func(model)
    finalize_calibration(model)
    model = convert(model)
    print(model)
