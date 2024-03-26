import torch
import argparse
import habana_frameworks.torch.core as htcore


# 1. python sample_one_step.py --calib --quantize --calib_result ./hqt_output/measure --quant_config=quant_config.json

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


from neural_compressor.torch import convert, prepare, save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habana FP8 sample code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        default=False,
        help="whether to do quantization"
    )
    parser.add_argument(
        '--calib',
        action='store_true',
        default=False,
        help="whether to do calibration"
    )
    parser.add_argument(
        '--quant_config',
        type=str,
        help="json file of quantization config"
    )
    parser.add_argument(
        '--calib_result',
        type=str,
        help="where to dump calibration statistics"
    )
    args = parser.parse_args()

    model = M().eval().to("hpu")

    model = prepare(model, args.quant_config)
    save(model, args.calib_result)
    model = convert(model, args.quant_config, args.calib_result)
    print(model)
