import torch
import argparse
import habana_frameworks.torch.core as htcore


# 1. python sample.py --calib --calib_result ./hqt_output_2/measure --quant_config=calib.json
# 2. python sample.py --quantize --calib_result ./hqt_output_2/measure --quant_config=quantize.json

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

    if args.calib:
        model = prepare(model, args.quant_config)

    if args.quantize:
        model = convert(model, args.quant_config, args.calib_result)
        print(model)

    eval_func(model)

    if args.calib_result:
        save(model, args.calib_result)
