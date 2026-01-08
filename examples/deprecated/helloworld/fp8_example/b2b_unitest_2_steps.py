
import argparse
import math

import torch
import habana_frameworks.torch.core as htcore
from torch.nn import Parameter, init

# Initialize HPU environment (must be called before HPU operations)
htcore.hpu_set_env()

from neural_compressor.torch.quantization import FP8Config, convert, finalize_calibration, prepare

torch.manual_seed(1)


class B2BMatmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, **kwargs):
        return torch.matmul(x, y, **kwargs)



class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul = B2BMatmul()

    def forward(self, inp0, inp1):
        res = self.matmul(inp0, inp1)

        return res


def main():
    parser = argparse.ArgumentParser(
        description="Habana FP8 sample code with B2BMatmul.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quant_config", type=str, help="JSON file of quantization config")
    args = parser.parse_args()

    # Build model & load config
    model = M().eval()
    config = FP8Config.from_json_file(args.quant_config)

    # Optional calibration preparation
    if config.measure:
        model = prepare(model, config)

    # Optional quantization
    if config.quantize:
        htcore.hpu_initialize()
        model = convert(model, config)
        print(model)

    # Create inputs and run

    with torch.no_grad():
        model.to("hpu")

        B = 6
        N = 100

        inp0=  torch.tensor([
            [1,0,0,0,0,0],  # row 0 <- X[0]
            [0,0,0,1,0,0],  # row 1 <- X[3]
            [0,1,0,0,0,0],  # row 2 <- X[1]
            [0,0,0,0,1,0],  # row 3 <- X[4]
            [0,0,0,0,0,0],  # row 4 <- X[2]
            [0,0,0,0,0,0],  # row 5 <- X[5]
        ], dtype=torch.float32).to("hpu")

        # Input for Matmul: [B, D] -> now [6, 100]
        inp1 = torch.randn(B, N)
        inp1[2, :] = 1000
        inp1[5, :] = 1000


        # Run the model
        output = model(inp0, inp1)
        print("Output shape:", output.shape)
        print(output)


    # Finalize calibration if measuring
    if config.measure:
        finalize_calibration(model)


if __name__ == "__main__":
    main()
