import argparse
import os
import subprocess
import optimum.version
from packaging.version import Version
OPTIMUM114_VERSION = Version("1.14.0")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default="")
    parser.add_argument("--output_model", type=str, required=True)
    return parser.parse_args()


def prepare_model(input_model, output_model):
    print("\nexport model...")
    if Version(optimum.version.__version__) >= OPTIMUM114_VERSION:
        subprocess.run(
            [
                "optimum-cli",
                "export",
                "onnx",
                "--model",
                f"{input_model}",
                "--task",
                "text-generation-with-past",
                "--legacy",
                f"{output_model}",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
    else:
        subprocess.run(
            [
                "optimum-cli",
                "export",
                "onnx",
                "--model",
                f"{input_model}",
                "--task",
                "text-generation-with-past",
                f"{output_model}",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )

    assert os.path.exists(output_model), f"{output_model} doesn't exist!"


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)
