import argparse
import os
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default="checkpoint-best-acc/")
    parser.add_argument("--output_model", type=str, required=False, default="onnx_model/")
    return parser.parse_args()

def prepare_model(input_model, output_model):
    print("\nexport model...")
    subprocess.run(
        [
            "optimum-cli",
            "export",
            "onnx",
            "--model",
            f"{input_model}",
            "--task",
            "text-classification",
            f"{output_model}",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    assert os.path.exists(output_model), f"{output_model} doesn't exist!"


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)