import argparse
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default="")
    parser.add_argument("--output_model", type=str, required=True)
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
            "text-generation-with-past",
            f"{output_model}",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)
