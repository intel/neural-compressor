import argparse
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        type=str,
                        required=False,
                        default="Intel/bert-base-uncased-mrpc")
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--task", type=str, required=False, default="text-classification")
    return parser.parse_args()


def prepare_model(input_model, output_model, task):
    # Use [tf2onnx tool](https://github.com/onnx/tensorflow-onnx) to convert tflite to onnx model.
    print("\nexport model...")
    subprocess.run(
        [
            "optimum-cli",
            "export",
            "onnx",
            "--model",
            input_model,
            "--task",
            task,
            output_model,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model, args.task)
