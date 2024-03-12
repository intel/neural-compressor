import argparse
import os
from optimum.exporters.onnx import main_export

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default="checkpoint-best-acc")
    parser.add_argument("--output_model", type=str, required=False, default="codebert-exported-onnx")
    return parser.parse_args()

def prepare_model(input_model, output_model):
    print("\nexport model...")
    print(f"Try to export model from {input_model} to {output_model}")
    main_export(input_model, output=output_model, task="text-classification")

    assert os.path.exists(output_model), f"{output_model} doesn't exist!"


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)