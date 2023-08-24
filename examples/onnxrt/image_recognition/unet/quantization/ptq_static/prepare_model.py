import argparse
import os
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument("--output_model", type=str, required=True)
    return parser.parse_args()


def prepare_model(input_model, output_model):
    # Use [tf2onnx tool](https://github.com/onnx/tensorflow-onnx) to convert tflite to onnx model.
    print("\nexport model...")
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/huggingface/diffusers.git",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["pip", "install", "--upgrade", "diffusers[torch]", "transformers"],
        stdout=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        [
            "python",
            "diffusers/scripts/convert_stable_diffusion_checkpoint_to_onnx.py",
            "--model_path",
            input_model,
            "--output_path",
            output_model,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert os.path.exists(output_model), f"Export failed! {output_model} doesn't exist!"


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)
