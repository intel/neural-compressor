import argparse
import os
import shutil
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument("--output_model", type=str, required=True)
    return parser.parse_args()

def move_and_rename_model(source_folder, destination_folder):
    if not os.path.exists(source_folder):
        raise RuntimeError("{} path is not exists".format(source_folder))
    for file_name in os.listdir(source_folder):
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        
        if os.path.isdir(source_file):
            continue
    
        shutil.move(source_file, destination_file)

        if file_name == "model.onnx":
            new_file_name = "unet-export.onnx"
            new_file_path = os.path.join(destination_folder, new_file_name)
            os.rename(destination_file, new_file_path)

def prepare_model(input_model, output_model):
    # Use [tf2onnx tool](https://github.com/onnx/tensorflow-onnx) to convert tflite to onnx model.
    print("\nexport model...")

    export_file = "prepare_unet"
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
            export_file,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    move_and_rename_model(os.path.join(export_file, "unet"), os.path.dirname(output_model))
    try:
        shutil.rmtree(export_file, ignore_errors=True)
    except OSError as e:
        raise e
    
    assert os.path.exists(output_model), f"Export failed! {output_model} doesn't exist!"


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)
