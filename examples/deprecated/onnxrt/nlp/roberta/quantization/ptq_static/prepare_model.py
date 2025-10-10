import argparse
import re
import subprocess

import torch
from transformers import RobertaForSequenceClassification


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default="roberta-base")
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument('--input_dir',
                        type=str,
                        default="roberta",
                        help='input_dir of bert model, must contain config.json')
    parser.add_argument('--task_name',
                        type=str,
                        choices=["MRPC", "MNLI"],
                        default="MRPC",
                        help='tasks names of bert model')
    parser.add_argument('--max_len',
                        type=int,
                        default=128,
                        help='Maximum length of the sentence pairs')
    return parser.parse_args()

def comment_out_line(filepath, code):
    modified_lines = []

    with open(filepath, 'r') as file:
        lines = file.readlines()
        file.seek(0)
        for line in lines:
            if re.match(code, line.strip()):
                line = "#" + line
            modified_lines.append(line)
    
    with open(filepath, 'w') as file:
        file.writelines(modified_lines)

def prepare_model(input_model, output_model, task_name):
    print("\nexport model...")
    subprocess.run(
        ["git", "clone", "https://github.com/huggingface/transformers.git", "my_transformers"],
        stdout=subprocess.PIPE,
        text=True,
    )

    subprocess.run(
        [
            "pip", "install", "-r",
            "my_transformers/examples/pytorch/text-classification/requirements.txt"
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    # remove transformers min version check
    comment_out_line("my_transformers/examples/pytorch/text-classification/run_glue.py", 
                     r"check_min_version\(.*\)")

    subprocess.run(
        [
            "python",
            "-m",
            "my_transformers.examples.pytorch.text-classification.run_glue",
            f"--model_name_or_path={input_model}",
            f"--task_name={task_name}",
            "--do_train",
            "--do_eval",
            "--max_seq_length=128",
            "--per_gpu_eval_batch_size=8 ",
            "--per_gpu_train_batch_size=8 ",
            "--learning_rate=2e-5",
            "--num_train_epochs=3.0",
            "--save_steps=100000 ",
            f"--output_dir={output_model}",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )


def export_onnx_model(args, model, onnx_model_path):
    with torch.no_grad():
        inputs = {
            'input_ids': torch.ones(1, args.max_len, dtype=torch.int64),
            'attention_mask': torch.ones(1, args.max_len, dtype=torch.int64)
        }

        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,                            # model being run
                    (inputs['input_ids'],
                    inputs['attention_mask']),              # model input (or a tuple for
                                                            # multiple inputs)
                    onnx_model_path,                        # where to save the model (can be a file
                                                            # or file-like object)
                    opset_version=14,                       # the ONNX version to export the model
                    do_constant_folding=True,               # whether to execute constant folding
                    input_names=['input_ids',               # the model's input names
                                'input_mask'],
                    output_names=['output'],                # the model's output names
                    dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                'input_mask' : symbolic_names})
        print("ONNX Model exported to {0}".format(onnx_model_path))



if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    prepare_model(args.input_model, args.input_dir, args.task_name)
    model = RobertaForSequenceClassification.from_pretrained(args.input_dir)
    export_onnx_model(args, model, args.output_model)
