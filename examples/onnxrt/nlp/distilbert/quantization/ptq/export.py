import argparse

import torch
from transformers import DistilBertForSequenceClassification

def export_onnx_model(args, model, onnx_model_path):
    with torch.no_grad():
        inputs = {'input_ids':      torch.ones(1,args.max_len, dtype=torch.int64),
                    'attention_mask': torch.ones(1,args.max_len, dtype=torch.int64)}
        outputs = model(**inputs)

        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,                            # model being run
                    (inputs['input_ids'],                
                    inputs['attention_mask']),              # model input (or a tuple for
                                                            # multiple inputs)
                    onnx_model_path,                        # where to save the model (can be a file
                                                            # or file-like object)
                    opset_version=11,                       # the ONNX version to export the model
                    do_constant_folding=True,               # whether to execute constant folding
                    input_names=['input_ids',               # the model's input names
                                'input_mask'],
                    output_names=['output'],                # the model's output names
                    dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                'input_mask' : symbolic_names})
        print("ONNX Model exported to {0}".format(onnx_model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Export bert onnx model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_dir',
        type=str,
        help='input_dir of bert model, must contain config.json')
    parser.add_argument(
        '--task_name',
        type=str,
        choices=["MRPC", "MNLI"],
        help='tasks names of bert model')
    parser.add_argument(
        '--max_len',
        type=int,
        default=128,
        help='Maximum length of the sentence pairs')
    parser.add_argument(
        '--do_lower_case',
        type=bool,
        default=True,
        help='whether lower the tokenizer')
    parser.add_argument(
        '--output_model',
        type=str,
        default='bert.onnx',
        help='path to exported model file')
    args = parser.parse_args()

    model = DistilBertForSequenceClassification.from_pretrained(args.input_dir)
    export_onnx_model(args, model, args.output_model)
