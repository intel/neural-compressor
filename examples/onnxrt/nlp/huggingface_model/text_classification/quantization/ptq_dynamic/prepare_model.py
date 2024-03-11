import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

def export_onnx_model(args, model):
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        if args.input_model in [
                'Intel/roberta-base-mrpc',
                'Intel/xlm-roberta-base-mrpc',
                'Intel/camembert-base-mrpc',
                'distilbert-base-uncased-finetuned-sst-2-english',
                'Intel/xlnet-base-cased-mrpc',
                'Intel/deberta-v3-base-mrpc',
        ]:
            inputs = {'input_ids':      torch.ones(1, args.max_len, dtype=torch.int64),
                      'attention_mask': torch.ones(1, args.max_len, dtype=torch.int64)}
            torch.onnx.export(model,                            # model being run
                            (inputs['input_ids'],               # model input (or a tuple for multiple inputs) 
                            inputs['attention_mask']),
                            args.output_model,                  # where to save the model (can be a file or file-like object)
                            opset_version=14,                   # the ONNX version to export the model
                            do_constant_folding=True,           # whether to execute constant folding
                            input_names=['input_ids',           # the model's input names
                                        'attention_mask'],
                            output_names=['logits'],
                            dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                        'attention_mask' : symbolic_names})
        else:
            inputs = {'input_ids':      torch.ones(1, args.max_len, dtype=torch.int64),
                      'attention_mask': torch.ones(1, args.max_len, dtype=torch.int64),
                      'token_type_ids': torch.ones(1, args.max_len, dtype=torch.int64)}
            torch.onnx.export(model,                            # model being run
                            (inputs['input_ids'],               # model input (or a tuple for multiple inputs) 
                            inputs['attention_mask'],
                            inputs['token_type_ids']),
                            args.output_model,                  # where to save the model (can be a file or file-like object)
                            opset_version=14,                   # the ONNX version to export the model
                            do_constant_folding=True,           # whether to execute constant folding
                            input_names=['input_ids',           # the model's input names
                                        'attention_mask',
                                        'token_type_ids'],
                            output_names=['logits'],
                            dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                        'attention_mask' : symbolic_names,
                                        'token_type_ids' : symbolic_names})
        assert os.path.exists(args.output_model), f"{args.output_model} doesn't exist!"
        print("ONNX Model exported to {0}".format(args.output_model))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Export huggingface onnx model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_model',
        type=str,
        default='Intel/bert-base-uncased-mrpc',
        const='Intel/bert-base-uncased-mrpc',
        nargs='?',
        choices=['Intel/bert-base-uncased-mrpc',
                'Intel/roberta-base-mrpc',
                'Intel/xlm-roberta-base-mrpc',
                'Intel/camembert-base-mrpc',
                'distilbert-base-uncased-finetuned-sst-2-english',
                'Alireza1044/albert-base-v2-sst2',
                'philschmid/MiniLM-L6-H384-uncased-sst2',
                'Intel/MiniLM-L12-H384-uncased-mrpc',
                'bert-base-cased-finetuned-mrpc',
                'Intel/electra-small-discriminator-mrpc',
                'M-FAC/bert-mini-finetuned-mrpc',
                'Intel/xlnet-base-cased-mrpc',
                'Intel/bart-large-mrpc',
                'Intel/deberta-v3-base-mrpc'
                ],
        help='pretrained model name or path')
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument(
        '--max_len',
        type=int,
        default=128,
        help='Maximum length of the sentence pairs')
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(
        args.input_model,
        config=AutoConfig.from_pretrained(args.input_model))

    if args.input_model == 'Intel/bart-large-mrpc':
        import shutil
        from optimum.exporters.onnx import main_export

        main_export(args.input_model, output="bart-large-mrpc", task="text-classification")
        shutil.move("bart-large-mrpc/model.onnx", args.output_model)
    else:
        export_onnx_model(args, model)
