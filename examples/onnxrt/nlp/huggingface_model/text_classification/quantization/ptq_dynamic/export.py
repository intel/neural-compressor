import argparse

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

def export_onnx_model(args, model):
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        if args.model_name_or_path in ['Intel/roberta-base-mrpc', 
                                        'Intel/xlm-roberta-base-mrpc', 
                                        'Intel/camembert-base-mrpc', 
                                        'distilbert-base-uncased-finetuned-sst-2-english',
                                        'Intel/xlnet-base-cased-mrpc']:
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
        print("ONNX Model exported to {0}".format(args.output_model))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Export huggingface onnx model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_name_or_path',
        type=str,
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
                'Intel/bart-large-mrpc'],
        help='pretrained model name or path')
    parser.add_argument(
        '--max_len',
        type=int,
        default=128,
        help='Maximum length of the sentence pairs')
    args = parser.parse_args()
    args.output_model = args.model_name_or_path.split('/')[-1] + '.onnx'

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=AutoConfig.from_pretrained(args.model_name_or_path))

    if args.model_name_or_path == 'Intel/bart-large-mrpc':
        import os
        os.system('python -m transformers.onnx --model=Intel/bart-large-mrpc --feature=sequence-classification bart-large-mrpc')
    else:
        export_onnx_model(args, model)