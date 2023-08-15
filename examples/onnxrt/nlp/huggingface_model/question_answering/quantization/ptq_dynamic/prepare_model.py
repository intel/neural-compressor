import argparse

import torch
from transformers import AutoConfig, AutoModelForQuestionAnswering

def export_onnx_model(args, model):
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        if args.input_model in ['distilbert-base-uncased-distilled-squad',
                                       'deepset/roberta-large-squad2']:
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
                            output_names=['start_logits',
                                          'end_logits'],
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
                            output_names=['start_logits',
                                          'end_logits'],
                            dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                          'attention_mask' : symbolic_names,
                                          'token_type_ids' : symbolic_names})
        print("ONNX Model exported to {0}".format(args.output_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Export huggingface onnx model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_model',
                        type=str,
                        default='mrm8488/spanbert-finetuned-squadv1',
                        const='mrm8488/spanbert-finetuned-squadv1',
                        nargs='?',
                        choices=[
                            'mrm8488/spanbert-finetuned-squadv1',
                            'salti/bert-base-multilingual-cased-finetuned-squad',
                            'distilbert-base-uncased-distilled-squad',
                            'bert-large-uncased-whole-word-masking-finetuned-squad',
                            'deepset/roberta-large-squad2'
                        ],
                        help='pretrained model name or path ')
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument(
        '--max_len',
        type=int,
        default=512,
        help='Maximum length of the sentence pairs')
    args = parser.parse_args()

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.input_model,
        config=AutoConfig.from_pretrained(args.input_model))

    export_onnx_model(args, model)