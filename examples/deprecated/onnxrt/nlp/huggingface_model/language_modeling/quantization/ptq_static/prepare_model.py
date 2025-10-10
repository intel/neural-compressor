import argparse
import os

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODELS = {
    'gpt2':(GPT2LMHeadModel, GPT2Tokenizer, 'gpt2'),
    'distilgpt2': (GPT2LMHeadModel, GPT2Tokenizer, 'distilgpt2')
}
data_dir = 'test_data_set_0'


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list


def save_model(name, model, inputs, outputs, input_names=None, output_names=None, **kwargs):
    if hasattr(model, 'train'):
        model.train(False)

    inputs_flatten = flatten(inputs)
    inputs_flatten = update_flatten_list(inputs_flatten, [])
    outputs_flatten = flatten(outputs)
    outputs_flatten = update_flatten_list(outputs_flatten, [])
    if input_names is None:
        input_names = []
        for i, _ in enumerate(inputs_flatten):
            input_names.append('input' + str(i+1))
    else:
        np.testing.assert_equal(len(input_names), len(inputs_flatten),
                                "Number of input names provided is not equal to the number of inputs.")

    if output_names is None:
        output_names = []
        for i, _ in enumerate(outputs_flatten):
            output_names.append('output' + str(i+1))
    else:
        np.testing.assert_equal(len(output_names), len(outputs_flatten),
                                "Number of output names provided is not equal to the number of output.")

    if isinstance(model, torch.jit.ScriptModule):
        torch.onnx._export(model, inputs, name, verbose=True, input_names=input_names,
                           output_names=output_names, **kwargs)
    else:
        torch.onnx.export(model, inputs, name, verbose=True, input_names=input_names,
                          output_names=output_names, **kwargs)
    assert os.path.exists(name), f"Export failed, {name} doesn't exist!"


def gpt2_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', type=str,  required=False, default="gpt2")
    parser.add_argument('--output_model', type=str,  required=True, 
                        help='model name or path.')
    args = parser.parse_args()

    model_class, tokenizer_class, pretrained_weights = MODELS[args.input_model]
        # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    model.eval()
    # Encode text
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    input_ids_1 = torch.tensor(
        [[tokenizer.encode("Here is some text to encode Hello World", add_special_tokens=True)]])
    with torch.no_grad():
        output_1 = model(input_ids_1)  # Models outputs are now tuples

    save_model(args.output_model, model.cpu(), input_ids_1, output_1, opset_version=14, input_names=['input1'], dynamic_axes={'input1': [0, 1, 2]}) 


gpt2_test()
