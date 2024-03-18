# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_x86_inductor.html

import copy
import os

import torch
import torch._inductor.config as config
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

# set TOKENIZERS_PARALLELISM to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse

from tqdm import tqdm


def quant(model_name_or_path, fold_quantize=False, eval=False):
    # Create the Eager Model
    # model_name = "resnet18"
    # model = models.__dict__[model_name](pretrained=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    text = "Hello, welcome to LLM world."
    encoded_input = tokenizer(text, return_tensors="pt")

    example_inputs = encoded_input
    print(example_inputs)

    tuple_inputs = (example_inputs["input_ids"],)

    # Set the model to eval mode
    model = model.eval()

    # Create the data, using the dummy data here as an example

    with torch.no_grad():
        float_output = model(*tuple_inputs)
        print(float_output)

    # Capture the FX Graph to be quantized
    with torch.no_grad():
        # if you are using the PyTorch nightlies or building from source with the pytorch master,
        # use the API of `capture_pre_autograd_graph`
        # Note 1: `capture_pre_autograd_graph` is also a short-term API, it will be updated to use the official `torch.export` API when that is ready.
        exported_model = capture_pre_autograd_graph(model, tuple_inputs)
        # Note 2: if you are using the PyTorch 2.1 release binary or building from source with the PyTorch 2.1 release branch,
        # please use the API of `torch._dynamo.export` to capture the FX Graph.
        # exported_model, guards = torch._dynamo.export(
        #     model,
        #     *copy.deepcopy(example_inputs),
        #     aten_graph=True,
        # )

    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    prepared_model = prepare_pt2e(exported_model, quantizer)

    with torch.no_grad():
        for i in tqdm(range(2)):
            prepared_model(*tuple_inputs)

    converted_model = convert_pt2e(prepared_model, fold_quantize=fold_quantize)

    # Optional: using the C++ wrapper instead of default Python wrapper
    config.cpp_wrapper = False

    # # Running some benchmark
    with torch.no_grad():
        optimized_model = torch.compile(converted_model)
        output = optimized_model(*tuple_inputs)
        print(output)
        return output is not None
    return False

    # # move the quantized model to eval mode, equivalent to `m.eval()`
    # torch.ao.quantization.move_exported_model_to_eval(converted_model)

    # # Lower the model into Inductor
    # with torch.no_grad():
    #   optimized_model = torch.compile(converted_model)
    #   _ = optimized_model(*example_inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--fold_quantize", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    args = parser.parse_args()
    quant(model_name_or_path=args.model_name_or_path, fold_quantize=args.fold_quantize, eval=args.eval)


if __name__ == "__main__":
    main()


# Successful runs:
# TORCH_LOGS="+dynamo"  p  x86_quantizer_hf.py --model_name_or_path /mnt/disk4/modelHub/gpt-j-6b
# TORCH_LOGS="+dynamo"  p  x86_quantizer_hf.py --model_name_or_path /mnt/disk4/modelHub/llama-2-7b-chat-hg
# TORCH_LOGS="+dynamo"  p  x86_quantizer_hf.py --model_name_or_path /mnt/disk4/modelHub/Mistral-7B-v0.1
# TORCH_LOGS="+dynamo"  p  x86_quantizer_hf.py --model_name_or_path gpt2
# TORCH_LOGS="+dynamo"  p  x86_quantizer_hf.py --model_name_or_path facebook/opt-125m
# TORCH_LOGS="+dynamo"  p  x86_quantizer_hf.py --model_name_or_path bigscience/bloom

# Failed runs:
# TORCH_LOGS="+dynamo"  p  x86_quantizer_hf.py --model_name_or_path facebook/opt-125m --fold_quantize
