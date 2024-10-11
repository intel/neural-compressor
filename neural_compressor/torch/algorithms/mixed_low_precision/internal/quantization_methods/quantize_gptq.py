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

# code ported from https://github.com/AutoGPTQ/AutoGPTQ
# PT_HPU_LAZY_MODE=2 python internal/quantization_methods/quantize_gptq.py

import argparse
import logging
import random
import time

import auto_gptq
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.gpu_migration
import numpy as np
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_compressor.torch.algorithms.mixed_low_precision.custom_methods.gptq import BaseGaudiGPTQForCausalLM
from neural_compressor.torch.algorithms.mixed_low_precision.custom_methods.quarot import rotate

# Over-ride default AutoGPTQ quantization method to Gaudi friendly method
auto_gptq.modeling._base.BaseGPTQForCausalLM.quantize = BaseGaudiGPTQForCausalLM.quantize

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Run GPTQ on Gaudi",
)
parser.add_argument("--pretrained_model", type=str, help="HF pretrained model", default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--quantized_model_dir", type=str, help="output quantized model dir", default="llama-2-7b-4bit")
parser.add_argument("--rotate_weights", action="store_true", help="Whether to use QuaRot for weights only rotation.")
parser.add_argument("--rotate_mlp", action="store_true", help="Whether to use QuaRot for weights+mlp rotation.")
parser.add_argument("--rotate_values", action="store_true", help="Whether to use QuaRot for weights+values rotation.")
args = parser.parse_args()


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = args.pretrained_model
quantized_model_dir = args.quantized_model_dir


def get_data(nsamples, seed, seqlen, model):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset


try:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=False)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
traindataset = get_data(128, 0, 4096, pretrained_model_dir)  # seqlen prev = 2048

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # desc_act and group size only works on triton,
    model_file_base_name="model",  # added so model can be loaded using HF AutoModel which requires model.safetensors
)

# load un-quantized model, the model will always be force loaded into cpu
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

if args.rotate_weights:
    rotate(model.model, args)
start = time.time()

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
# with value under torch.LongTensor type.
with torch.no_grad():
    model.quantize(traindataset, use_triton=False)

print(f"quantization took {time.time() - start} seconds")

# save quantized model using safetensors
htcore.mark_step()
model.save_quantized(quantized_model_dir, use_safetensors=True)
tokenizer.save_pretrained(quantized_model_dir)  # save tokenizer to quantized model dir in order to load it later


model = AutoModelForCausalLM.from_pretrained(quantized_model_dir)
