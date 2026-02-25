# Copyright (c) 2025 Intel Corporation
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

import os
import shutil

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from neural_compressor.torch.quantization import (
    FP8Config,
    convert,
    load,
    save,
)

torch.manual_seed(1)
torch.set_grad_enabled(False)


def get_model_param_buffers(model):
    tmp = {}
    for name, param in model.named_parameters():
        tmp[name] = param
    for name, buffer in model.named_buffers():
        tmp[name] = buffer
    return tmp


def compare_parameters_buffers(model1, model2):
    dict1 = get_model_param_buffers(model1)
    dict2 = get_model_param_buffers(model2)
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    unique_keys_in_dict1 = keys1 - keys2
    unique_keys_in_dict2 = keys2 - keys1
    unique_keys = unique_keys_in_dict1.union(unique_keys_in_dict2)
    assert len(dict1) == len(dict2), (
        f"The number of parameters and buffers are different, {unique_keys}.\n"
        + f"unique_keys_in_model1: {unique_keys_in_dict1}\nunique_keys_in_model2: {unique_keys_in_dict2}\n"
    )
    for k, v in dict1.items():
        assert k in dict2, "k not in dict2"
        assert v.dtype == dict2[k].dtype, f"dtype of {k} is different.\n{v.dtype}\n{dict2[k].dtype}"

        # torch.allclose operation doesn't support FP8 data type on CPU, so convert to fp32 to compare
        assert torch.allclose(v.to(torch.float), dict2[k].to(torch.float)), (
            f"{k} is different in model1 and model2.\n" + f"{v}\n" + f"{dict2[k]}\n"
        )


def test_save_load():
    config = LlamaConfig(hidden_size=128, num_attention_heads=2, num_hidden_layers=2, vocab_size=512)
    model = LlamaForCausalLM(config).to("cpu").to(torch.bfloat16)
    model = model.eval()

    qconfig = FP8Config(fp8_config="E4M3", scale_method="unit_scale", use_qdq=True)
    qmodel = convert(model, qconfig)

    save(qmodel, "saved_results", format="huggingface")
    new_model = load("saved_results", format="huggingface", device="cpu")

    compare_parameters_buffers(qmodel, new_model)
    shutil.rmtree("saved_results", ignore_errors=True)

    example_input = torch.tensor([[5, 6]]).to("cpu")
    with torch.no_grad():
        out1 = model(example_input)[0].cpu()
        out2 = new_model(example_input)[0].cpu()
    assert (
        out1 == out2
    ).all(), f"The output of the model is different after save and load with scale_method: {scale_method}"
