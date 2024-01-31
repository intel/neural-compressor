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

import argparse

import torch
from config import HQQModuleConfig, QTensorConfig
from quantizer import HQQuantizer, get_model_info


def get_hqq_module_config(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=False,
    scale_quant_group_size=128,
) -> HQQModuleConfig:
    # def hqq_base_quant_config(nbits=4, group_size=64, quant_zero=True, quant_scale=False, scale_quant_group_size=128):
    #    assert nbits in Quantizer.SUPPORTED_BITS, "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
    #    if(group_size is not None):
    #       assert is_divisible(group_size, 8), "Invalid group_size param: the value should be a multiple of 8."
    #    weight_quant_params = {'nbits':nbits,'channel_wise':True,  'group_size':group_size, 'optimize':True, 'round_zero':True if nbits==4 else False}
    #    scale_quant_params  = {'nbits':8,    'channel_wise':True,  'group_size':scale_quant_group_size,        'optimize':False} if (quant_scale) else None
    #    zero_quant_params   = {'nbits':8,    'channel_wise':False, 'group_size':None,       'optimize':False} if (quant_zero)  else None
    #    return {'weight_quant_params':weight_quant_params, 'scale_quant_params':scale_quant_params, 'zero_quant_params':zero_quant_params}

    weight_qconfig = QTensorConfig(
        nbits=nbits, channel_wise=True, group_size=group_size, optimize=True, round_zero=True if nbits == 4 else False
    )
    zero_qconfig = None
    if quant_zero:
        zero_qconfig = QTensorConfig(nbits=8, channel_wise=False, group_size=None, optimize=False)
    scale_qconfig = None
    if quant_scale:
        scale_qconfig = QTensorConfig(nbits=8, channel_wise=True, group_size=scale_quant_group_size, optimize=False)
    hqq_module_config = HQQModuleConfig(weight=weight_qconfig, scale=scale_qconfig, zero=zero_qconfig)
    print(hqq_module_config)
    return hqq_module_config


def get_hqq_qconfig_mapping(
    model, nbits=4, group_size=64, quant_zero=True, quant_scale=False, scale_quant_group_size=128, quant_lm_head=False
):
    model_info = get_model_info(model)
    qconfig_mapping = {}
    for name, _ in model_info:
        if "lm_head" in name and not quant_lm_head:
            continue
        qconfig_mapping[name] = get_hqq_module_config(
            nbits, group_size, quant_zero, quant_scale, scale_quant_group_size
        )
    return qconfig_mapping


def get_model_and_tokenizer(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def quant_hqq(args):
    model, tokenizer = get_model_and_tokenizer(args.model_id)
    qconfig_mapping = get_hqq_qconfig_mapping(
        model,
        args.nbits,
        args.group_size,
        args.quant_zero,
        args.quant_scale,
        args.scale_quant_group_size,
        args.quant_lm_head,
    )

    hqq_quantizer = HQQuantizer(qconfig_mapping)
    q_model = hqq_quantizer.prepare(model)
    if args.eval:
        from eval_wiki2 import eval_wikitext2

        eval_wikitext2(q_model, tokenizer, verbose=True)
    return q_model


def main():
    parser = argparse.ArgumentParser(description="HQQ quantization")
    parser.add_argument("--model_id", type=str, default="facebook/opt-125m", help="Model id")
    parser.add_argument("--nbits", type=int, default=4, help="Number of bits")
    parser.add_argument("--group_size", type=int, default=128, help="Group size")
    parser.add_argument("--quant_zero", action="store_true", default=False, help="Quantize zero")
    parser.add_argument("--quant_scale", action="store_true", default=False, help="Quantize scale")
    parser.add_argument("--scale_quant_group_size", type=int, default=128, help="Scale quant group size")
    parser.add_argument("--quant_lm_head", action="store_true", default=False, help="Quant lm head")
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the model")
    args = parser.parse_args()
    # Call your quant_hqq function here
    q_model = quant_hqq(args)


if __name__ == "__main__":
    main()

# python quant_model.py --model_id /models/opt-125m --nbits 4 --group_size 128 --quant_zero --quant_scale --scale_quant_group_size 64 --eval
# python quant_model.py --model_id /models/Llama-2-7b-hf --nbits 4 --group_size 128 --quant_zero --quant_scale --scale_quant_group_size 64 --eval


# python quant_model.py --model_id /mnt/disk4/modelHub/opt-125m --nbits 4 --group_size 128 --quant_zero --quant_scale --scale_quant_group_size 64
