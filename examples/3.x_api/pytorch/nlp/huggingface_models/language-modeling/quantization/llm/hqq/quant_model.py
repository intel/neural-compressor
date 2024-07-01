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



def get_model_and_tokenizer(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def quant_hqq(args):
    model, tokenizer = get_model_and_tokenizer(args.model_id)
    from neural_compressor.torch.quantization import HQQConfig, quantize

    hqq_config = HQQConfig(
        bits=args.bits,
        group_size=args.group_size,
        quant_zero=args.quant_zero,
        quant_scale=args.quant_scale,
        scale_quant_group_size=args.scale_quant_group_size,
    )

    q_model = quantize(model=model, quant_config=hqq_config)
    if args.eval:
        from eval_wiki2 import eval_wikitext2
        eval_wikitext2(q_model, tokenizer, verbose=True)
    return q_model


def main():
    parser = argparse.ArgumentParser(description="HQQ quantization")
    parser.add_argument("--model_id", type=str, default="facebook/opt-125m", help="Model id")
    parser.add_argument("--bits", type=int, default=4, help="Number of bits")
    parser.add_argument("--group_size", type=int, default=64, help="Group size")
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

# python quant_model.py --model_id /models/opt-125m --bits 4 --group_size 64 --quant_zero --quant_scale --scale_quant_group_size 128 --eval
# python quant_model.py --model_id /models/Llama-2-7b-hf --bits 4 --group_size 64 --quant_zero --quant_scale --scale_quant_group_size 128 --eval
# python quant_model.py --model_id /mnt/disk4/modelHub/opt-125m --bits 4 --group_size 64 --quant_zero --quant_scale --scale_quant_group_size 128  --eval
