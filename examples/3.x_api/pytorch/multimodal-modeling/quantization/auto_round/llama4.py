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
import sys
import argparse
import json

import torch
import transformers

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
from neural_compressor.transformers import AutoModelForCausalLM, AutoRoundConfig


@torch.no_grad()
def run_fn(model, dataloader, **kargs):
    for data in dataloader:
        if isinstance(data, tuple) or isinstance(data, list):
            model(*data)
        elif isinstance(data, dict):
            model(**data)
        else:
            model(data)


class BasicArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--model", "--model_name", "--model_name_or_path",
                          default="Qwen/Qwen2-VL-2B-Instruct",
                          help="model name or path")

        self.add_argument('--eval', action='store_true',
                          help="whether to use eval only mode.")

       
        self.add_argument("--quantize", action="store_true")

        self.add_argument("--eval_bs", default=None, type=int,
                          help="batch size in evaluation")

        self.add_argument("--device", "--devices", default="auto", type=str,
                          help="the device to be used for tuning. The default is set to auto,"
                               "allowing for automatic detection."
                               "Currently, device settings support CPU, GPU, and HPU.")

        self.add_argument("--dataset", type=str, default=None,
                          help="the dataset for quantization training."
                               " current support NeelNanda/pile-10k,llava_conv_58k,llava_instruct_80k "
                               "It can be a custom one. Default is NeelNanda/pile-10k")

        self.add_argument("--export_format", default="itrex", type=str,
                          help="the format to save the model"
                          )

        self.add_argument("--output_dir", default="./tmp_autoround", type=str,
                          help="the directory to save quantized model")

        self.add_argument("--model_dtype", default=None, type=str, choices=["fp16", "float16",
                                                                            "bf16", "bfloat16", "fp32", "float32"],
                          help="force to convert the dtype, some backends supports fp16 dtype better")

        self.add_argument("--fp_layers", default="", type=str,
                          help="layers to maintain original data type")

        self.add_argument("--enable_torch_compile", default=None, type=bool,
                          help="whether to enable torch compile")


def setup_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int,
                        help="train batch size")

    parser.add_argument("--iters", "--iter", default=200, type=int,
                        help=" iters")

    parser.add_argument("--nsamples", default=128, type=int,
                        help="number of samples")

    args = parser.parse_args()
    return args


def tune(args):
    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    print(f"start to quantize {model_name}")

    devices = args.device.replace(" ", "").split(',')
    use_auto_mapping = False

    qconfig = AutoRoundConfig(
        scheme=args.scheme,
        iters=args.iters,
        layer_config=layer_config,
        format="llm_compressor",
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True, attn_implementation='eager')
        
    model.eval()
    kwargs = {}
    kwargs['safe_serialization'] = 'False'  # for phi3 saving model
    model.save_pretrained(args.output_dir, safe_serialization=False)

if __name__ == '__main__':
    if "--quantize" in sys.argv:
        args = setup_parser()
        tune(args)
    elif "--inference" in sys.argv:
        sys.argv.remove("--inference")
        from transformers import AutoProcessor
        import requests
        from PIL import Image
        
        args = setup_parser()
        model_name = args.model
        if model_name[-1] == "/":
            model_name = model_name[:-1]
        
        # Preparation for inference    
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir, 
            trust_remote_code=True,
            attn_implementation='eager',
            torch_dtype=torch.float16,
        )
        processor = AutoProcessor.from_pretrained(model_name,  trust_remote_code=True)
        image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

        content = "Describe this image."

        messages = [
            {"role": "user", 
            "content": "<|image_1|>\n"+content},
        ]
        
        prompt = processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(prompt, image_inputs, return_tensors="pt").to(model.device)
        generation_args = { 
            "max_new_tokens": 50, 
            "temperature": 0.0, 
            "do_sample": False, 
        }
        
        generate_ids = model.generate(**inputs, 
                eos_token_id=processor.tokenizer.eos_token_id, 
                **generation_args
        )
        
        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(response)
        
