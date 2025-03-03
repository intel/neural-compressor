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

        self.add_argument("--bits", default=4, type=int,
                          help="weight bits")
        
        self.add_argument("--quantize", action="store_true")

        self.add_argument("--eval_bs", default=None, type=int,
                          help="batch size in evaluation")

        self.add_argument("--device", "--devices", default="auto", type=str,
                          help="the device to be used for tuning. The default is set to auto,"
                               "allowing for automatic detection."
                               "Currently, device settings support CPU, GPU, and HPU.")

        self.add_argument("--asym", action='store_true',
                          help="whether to use asym quantization")

        self.add_argument("--dataset", type=str, default=None,
                          help="the dataset for quantization training."
                               " current support NeelNanda/pile-10k,llava_conv_58k,llava_instruct_80k "
                               "It can be a custom one. Default is NeelNanda/pile-10k")

        self.add_argument("--lr", default=None, type=float,
                          help="learning rate, if None, it will be set to 1.0/iters automatically")

        self.add_argument("--minmax_lr", default=None, type=float,
                          help="minmax learning rate, if None,it will beset to be the same with lr")

        self.add_argument("--seed", default=42, type=int,
                          help="random seed")

        self.add_argument("--adam", action='store_true',
                          help="whether to use adam optimizer instead of SignSGD")

        self.add_argument("--gradient_accumulate_steps", default=1, type=int,
                          help="gradient accumulate steps")

        self.add_argument("--nblocks", default=1, type=int,
                          help="how many blocks to tune together")

        self.add_argument("--low_gpu_mem_usage", action='store_true',
                          help="offload intermediate features to cpu")

        self.add_argument("--export_format", default="itrex", type=str,
                          help="the format to save the model"
                          )

        self.add_argument("--data_type", "--dtype", default='int',
                          help="data type for tuning, 'int', 'mx_fp' and etc")

        self.add_argument("--scale_dtype", default='fp16', choices=["fp16", "float16",
                                                                    "bf16", "bfloat16", "fp32", "float32"],
                          help="scale data type to use for quantization")

        self.add_argument("--output_dir", default="./tmp_autoround", type=str,
                          help="the directory to save quantized model")

        self.add_argument("--disable_amp", action='store_true',
                          help="disable amp")

        self.add_argument("--disable_minmax_tuning", action='store_true',
                          help="whether disable enable weight minmax tuning")

        self.add_argument("--enable_norm_bias_tuning", action='store_true',
                          help="whether enable norm bias tuning")

        self.add_argument("--disable_trust_remote_code", action='store_true',
                          help="whether to disable trust_remote_code")

        self.add_argument("--disable_quanted_input", action='store_true',
                          help="whether to disuse the output of quantized block to tune the next block")

        self.add_argument("--quant_lm_head", action='store_true',
                          help="whether to quant lm_head")

        self.add_argument("--low_cpu_mem_mode", default=0, type=int, choices=[0, 1, 2],
                          help="choose which low cpu memory mode to use. "
                               "Can significantly reduce cpu memory footprint but cost more time."
                               "1 means choose block-wise mode, load the weights of each block"
                               " from disk when tuning and release the memory of the block after tuning."
                               "2 means choose layer-wise mode, load the weights of each layer from disk when tuning,"
                               " minimum memory consumption and also slowest running speed."
                               "others means not use low cpu memory. Default to 0, not use low cpu memory.")

        self.add_argument("--low_cpu_mem_tmp_dir", default=None, type=str,
                          help="temporary work space to store the temporary files "
                               "when using low cpu memory mode. Will remove after tuning.")

        self.add_argument("--model_dtype", default=None, type=str, choices=["fp16", "float16",
                                                                            "bf16", "bfloat16", "fp32", "float32"],
                          help="force to convert the dtype, some backends supports fp16 dtype better")

        self.add_argument("--act_bits", default=32, type=int,
                          help="activation bits")

        self.add_argument("--fp_layers", default="", type=str,
                          help="layers to maintain original data type")

        self.add_argument("--not_use_best_mse", action='store_true',
                          help="whether to use the iter of best mes loss in the tuning phase")

        self.add_argument("--enable_torch_compile", default=None, type=bool,
                          help="whether to enable torch compile")

        ## ======================= VLM =======================
        self.add_argument("--quant_nontext_module", action='store_true',
                          help="whether to quantize non-text module, e.g. vision component")

        self.add_argument("--extra_data_dir", default=None, type=str,
                          help="dataset dir for storing images/audio/videos. "
                               "Can be a dir path or multiple dir path with format as "
                               "'image=path_to_image,video=path_to_video,audio=path_to_audio'"
                               "By default, it will search in the relative path, "
                               "and if not find, will automatic download.")

        self.add_argument("--template", default=None, type=str,
                          help="the template for building training dataset. It can be a custom one.")

        self.add_argument("--truncation", action="store_true",
                          help="whether to truncate sequences at the maximum length."
                               " Default True for pile and False for llava dataset.")

        self.add_argument("--to_quant_block_names", default=None, type=str,
                          help="Names of quantitative blocks, please use commas to separate them.")


def setup_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--batch_size", "--train_bs", "--bs", default=8, type=int,
                        help="train batch size")

    parser.add_argument("--iters", "--iter", default=200, type=int,
                        help=" iters")

    parser.add_argument("--seqlen", "--seq_len", default=None, type=int,
                        help="sequence length, default 2048 for text-only, 512 for liuhaotian/llava")

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

    if all(s.isdigit() for s in devices):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            current_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            current_visible_devices = current_visible_devices.split(',')
            indices = [int(device) for device in devices]
            try:
                pick_device = [current_visible_devices[i] for i in indices]
            except:
                raise ValueError(
                    "Invalid '--device' value: It must be smaller than the number of available devices. "
                    "For example, with CUDA_VISIBLE_DEVICES=4,5, "
                    "--device 0,1 is valid, but --device 4,5 is not supported.")
            visible_devices = ','.join(pick_device)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            args.device = ",".join(map(str, range(len(devices))))
            devices = args.device.replace(" ", "").split(',')
        use_auto_mapping = True

    woq_config = AutoRoundConfig(
        is_vlm=True,
        bits=args.bits,
        sym=not args.asym,
        group_size=args.group_size,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        iters=args.iters,
        seqlen=args.seqlen,
        quant_nontext_module=args.quant_nontext_module,
        truncation=args.truncation,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        nblocks=args.nblocks,
        lr=args.lr,
        minmax_lr=args.minmax_lr,
        enable_quanted_input=not args.disable_quanted_input,
        scale_dtype=args.scale_dtype,
        enable_minmax_tuning=not args.disable_minmax_tuning,
        act_bits=args.act_bits,
        export_format=args.export_format
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
        
