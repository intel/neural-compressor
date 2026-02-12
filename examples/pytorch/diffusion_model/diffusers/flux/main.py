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

import json
import os
import sys
import argparse

import pandas as pd
import tabulate
import torch

from diffusers import AutoPipelineForText2Image, FluxTransformer2DModel
from functools import partial
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    convert,
    prepare,
)
from auto_round.data_type.mxfp import quant_mx_rceil
from auto_round.data_type.fp8 import quant_fp8_sym
from auto_round.utils import get_block_names, get_module
from auto_round.compressors.diffusion.eval import metric_map
from auto_round.compressors.diffusion.dataset import get_diffusion_dataloader


parser = argparse.ArgumentParser(
    description="Flux quantization.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--model", "--model_name", "--model_name_or_path", help="model name or path")
parser.add_argument('--scheme', default="MXFP8", type=str, help="quantizaion scheme.")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--inference", action="store_true")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--dataset", type=str, default="coco2014", help="the dataset for quantization training.")
parser.add_argument("--output_dir", "--quantized_model_path", default="./tmp_autoround", type=str, help="the directory to save quantized model")
parser.add_argument("--eval_dataset", default="captions_source.tsv", type=str, help="eval datasets")
parser.add_argument("--output_image_path", default="./tmp_imgs", type=str, help="the directory to save quantized model")
parser.add_argument("--iters", "--iter", default=1000, type=int, help="tuning iters")
parser.add_argument("--limit", default=-1, type=int, help="limit the number of prompts for evaluation")

args = parser.parse_args()


def inference_worker(eval_file, pipe, image_save_dir):
    gen_kwargs = {
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "generator": None,
    }
 
    dataloader, _, _ = get_diffusion_dataloader(eval_file, nsamples=args.limit, bs=1)
    for image_ids, prompts in dataloader:

        new_ids = []
        new_prompts = []
        for idx, image_id in enumerate(image_ids):
            image_id = image_id.item()

            if os.path.exists(os.path.join(image_save_dir, str(image_id) + ".png")):
                continue
            new_ids.append(image_id)
            new_prompts.append(prompts[idx])

        if len(new_prompts) == 0:
            continue

        output = pipe(prompt=new_prompts, **gen_kwargs)
        for idx, image_id in enumerate(new_ids):
            output.images[idx].save(os.path.join(image_save_dir, str(image_id) + ".png"))


def tune(device):
    pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model = pipe.transformer
    layer_config = {}
    kwargs = {}
    if args.scheme == "FP8":
        for n, m in model.named_modules():
            if m.__class__.__name__ == "Linear":
                layer_config[n] = {"bits": 8, "data_type": "fp", "group_size": 0}
    elif args.scheme == "MXFP8":
        kwargs["scheme"] = {
            "bits": 8,
            "group_size": 32,
            "data_type": "mx_fp",
        }

    qconfig = AutoRoundConfig(
        iters=args.iters,
        dataset=args.dataset,
        layer_config=layer_config,
        num_inference_steps=3,
        export_format="fake",
        nsamples=128,
        batch_size=1,
        output_dir=args.output_dir,
        **kwargs
    )
    model = prepare(model, qconfig)
    model = convert(model, qconfig, pipeline=pipe)

if __name__ == '__main__':
    device = "cpu" if torch.cuda.device_count() == 0 else "cuda"

    if args.quantize:
        print(f"Start to quantize {args.model}.")
        tune(device)
        exit(0)

    if args.inference:
        pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=torch.bfloat16)

        os.makedirs(args.output_image_path, exist_ok=True)

        if os.path.exists(args.output_dir) and os.path.exists(os.path.join(args.output_dir, "diffusion_pytorch_model.safetensors.index.json")):
            print(f"Loading quantized model from {args.output_dir}")
            model = FluxTransformer2DModel.from_pretrained(args.output_dir, torch_dtype=torch.bfloat16)

            # replace Linear's forward function
            if args.scheme == "MXFP8":
                def act_qdq_forward(module, x, *args, **kwargs):
                    qdq_x, _, _ = quant_mx_rceil(x, bits=8, group_size=32, data_type="mx_fp_rceil")
                    return module.orig_forward(qdq_x, *args, **kwargs)

                all_quant_blocks = get_block_names(model)

                for block_names in all_quant_blocks:
                    for block_name in block_names:
                        block = get_module(model, block_name)
                        for n, m in block.named_modules():
                            if m.__class__.__name__ == "Linear":
                                m.orig_forward = m.forward
                                m.forward = partial(act_qdq_forward, m)

            if args.scheme == "FP8":
                def act_qdq_forward(module, x, *args, **kwargs):
                    qdq_x, _, _ = quant_fp8_sym(x, group_size=0)
                    return module.orig_forward(qdq_x, *args, **kwargs)

                for n, m in model.named_modules():
                    if m.__class__.__name__ == "Linear":
                        m.orig_forward = m.forward
                        m.forward = partial(act_qdq_forward, m)

            pipe.transformer = model

        else:
            print("Don't supply quantized_model_path or quantized model doesn't exist, evaluate BF16 accuracy.")

        inference_worker(args.eval_dataset, pipe.to(device), args.output_image_path)

    if args.accuracy:
        df = pd.read_csv(args.eval_dataset, sep="\t")
        prompt_list = []
        image_list = []
        for index, row in df.iterrows():
            assert "id" in row and "caption" in row
            caption_id = row["id"]
            caption_text = row["caption"]
            if os.path.exists(os.path.join(args.output_image_path, str(caption_id) + ".png")):
                prompt_list.append(caption_text)
                image_list.append(os.path.join(args.output_image_path, str(caption_id) + ".png"))

        result = {}
        metrics = ["clip", "clip-iqa", "imagereward"]
        for metric in metrics:
            result.update(metric_map[metric](prompt_list, image_list, device))

        print(tabulate.tabulate(result.items(), tablefmt="grid"))
