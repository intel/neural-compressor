# Copyright (c) 2026 Intel Corporation
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
import os
from functools import partial

import pandas as pd
import tabulate
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

from auto_round.compressors.diffusion.dataset import get_diffusion_dataloader
from auto_round.compressors.diffusion.eval import metric_map
from auto_round.data_type.mxfp import quant_mx_rceil
from auto_round.utils import get_block_names, get_module
from neural_compressor.torch.quantization import AutoRoundConfig, convert, prepare


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize and evaluate Stable Diffusion XL with MXFP8.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "--model_name_or_path",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL model name or local path",
    )
    parser.add_argument("--scheme", default="MXFP8", choices=["BF16", "MXFP8"])
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument(
        "--disable_opt_rtn",
        action="store_true",
        help="use calibration-free pure RTN instead of optimized RTN",
    )
    parser.add_argument(
        "--output_dir",
        "--quantized_model_path",
        default="./saved_results",
        help="directory used to save or load the quantized pipeline",
    )
    parser.add_argument("--eval_dataset", default="captions_source.tsv", help="evaluation TSV file")
    parser.add_argument("--output_image_path", default="./tmp_imgs", help="generated image directory")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="denoising steps during evaluation")
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--limit", default=-1, type=int, help="maximum number of evaluation prompts")
    return parser.parse_args()


def quantize(args, device):
    if args.scheme != "MXFP8":
        raise ValueError("Only MXFP8 requires quantization; use BF16 directly for the baseline.")
    if not args.disable_opt_rtn:
        raise ValueError("This MXFP8 example uses pure RTN; specify --disable_opt_rtn.")

    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    quant_config = AutoRoundConfig(
        scheme="MXFP8",
        iters=0,
        disable_opt_rtn=args.disable_opt_rtn,
        export_format="fake",
        output_dir=args.output_dir,
    )
    prepared_unet = prepare(pipe.unet, quant_config)
    convert(prepared_unet, quant_config, pipeline=pipe)


def enable_mxfp8_activation_qdq(unet):
    def act_qdq_forward(module, inputs, *forward_args, **forward_kwargs):
        qdq_inputs, _, _ = quant_mx_rceil(inputs, bits=8, group_size=32, data_type="mx_fp_rceil")
        return module.orig_forward(qdq_inputs, *forward_args, **forward_kwargs)

    for block_names in get_block_names(unet):
        for block_name in block_names:
            block = get_module(unet, block_name)
            for module in block.modules():
                if module.__class__.__name__ == "Linear" and not hasattr(module, "orig_forward"):
                    module.orig_forward = module.forward
                    module.forward = partial(act_qdq_forward, module)


def load_pipeline(args, device):
    if args.scheme == "BF16":
        print("Loading the BF16 baseline pipeline.")
        return StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)

    if not os.path.isdir(args.output_dir):
        raise ValueError(f"Quantized model directory does not exist: {args.output_dir}")

    if os.path.isfile(os.path.join(args.output_dir, "model_index.json")):
        print(f"Loading the quantized pipeline from {args.output_dir}.")
        pipe = StableDiffusionXLPipeline.from_pretrained(args.output_dir, torch_dtype=torch.bfloat16)
    else:
        print(f"Loading the quantized UNet from {args.output_dir}.")
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        pipe.unet = UNet2DConditionModel.from_pretrained(args.output_dir, torch_dtype=torch.bfloat16)

    enable_mxfp8_activation_qdq(pipe.unet)
    return pipe.to(device)


def generate_images(args, pipe, device):
    os.makedirs(args.output_image_path, exist_ok=True)
    dataloader, _ = get_diffusion_dataloader(args.eval_dataset, nsamples=args.limit, bs=1, seed=args.seed)

    for image_ids, prompts in dataloader:
        image_id = image_ids[0].item()
        image_path = os.path.join(args.output_image_path, f"{image_id}.png")
        if os.path.exists(image_path):
            continue

        generator = torch.Generator(device=device).manual_seed(args.seed + int(image_id))
        output = pipe(
            prompt=[prompts[0]],
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        output.images[0].save(image_path)


def evaluate_accuracy(args, device):
    dataframe = pd.read_csv(args.eval_dataset, sep="\t")
    if not {"id", "caption"}.issubset(dataframe.columns):
        raise ValueError("The evaluation TSV must contain 'id' and 'caption' columns.")

    if args.limit > 0:
        dataframe = dataframe.iloc[: args.limit]

    prompts = []
    images = []
    for row in dataframe.itertuples(index=False):
        image_path = os.path.join(args.output_image_path, f"{row.id}.png")
        if os.path.exists(image_path):
            prompts.append(row.caption)
            images.append(image_path)

    if not images:
        raise ValueError(f"No generated images found in {args.output_image_path}.")

    results = {}
    for metric in ("clip", "clip-iqa", "imagereward"):
        results.update(metric_map[metric](prompts, images, device))
    print(tabulate.tabulate(results.items(), headers=("Metric", "Score"), tablefmt="grid"))


def main():
    args = parse_args()
    if not any((args.quantize, args.inference, args.accuracy)):
        raise ValueError("Specify at least one of --quantize, --inference, or --accuracy.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quantize:
        print(f"Quantizing {args.model} to MXFP8.")
        quantize(args, device)
    if args.inference:
        generate_images(args, load_pipeline(args, device), device)
    if args.accuracy:
        evaluate_accuracy(args, device)


if __name__ == "__main__":
    main()
