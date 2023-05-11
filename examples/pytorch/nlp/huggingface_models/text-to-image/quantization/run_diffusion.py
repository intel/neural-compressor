#!/usr/bin/env python
# coding=utf-8
#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

""" Example for stable-diffusion to generate a picture from a text ."""
# You can also adapt this script on your own text to image task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import sys
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
from pytorch_fid import fid_score


os.environ["CUDA_VISIBLE_DEVICES"] = ""


logger = logging.getLogger(__name__)


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Example of a post-training quantization script.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="a drawing of a gray and black dragon",
        help="The input of the model, like: 'a photo of an astronaut riding a horse on mars'.",
    )
    parser.add_argument(
        "--calib_text",
        type=str,
        default="Womens Princess Little Deer Native American Costume",
        help="The calibration data of the model, like: 'Womens Princess Little Deer Native American Costume'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_results",
        help="The path to save model and quantization configures.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="The number of images to generate per prompt, defaults to 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="random seed",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed: local_rank",
    )
    parser.add_argument(
        "--base_images",
        type=str,
        default="base_images",
        help="Path to training images for FID input.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Whether or not to apply quantization.",
    )
    parser.add_argument(
        "--quantization_approach",
        type=str,
        default="static",
        help="Quantization approach. Supported approach are static, "
                  "dynamic and auto.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="pytorch",
        help="Deep learning framework. Supported framework are pytorch, ipex",
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="eval_f1",
        help="Metric used for the tuning strategy.",
    )
    parser.add_argument(
        "--is_relative",
        type=bool,
        default="True",
        help="Metric tolerance mode, True for relative, otherwise for absolute.",
    )
    parser.add_argument(
        "--perf_tol",
        type=float,
        default=0.01,
        help="Performance tolerance when optimizing the model.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Only test performance for model.",
    )
    parser.add_argument(
        "--accuracy_only",
        action="store_true",
        help="Only test accuracy for model.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="benchmark for int8 model",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def image_grid(imgs, rows, cols):
    if not len(imgs) == rows * cols:
        raise ValueError("The specified number of rows and columns are not correct.")

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def benchmark(pipe, generator):
    warmup = 2
    total = 5
    total_time = 0
    with torch.no_grad():
        for i in range(total):
            prompt = "a photo of an astronaut riding a horse on mars"
            start2 = time.time()
            images = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images
            end2 = time.time()
            if i >= warmup:
                total_time += end2 - start2
    print("Average Latency: ", (total_time) / (total - warmup), "s")
    print("Average Throughput: {:.5f} samples/sec".format((total - warmup) / (total_time)))


def accuracy(pipe, generator, rows, args):
    with torch.no_grad():
        new_images = pipe(
            args.input_text,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
            num_images_per_prompt=args.num_images_per_prompt,
        ).images
        tmp_save_images = "tmp_save_images"
        os.makedirs(tmp_save_images, exist_ok=True)
        if os.path.isfile(os.path.join(tmp_save_images, "image.png")):
            os.remove(os.path.join(tmp_save_images, "image.png"))
        grid = image_grid(new_images, rows=rows, cols=args.num_images_per_prompt // rows)
        grid.save(os.path.join(tmp_save_images, "image.png"))
        fid = fid_score.calculate_fid_given_paths((args.base_images, tmp_save_images), 1, "cpu", 2048, 8)
        print("Finally FID score Accuracy: {}".format(fid))
        return fid

class CalibDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        data = "a photo of an astronaut riding a horse on mars"
        return data


def main():
    # Passing the --help flag to this script.

    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Parameters {args}")

    # Set seed before initializing model.
    set_seed(args.seed)

    # Load pretrained model and generate a pipeline
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    pipe = StableDiffusionPipeline.from_pretrained(args.model_name_or_path)
    _rows = int(math.sqrt(args.num_images_per_prompt))

    if args.tune:
        tmp_fp32_images = "tmp_fp32_images"
        tmp_int8_images = "tmp_int8_images"
        os.makedirs(tmp_fp32_images, exist_ok=True)
        os.makedirs(tmp_int8_images, exist_ok=True)
        generator = torch.Generator("cpu").manual_seed(args.seed)
        fp32_images = pipe(
            args.input_text,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
            num_images_per_prompt=args.num_images_per_prompt,
        ).images
        grid = image_grid(fp32_images, rows=_rows, cols=args.num_images_per_prompt // _rows)
        grid.save(os.path.join(tmp_fp32_images, "fp32.png"))

        attr_list = ["unet"]
        for name in attr_list:
            model = getattr(pipe, name)

            def calibration_func(model):
                calib_num = 5
                setattr(pipe, name, model)
                with torch.no_grad():
                    for i in range(calib_num):
                        pipe(
                            args.calib_text,
                            guidance_scale=7.5,
                            num_inference_steps=50,
                            generator=generator,
                            num_images_per_prompt=args.num_images_per_prompt,
                        )

            def eval_func(model):
                setattr(pipe, name, model)
                generator = torch.Generator("cpu").manual_seed(args.seed)
                with torch.no_grad():
                    new_images = pipe(
                        args.input_text,
                        guidance_scale=7.5,
                        num_inference_steps=50,
                        generator=generator,
                        num_images_per_prompt=args.num_images_per_prompt,
                    ).images
                    if os.path.isfile(os.path.join(tmp_int8_images, "int8.png")):
                        os.remove(os.path.join(tmp_int8_images, "int8.png"))
                    grid = image_grid(new_images, rows=_rows, cols=args.num_images_per_prompt // _rows)
                    grid.save(os.path.join(tmp_int8_images, "int8.png"))
                    fid = fid_score.calculate_fid_given_paths((args.base_images, tmp_int8_images), 1, "cpu", 2048, 8)
                    return fid

            from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion
            from neural_compressor.quantization import fit
            accuracy_criterion = AccuracyCriterion(
                higher_is_better=False,
                criterion="relative" if args.is_relative else "absolute",
                tolerable_loss=args.perf_tol)
            quantization_config = PostTrainingQuantConfig(
                approach=args.quantization_approach,
                accuracy_criterion=accuracy_criterion
            )
            model = fit(
                model=pipe.unet,
                conf=quantization_config,
                eval_func=eval_func,
                calib_func=calibration_func,
            )
            setattr(pipe, name, model)
            model.save(args.output_dir)
            logger.info(f"Optimized model {name} saved to: {args.output_dir}.")

    if args.benchmark or args.accuracy_only:

        def b_func(model):
            setattr(pipe, "unet", model)
            benchmark(pipe, generator)

        if args.int8:
            print("====int8 inference====")
            from neural_compressor.utils.pytorch import load
            checkpoint = os.path.join(args.output_dir)
            model = load(checkpoint, model=getattr(pipe, "unet"))
            model.eval()
        else:
            print("====fp32 inference====")
            model = getattr(pipe, "unet")
        generator = torch.Generator("cpu").manual_seed(args.seed)
        if args.benchmark:
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig

            b_conf = BenchmarkConfig(cores_per_instance=4, num_of_instance=1)
            fit(model, conf=b_conf, b_func=b_func)
        if args.accuracy_only:
            setattr(pipe, "unet", model)
            accuracy(pipe, generator, _rows, args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
