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

import pandas as pd
import tabulate
import torch

from diffusers import AutoPipelineForText2Image
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    convert,
    prepare,
)
import multiprocessing as mp

from auto_round.compressors.diffusion.eval import metric_map
from auto_round.compressors.diffusion.dataset import get_diffusion_dataloader
from torch.multiprocessing import Process, Queue


def inference_worker(device, eval_file, pipe, image_save_dir, queue):
    if device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        torch.cuda.set_device(device)

    gen_kwargs = {
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "generator": None,
    }
 
    dataloader, _, _ = get_diffusion_dataloader(eval_file, nsamples=-1, bs=1)
    prompt_list = []
    image_list = []
    for image_ids, prompts in dataloader:
        prompt_list.extend(prompts)

        new_ids = []
        new_prompts = []
        for idx, image_id in enumerate(image_ids):
            image_id = image_id.item()
            image_list.append(os.path.join(image_save_dir, str(image_id) + ".png"))

            if os.path.exists(os.path.join(image_save_dir, str(image_id) + ".png")):
                continue
            new_ids.append(image_id)
            new_prompts.append(prompts[idx])

        if len(new_prompts) == 0:
            continue

        output = pipe(prompt=new_prompts, **gen_kwargs)
        for idx, image_id in enumerate(new_ids):
            output.images[idx].save(os.path.join(image_save_dir, str(image_id) + ".png"))

    queue.put((prompt_list, image_list))

class BasicArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--model", "--model_name", "--model_name_or_path",
                          help="model name or path")

        self.add_argument('--scheme', default="MXFP8", type=str,
                          help="quantizaion scheme.")

        self.add_argument("--quantize", action="store_true")

        self.add_argument("--inference", action="store_true")

        self.add_argument("--dataset", type=str, default="coco2014",
                          help="the dataset for quantization training.")

        self.add_argument("--output_dir", default="./tmp_autoround", type=str,
                          help="the directory to save quantized model")

        self.add_argument("--eval_dataset", default="captions_source.tsv", type=str,
                          help="eval datasets")

        self.add_argument("--output_image_path", default="./tmp_imgs", type=str,
                          help="the directory to save quantized model")


def setup_parser():
    parser = BasicArgumentParser()

    parser.add_argument("--iters", "--iter", default=1000, type=int,
                        help="tuning iters")

    args = parser.parse_args()
    return args


def tune(args, pipe):
    model = pipe.transformer
    layer_config = {}
    kwargs = {}
    if args.scheme == "FP8":
        for n, m in model.named_modules():
            if m.__class__.__name__ == "Linear":
                layer_config[n] = {"bits": 8, "act_bits": 8, "data_type": "fp", "act_data_type": "fp", "group_size": 0, "act_group_size": 0}
    elif args.scheme == "MXFP8":
        kwargs["scheme"] = "MXFP8"

    qconfig = AutoRoundConfig(
        iters=args.iters,
        dataset=args.dataset,
        layer_config=layer_config,
        num_inference_steps=3,
        export_format="fake",
        nsamples=128,
        batch_size=1,
        **kwargs
    )
    model = prepare(model, qconfig)
    model = convert(model, qconfig, pipeline=pipe)
    delattr(model, "save")
    return pipe

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = setup_parser()
    model_name = args.model
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    if "--quantize" in sys.argv:
        print(f"start to quantize {model_name}")
        pipe = tune(args, pipe)
    if "--inference" in sys.argv:
        if not os.path.exists(args.output_image_path):
            os.makedirs(args.output_image_path)

        visible_gpus = torch.cuda.device_count()

        if visible_gpus == 0:
            prompt_list, image_list = inference_worker("cpu", args.eval_dataset, pipe, args.output_image_path)

        else:
            df = pd.read_csv(args.eval_dataset, sep='\t')
            subsut_sample_num = len(df) // visible_gpus
            for i in range(visible_gpus):
                start = i * subsut_sample_num
                end = min((i + 1) * subsut_sample_num, len(df))
                df_subset = df.iloc[start : end]
                df_subset.to_csv(f"subset_{i}.tsv", sep='\t', index=False)

            processes = []
            queue = Queue()
            for i in range(visible_gpus):
                p = Process(target=inference_worker, args=(i, f"subset_{i}.tsv", pipe.to(f"cuda:{i}"), args.output_image_path, queue))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            outputs = [queue.get() for _ in range(visible_gpus)]

            prompt_list = []
            image_list = []
            for output in outputs:
                prompt_list.extend(output[0])
                image_list.extend(output[1])

            print("Evaluations for subset are done! Getting the final accuracy...")

        result = {}
        metrics = ["clip", "clip-iqa", "imagereward"]
        for metric in metrics:
            result.update(metric_map[metric](prompt_list, image_list, pipe.device))

        print(tabulate.tabulate(result.items(), tablefmt="grid"))
