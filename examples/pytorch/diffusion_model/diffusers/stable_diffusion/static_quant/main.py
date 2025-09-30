#
# -*- coding: utf-8 -*-
#
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
#

import argparse
import logging
import os
import time
import threading
from tqdm import tqdm

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.datasets as dset
import torchvision.transforms as transforms

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1", help="Model path")
    parser.add_argument("--quantized_model_path", type=str, default="quantized_model.pt", help="INT8 model path")
    parser.add_argument("--dataset_path", type=str, default=None, help="COCO2017 dataset path")
    parser.add_argument("--prompt", type=str, default="A big burly grizzly bear is show with grass in the background.", help="input text")
    parser.add_argument("--output_dir", type=str, default=None,help="output path")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--precision', type=str, default="fp32", help='precision: fp32, bf32, bf16, fp16, int8-bf16, int8-fp32')
    parser.add_argument('--calibration', action='store_true', default=False, help='doing calibration step for LCM int8')
    parser.add_argument('--compile_inductor', action='store_true', default=False, help='compile with inductor backend')
    parser.add_argument('--profile', action='store_true', default=False, help='profile')
    parser.add_argument('--benchmark', action='store_true', default=False, help='test performance')
    parser.add_argument('--accuracy', action='store_true', default=False, help='test accuracy')
    parser.add_argument('-w', '--warmup_iterations', default=-1, type=int, help='number of warmup iterations to run')
    parser.add_argument('-i', '--iterations', default=-1, type=int, help='number of total iterations to run')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='ccl', type=str, help='distributed backend')
    parser.add_argument("--weight-sharing", action='store_true', default=False, help="using weight_sharing to test the performance of inference")
    parser.add_argument("--number-instance", default=0, type=int, help="the instance numbers for test the performance of latcy, only works when enable weight-sharing")

    args = parser.parse_args()
    return args

def run_weights_sharing_model(pipe, tid, args):
    total_time = 0
    for i in range(args.iterations + args.warmup_iterations):
        # run model
        start = time.time()
        if args.precision == "bf16" or args.precision == "fp16" or args.precision == "int8-bf16":
            with torch.autocast("cpu", dtype=args.dtype), torch.no_grad():
                output = pipe(args.prompt, generator=torch.manual_seed(args.seed)).images
        else:
            with torch.no_grad():
                output = pipe(args.prompt, generator=torch.manual_seed(args.seed)).images
        end = time.time()
        print('time per prompt(s): {:.2f}'.format((end - start)))
        if i >= args.warmup_iterations:
            total_time += end - start

    print("Instance num: ", tid)
    print("Latency: {:.2f} s".format(total_time / args.iterations))
    print("Throughput: {:.5f} samples/sec".format(args.iterations / total_time))

def main():

    args = parse_args()
    logging.info(f"Parameters {args}")

    # CCL related
    os.environ['MASTER_ADDR'] = str(os.environ.get('MASTER_ADDR', '127.0.0.1'))
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        print("World size: ", args.world_size)

    args.distributed = args.world_size > 1
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

    # load model
    pipe = DiffusionPipeline.from_pretrained(args.model_name_or_path)
    if not args.accuracy:
        pipe.safety_checker = None

    # data type
    if args.precision == "fp32":
        print("Running fp32 ...")
        args.dtype=torch.float32
    elif args.precision == "bf32":
        print("Running bf32 ...")
        args.dtype=torch.float32
    elif args.precision == "bf16":
        print("Running bf16 ...")
        args.dtype=torch.bfloat16
    elif args.precision == "fp16":
        print("Running fp16 ...")
        args.dtype=torch.half
    elif args.precision == "int8-bf16":
        print("Running int8-bf16 ...")
        args.dtype=torch.bfloat16
    elif args.precision == "int8-fp32":
        print("Running int8-fp32 ...")
        args.dtype=torch.float32
    else:
        raise ValueError("--precision needs to be the following: fp32, bf32, bf16, fp16, int8-bf16, int8-fp32")

    if args.compile_inductor:
        pipe.precision = torch.float32
    elif args.model_name_or_path == "SimianLuo/LCM_Dreamshaper_v7" and args.precision == "int8-bf16":
        pipe.precision = torch.float32
    else:
        pipe.precision = args.dtype
    if args.model_name_or_path == "stabilityai/stable-diffusion-2-1":
        text_encoder_input = torch.ones((1, 77), dtype=torch.int64)
        input = torch.randn(2, 4, 96, 96).to(memory_format=torch.channels_last).to(dtype=pipe.precision), torch.tensor(921), torch.randn(2, 77, 1024).to(dtype=pipe.precision)
    elif args.model_name_or_path == "SimianLuo/LCM_Dreamshaper_v7":
        text_encoder_input = torch.ones((1, 77), dtype=torch.int64)
        input = torch.randn(1, 4, 96, 96).to(memory_format=torch.channels_last).to(dtype=pipe.precision), torch.tensor(921), torch.randn(1, 77, 768).to(dtype=pipe.precision), torch.randn(1, 256).to(dtype=pipe.precision)
    else:
         raise ValueError("This script currently only supports stabilityai/stable-diffusion-2-1 and SimianLuo/LCM_Dreamshaper_v7.")

    if args.distributed:
        import oneccl_bindings_for_pytorch
        torch.distributed.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        print("Rank and world size: ", torch.distributed.get_rank()," ", torch.distributed.get_world_size())
        # print("Create DistributedDataParallel in CPU")
        # pipe = torch.nn.parallel.DistributedDataParallel(pipe)

    # prepare dataloader
    val_coco = dset.CocoCaptions(root = '{}/val2017'.format(args.dataset_path),
                                annFile = '{}/annotations/captions_val2017.json'.format(args.dataset_path),
                                transform=transforms.Compose([transforms.Resize((512, 512)), transforms.PILToTensor(), ]))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_coco, shuffle=False)
    else:
        val_sampler = None

    val_dataloader = torch.utils.data.DataLoader(val_coco,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0,
                                                sampler=val_sampler)


    # torch.compile with inductor backend
    if args.compile_inductor:
        print("torch.compile with inductor backend ...")
        # torch._inductor.config.profiler_mark_wrapper_call = True
        # torch._inductor.config.cpp.enable_kernel_profile = True
        torch._inductor.config.cpp.enable_concat_linear = True
        from torch._inductor import config as inductor_config
        inductor_config.cpp_wrapper = True
        if args.precision == "fp32":
            with torch.no_grad():
                pipe.unet = torch.compile(pipe.unet)
                pipe.unet(*input)
                pipe.unet(*input)
                pipe.text_encoder = torch.compile(pipe.text_encoder)
                pipe.vae.decode = torch.compile(pipe.vae.decode)
        elif args.precision == "bf16":
            with torch.autocast("cpu", ), torch.no_grad():
                pipe.unet = torch.compile(pipe.unet)
                pipe.unet(*input)
                pipe.unet(*input)
                pipe.text_encoder = torch.compile(pipe.text_encoder)
                pipe.vae.decode = torch.compile(pipe.vae.decode)
        elif args.precision == "fp16":
            with torch.autocast("cpu", dtype=torch.half), torch.no_grad():
                pipe.unet = torch.compile(pipe.unet)
                pipe.unet(*input)
                pipe.unet(*input)
                pipe.text_encoder = torch.compile(pipe.text_encoder)
                pipe.vae.decode = torch.compile(pipe.vae.decode)
        elif args.precision == "int8-fp32" or args.precision == "int8-bf16":
            from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
            import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
            from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
            from torch.export import export_for_training
            if args.calibration:
                with torch.no_grad():
                    pipe.traced_unet = export_for_training(pipe.unet, input).module()
                    quantizer = X86InductorQuantizer()
                    if args.model_name_or_path == "SimianLuo/LCM_Dreamshaper_v7":
                        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config()) \
                            .set_module_name_qconfig("up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_q", None) \
                            .set_module_name_qconfig("up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_k", None) \
                            .set_module_name_qconfig("up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v", None) \
                            .set_module_name_qconfig("up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_out.0", None) \
                            .set_module_name_qconfig("up_blocks.2.attentions.2.transformer_blocks.0.ff.net.2", None) \
                            .set_module_name_qconfig("up_blocks.2.attentions.2.transformer_blocks.0.ff.net.0.proj", None) \
                            .set_module_name_qconfig("up_blocks.2.resnets.0.time_emb_proj", None) \
                            .set_module_name_qconfig("up_blocks.2.resnets.1.time_emb_proj", None) \
                            .set_module_name_qconfig("up_blocks.2.resnets.2.time_emb_proj", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_q", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_out.0", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.0.transformer_blocks.0.ff.net.2", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.0.transformer_blocks.0.ff.net.0.proj", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_q", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_k", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_out.0", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.1.transformer_blocks.0.ff.net.2", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.1.transformer_blocks.0.ff.net.0.proj", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2", None) \
                            .set_module_name_qconfig("up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj", None) \
                            .set_module_name_qconfig("up_blocks.3.resnets.0.time_emb_proj", None) \
                            .set_module_name_qconfig("up_blocks.3.resnets.1.time_emb_proj", None) \
                            .set_module_name_qconfig("up_blocks.3.resnets.2.time_emb_proj", None) \
                            .set_module_name_qconfig("mid_block.attentions.0.transformer_blocks.0.attn1.to_q", None) \
                            .set_module_name_qconfig("mid_block.attentions.0.transformer_blocks.0.attn1.to_k", None) \
                            .set_module_name_qconfig("mid_block.attentions.0.transformer_blocks.0.attn1.to_v", None) \
                            .set_module_name_qconfig("mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0", None) \
                            .set_module_name_qconfig("mid_block.attentions.0.transformer_blocks.0.attn2.to_q", None) \
                            .set_module_name_qconfig("mid_block.attentions.0.transformer_blocks.0.attn2.to_k", None) \
                            .set_module_name_qconfig("mid_block.attentions.0.transformer_blocks.0.attn2.to_v", None) \
                            .set_module_name_qconfig("mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0", None) \
                            .set_module_name_qconfig("mid_block.attentions.0.transformer_blocks.0.ff.net.2", None) \
                            .set_module_name_qconfig("mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj", None) \
                            .set_module_name_qconfig("mid_block.resnets.0.time_emb_proj", None) \
                            .set_module_name_qconfig("mid_block.resnets.slice(1, None, None)._modules.0.time_emb_proj", None)
                    else:
                        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                    pipe.traced_unet = prepare_pt2e(pipe.traced_unet, quantizer)
                    # calibration
                    if args.model_name_or_path == "SimianLuo/LCM_Dreamshaper_v7":
                        for i, (images, prompts) in enumerate(tqdm(val_dataloader)):
                            prompt = prompts[0][0]
                            pipe(prompt, generator=torch.manual_seed(args.seed))
                            if i == 119:
                                break
                    else:
                        pipe(args.prompt)
                    pipe.traced_unet = convert_pt2e(pipe.traced_unet)
                    
                    quantized_unet = torch.export.export(pipe.traced_unet, input)
                    torch.export.save(quantized_unet, args.quantized_model_path)
                    print(".........calibration step done..........")
                    return
            else:
                quantized_unet = torch.export.load(args.quantized_model_path)
                pipe.traced_unet = quantized_unet.module()
                torch.ao.quantization.move_exported_model_to_eval(pipe.traced_unet)
                if args.precision == "int8-fp32":
                    with torch.no_grad():
                        pipe.traced_unet = torch.compile(pipe.traced_unet)
                        pipe.traced_unet(*input)
                        pipe.traced_unet(*input)
                        pipe.text_encoder = torch.compile(pipe.text_encoder)
                        pipe.vae.decode = torch.compile(pipe.vae.decode)
                elif args.precision == "int8-bf16":
                    with torch.autocast("cpu", ), torch.no_grad():
                        pipe.traced_unet = torch.compile(pipe.traced_unet)
                        pipe.traced_unet(*input)
                        pipe.traced_unet(*input)
                        pipe.text_encoder = torch.compile(pipe.text_encoder)
                        pipe.vae.decode = torch.compile(pipe.vae.decode)
        else:
            raise ValueError("If you want to use torch.compile with inductor backend, --precision needs to be the following: fp32, bf16, int8-bf16, int8-fp32")

    # benchmark
    if args.benchmark:
        print("Running benchmark ...")
        if args.weight_sharing:
            print("weight sharing ...")
            threads = []
            for i in range(1, args.number_instance+1):
                thread = threading.Thread(target=run_weights_sharing_model, args=(pipe, i, args))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            exit()
        else:
            total_time = 0
            for i in range(args.iterations + args.warmup_iterations):
                # run model
                start = time.time()
                if args.precision == "bf16" or args.precision == "fp16" or args.precision == "int8-bf16":
                    with torch.autocast("cpu", dtype=args.dtype), torch.no_grad():
                        output = pipe(args.prompt, generator=torch.manual_seed(args.seed)).images
                else:
                    with torch.no_grad():
                        output = pipe(args.prompt, generator=torch.manual_seed(args.seed)).images
                end = time.time()
                print('time per prompt(s): {:.2f}'.format((end - start)))
                if i >= args.warmup_iterations:
                    total_time += end - start

            print("Latency: {:.2f} s".format(total_time / args.iterations))
            print("Throughput: {:.5f} samples/sec".format(args.iterations / total_time))

    if args.accuracy:
        print("Running accuracy ...")
        # run model
        if args.distributed:
            torch.distributed.barrier()
        fid = FrechetInceptionDistance(normalize=True)
        for i, (images, prompts) in enumerate(tqdm(val_dataloader)):
            prompt = prompts[0][0]
            real_image = images[0]
            print("prompt: ", prompt)
            if args.precision == "bf16" or args.precision == "fp16" or args.precision == "int8-bf16":
                with torch.autocast("cpu", dtype=args.dtype), torch.no_grad():
                    output = pipe(prompt, generator=torch.manual_seed(args.seed), output_type="numpy").images
            else:
                with torch.no_grad():
                    output = pipe(prompt, generator=torch.manual_seed(args.seed), output_type="numpy").images

            if args.output_dir:
                if not os.path.exists(args.output_dir):
                    os.mkdir(args.output_dir)
                image_name = time.strftime("%Y%m%d_%H%M%S")
                Image.fromarray((output[0] * 255).round().astype("uint8")).save(f"{args.output_dir}/fake_image_{image_name}.png")
                Image.fromarray(real_image.permute(1, 2, 0).numpy()).save(f"{args.output_dir}/real_image_{image_name}.png")

            fake_image = torch.tensor(output[0]).unsqueeze(0).permute(0, 3, 1, 2)
            real_image = real_image.unsqueeze(0) / 255.0

            fid.update(real_image, real=True)
            fid.update(fake_image, real=False)

            if args.iterations > 0 and i == args.iterations - 1:
                break

        print(f"FID: {float(fid.compute())}")

if __name__ == '__main__':
    main()
