import os
import torch
import time
import argparse
import sys
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

torch.backends.quantized.engine = 'onednn'

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def test(args, model, prompt):
    generator = torch.Generator("cpu").manual_seed(333)
    total_sample = 0
    total_time = 0.0
    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter):
                elapsed = time.time()
                images = model(prompt, 
                               generator=generator, 
                               num_inference_steps=args.num_inference_steps, 
                               height=args.resolution, 
                               width=args.resolution, 
                               guidance_scale=args.guidance_scale)["images"]
                p.step()
                elapsed = time.time() - elapsed
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_time += elapsed
                    total_sample += 1
    else:
        for i in range(args.num_iter):
            elapsed = time.time()
            images = model(prompt, 
                           generator=generator, 
                           num_inference_steps=args.num_inference_steps, 
                           height=args.resolution, 
                           width=args.resolution, 
                           guidance_scale=args.guidance_scale)["images"]
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_time += elapsed
                total_sample += 1

    print("\n", "-"*20, "Summary", "-"*20)
    latency = total_time / total_sample * 1000
    throughput = total_sample * args.per_device_eval_batch_size / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

    # save image
    if args.save_image:
        from math import ceil
        grid = image_grid(images, rows=args.image_rows, cols=ceil(args.per_device_eval_batch_size / args.image_rows))
        grid.save("astronaut_rides_horse-" + args.precision + ".png")

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                'sd-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument("--model_name_or_path", type=str, default='runwayml/stable-diffusion-v1-5', help="model name")
    parser.add_argument('--prompt', default=["a photo of an astronaut riding a horse on mars"], type=list, help='prompt')
    parser.add_argument('--precision', default="bfloat16", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=5, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=2, type=int, help='test warmup')
    parser.add_argument('--per_device_eval_batch_size', default=1, type=int, help='per_device_eval_batch_size')
    parser.add_argument('--num_inference_steps', default=20, type=int, help='num_inference_steps')
    parser.add_argument('--save_image', action='store_true', default=False, help='save image')
    parser.add_argument('--image_rows', default=1, type=int, help='saved image array')
    parser.add_argument('--guidance_scale', default=7.5, type=float, help='test guidance_scale')
    parser.add_argument('--resolution', default=512, type=int, help='resolution')
    args = parser.parse_args()
    print(args)


    # model
    dpm = DPMSolverMultistepScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")
    model = StableDiffusionPipeline.from_pretrained(args.model_name_or_path, scheduler=dpm, torch_dtype=torch.float)

    # sample input
    sample = torch.randn(2,4,64,64)
    timestep = torch.rand(1)*999
    encoder_hidden_status = torch.randn(2,77,768)
    input_example = (sample, timestep, encoder_hidden_status)
    prompt = args.prompt * args.per_device_eval_batch_size

    # configure model
    if args.channels_last:
        # model = model.to(memory_format=torch.channels_last)
        model.unet = model.unet.to(memory_format=torch.channels_last)
        model.vae = model.vae.to(memory_format=torch.channels_last)
        model.text_encoder = model.text_encoder.to(memory_format=torch.channels_last)
        model.safety_checker = model.safety_checker.to(memory_format=torch.channels_last)
        print("---- Use NHWC model.")

    # start test
    if args.precision == "bfloat16":
        print("---- Use AMP bfloat16")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            test(args, model, prompt)
    else:
        test(args, model, prompt)

