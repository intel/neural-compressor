import argparse
import logging
import os
import time
from tqdm import tqdm

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.datasets as dset
import torchvision.transforms as transforms
from neural_compressor.torch.quantization import (
    FP8Config,
    convert,
    finalize_calibration,
    prepare,
)

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Model path",
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="COCO2017 dataset path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A big burly grizzly bear is show with grass in the background.",
        help="input text",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="output path")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--calib", action="store_true", default=False)
    parser.add_argument("--quant", action="store_true", default=False)
    parser.add_argument(
        "--accuracy", action="store_true", default=False, help="test accuracy"
    )
    parser.add_argument(
        "-i",
        "--iterations",
        default=-1,
        type=int,
        help="number of total iterations to run",
    )
    parser.add_argument("--calib_out_dir", type=str, default="inc_fp8/measure", help="A folder to save calibration result")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    logging.info(f"Parameters {args}")

    # load model
    pipe = StableDiffusionPipeline.from_pretrained(args.model_path)
    if not args.accuracy:
        pipe.safety_checker = None

    qconfig = FP8Config(
        fp8_config="E4M3",
        use_qdq=True,
        scale_method="MAXABS_ARBITRARY",
        dump_stats_path=args.calib_out_dir,
        blocklist={"types": ["Conv2d"]},
        hp_dtype="fp32",
    )

    if args.calib:
        pipe.unet = prepare(
            pipe.unet.eval(), qconfig
        )
        pipe(args.prompt)
        finalize_calibration(pipe.unet)

    if args.quant:
        pipe.unet = convert(
            pipe.unet.eval(), qconfig
        )
        print(pipe.unet)
        with torch.no_grad():
           pipe.unet = torch.compile(pipe.unet)
           pipe.text_encoder = torch.compile(pipe.text_encoder)
           pipe.vae.decode = torch.compile(pipe.vae.decode)
 
    if args.accuracy:
        print("Running accuracy ...")
        # prepare dataloader
        val_coco = dset.CocoCaptions(
            root="{}/val2017".format(args.data_path),
            annFile="{}/annotations/captions_val2017.json".format(args.data_path),
            transform=transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.PILToTensor(),
                ]
            ),
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_coco, batch_size=1, shuffle=False, num_workers=0, sampler=None
        )

        # run model
        fid = FrechetInceptionDistance(normalize=True)
        for i, (images, prompts) in enumerate(tqdm(val_dataloader)):
            prompt = prompts[0][0]
            real_image = images[0]
            print("prompt: ", prompt)
            with torch.no_grad():
                output = pipe(
                    prompt,
                    generator=torch.manual_seed(args.seed),
                    output_type="numpy",
                ).images

            if args.output_dir:
                if not os.path.exists(args.output_dir):
                    os.mkdir(args.output_dir)
                image_name = time.strftime("%Y%m%d_%H%M%S")
                Image.fromarray((output[0] * 255).round().astype("uint8")).save(
                    f"{args.output_dir}/fake_image_{image_name}.png"
                )
                Image.fromarray(real_image.permute(1, 2, 0).numpy()).save(
                    f"{args.output_dir}/real_image_{image_name}.png"
                )

            fake_image = torch.tensor(output[0]).unsqueeze(0).permute(0, 3, 1, 2)
            real_image = real_image.unsqueeze(0) / 255.0

            fid.update(real_image, real=True)
            fid.update(fake_image, real=False)

            if args.iterations > 0 and i == args.iterations - 1:
                break

        print(f"FID: {float(fid.compute())}")


if __name__ == "__main__":
    main()

