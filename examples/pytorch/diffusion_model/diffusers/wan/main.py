import argparse
import json
import os
import random

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from functools import partial
from neural_compressor.torch.quantization import AutoRoundConfig, convert, prepare

from auto_round.data_type.fp8 import quant_fp8_sym
from auto_round.data_type.mxfp import quant_mx_rceil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wan quantization and evaluation example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "--model_name", "--model_name_or_path", required=True, type=str, help="Wan model path")
    parser.add_argument("--task", default="t2v", choices=["t2v", "i2v"], help="Wan task type")
    parser.add_argument("--scheme", default="BF16", choices=["BF16", "FP8", "MXFP8"], type=str, help="Quantization scheme")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--output_dir", "--quantized_model_path", default="./tmp_autoround", type=str, help="Directory to save quantized transformer weights")
    parser.add_argument("--prompt_folder", type=str, default=None, help="T2V prompt folder path")
    parser.add_argument("--image_folder", type=str, default=None, help="I2V image folder path")
    parser.add_argument("--info_json", type=str, default=None, help="I2V info json file path")
    parser.add_argument(
        "--dimension",
        type=str,
        default=None,
        help=(
            "VBench dimension used by t2v/i2v evaluation or input filtering "
            "(validated examples: t2v=subject_consistency,overall_consistency; "
            "i2v=i2v_subject,i2v_background)"
        ),
    )
    parser.add_argument("--output_video_path", default="./tmp_video", type=str, help="Directory to save generated videos")
    parser.add_argument("--limit", default=-1, type=int, help="Limit the number of prompts for evaluation")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--height", default=720, type=int)
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--num_frames", default=81, type=int)
    parser.add_argument("--num_inference_steps", default=40, type=int)
    parser.add_argument("--guidance_scale", default=4.0, type=float, help="Guidance scale for t2v/i2v")
    parser.add_argument("--guidance_scale_2", default=3.0, type=float, help="Second guidance scale for t2v only")
    parser.add_argument("--fps", default=16, type=int)
    parser.add_argument("--ratio", default="16-9", type=str, help="Aspect ratio used by i2v VBench dataset")
    parser.add_argument("--image_max_area", default=480 * 832, type=int, help="Maximum i2v image area")
    parser.add_argument(
        "--mxfp8_chunk_rows",
        default=2048,
        type=int,
        help="Row chunk size for MXFP8 activation QDQ",
    )
    parser.add_argument(
        "--disable_mxfp8_inplace_qdq",
        action="store_true",
        help="Disable in-place MXFP8 activation QDQ",
    )
    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_scheme_config(scheme):
    if scheme == "FP8":
        return {"bits": 8, "data_type": "fp", "group_size": 0, "sym": True}
    if scheme == "MXFP8":
        return {"bits": 8, "data_type": "mx_fp", "group_size": 32}
    return None


def build_pipeline(args):
    if args.task == "t2v":
        vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        return pipe

    if args.task == "i2v":
        pipe = WanImageToVideoPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        return pipe

    raise ValueError(f"Unsupported task: {args.task}. Supported tasks are: i2v, t2v")


def quantize_pipleine(pipe, args):
    scheme_cfg = get_scheme_config(args.scheme)
    if scheme_cfg is None:
        raise ValueError("BF16 does not need quantization. Use --scheme FP8 or --scheme MXFP8 with --quantize.")


    qconfig = AutoRoundConfig(
        iters=0,
        export_format="fake",
        output_dir=args.output_dir,
        disable_opt_rtn=True,
        scheme=scheme_cfg,
    )
    pipe = prepare(pipe, qconfig)
    convert(pipe, qconfig)


def apply_activation_qdq(pipe, scheme, runtime_args):
    if scheme == "BF16":
        return

    if scheme == "FP8":
        def act_qdq_forward(module, x, *f_args, **f_kwargs):
            qdq_x, _, _ = quant_fp8_sym(x, group_size=0)
            return module.orig_forward(qdq_x, *f_args, **f_kwargs)
    else:
        def act_qdq_forward(module, x, *f_args, **f_kwargs):
            chunk_rows = max(1, int(getattr(runtime_args, "mxfp8_chunk_rows", 2048)))
            use_inplace = not getattr(runtime_args, "disable_mxfp8_inplace_qdq", False)

            if use_inplace and x.is_cuda:
                # Chunked in-place QDQ reduces peak activation memory on large tensors.
                x_2d = x.reshape(-1, x.shape[-1])
                total_rows = x_2d.shape[0]
                for start in range(0, total_rows, chunk_rows):
                    end = min(start + chunk_rows, total_rows)
                    qdq_chunk = quant_mx_rceil(
                        x_2d[start:end],
                        bits=8,
                        group_size=32,
                        data_type="mx_fp_rceil",
                    )[0]
                    x_2d[start:end].copy_(qdq_chunk)
                    del qdq_chunk
                qdq_x = x
            else:
                qdq_x = quant_mx_rceil(
                    x,
                    bits=8,
                    group_size=32,
                    data_type="mx_fp_rceil",
                )[0]

            return module.orig_forward(qdq_x, *f_args, **f_kwargs)

    for module_name in ["transformer", "transformer_2"]:
        module = getattr(pipe, module_name)
        for n, m in module.named_modules():
            if m.__class__.__name__ == "Linear" and "blocks" in n:
                m.orig_forward = m.forward
                m.forward = partial(act_qdq_forward, m)


def load_quantized_transformers(pipe, output_dir):
    for module_name in ["transformer", "transformer_2"]:
        q_path = os.path.join(output_dir, module_name)
        if not os.path.isdir(q_path):
            raise ValueError(f"Quantized path does not exist: {q_path}")
        print(f"Loading quantized {module_name} from {q_path}")
        setattr(pipe, module_name, WanTransformer3DModel.from_pretrained(q_path, torch_dtype=torch.bfloat16))


def build_t2v_inputs(args):
    prompt_folder = args.prompt_folder

    if not prompt_folder:
        raise ValueError("--prompt_folder is required for t2v inference/eval")
    if not args.dimension:
        raise ValueError("--dimension is required for t2v inference/eval")
    if not os.path.isdir(prompt_folder):
        raise FileNotFoundError(f"Prompt folder not found: {prompt_folder}")

    prompt_file = os.path.join(prompt_folder, f"{args.dimension}.txt")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found for dimension '{args.dimension}': {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_list = [line.strip() for line in f if line.strip()]

    if args.dimension not in {"subject_consistency", "overall_consistency"}:
        print(
            "[WARN] t2v --dimension is not in validated examples "
            "(subject_consistency, overall_consistency). Continue anyway."
        )

    if args.limit >= 0:
        prompt_list = prompt_list[: args.limit]

    return [{"prompt": prompt} for prompt in prompt_list]


def build_i2v_inputs(args):
    image_folder = args.image_folder
    info_json = args.info_json

    if not image_folder:
        raise ValueError("--image_folder is required for i2v inference/eval")
    if not info_json:
        raise ValueError("--info_json is required for i2v inference/eval")
    if not args.dimension:
        raise ValueError(
            "--dimension is required for i2v inference/eval "
            "(validated examples: i2v_subject, i2v_background)"
        )
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    if not os.path.exists(info_json):
        raise FileNotFoundError(f"Info json not found: {info_json}")

    with open(info_json, "r", encoding="utf-8") as f:
        info_list = json.load(f)

    results = []
    for info in info_list:
        if args.dimension not in info["dimension"]:
            continue

        image_path = os.path.join(image_folder, info["image_name"])
        if not os.path.exists(image_path):
            continue
        results.append(
            {
                "prompt": info["prompt_en"],
                "image_path": image_path,
            }
        )

    if args.limit >= 0:
        results = results[: args.limit]

    return results


def safe_output_path(base_dir, prompt):
    return os.path.join(base_dir, f"{prompt}-0.mp4")


@torch.no_grad()
def run_inference(args, pipe):
    setup_seed(args.seed)
    os.makedirs(args.output_video_path, exist_ok=True)
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)

    if args.task == "t2v":
        inputs = build_t2v_inputs(args)
    else:
        inputs = build_i2v_inputs(args)

    for item in inputs:
        prompt = item["prompt"]
        save_path = safe_output_path(args.output_video_path, prompt)
        if os.path.exists(save_path):
            continue

        if args.task == "t2v":
            frames = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                guidance_scale_2=args.guidance_scale_2,
                num_inference_steps=args.num_inference_steps,
                generator=gen,
            ).frames[0]
        else:
            image = load_image(item["image_path"])
            aspect_ratio = image.height / image.width
            mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
            height = round(np.sqrt(args.image_max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(args.image_max_area / aspect_ratio)) // mod_value * mod_value
            image = image.resize((width, height))

            frames = pipe(
                image=image,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=gen,
            ).frames[0]

        export_to_video(frames, save_path, fps=args.fps)
        print(f"Saved: {save_path}")


def main():
    args = parse_args()

    if not (args.quantize or args.inference):
        raise ValueError("Please enable at least one stage: --quantize or --inference")

    if args.quantize or args.inference:
        pipe = build_pipeline(args)
    else:
        pipe = None

    if args.quantize:
        quantize_pipleine(pipe, args)

    if args.inference:
        if args.scheme in ["FP8", "MXFP8"]:
            load_quantized_transformers(pipe, args.output_dir)
            apply_activation_qdq(pipe, args.scheme, args)
        run_inference(args, pipe)


if __name__ == "__main__":
    main()


