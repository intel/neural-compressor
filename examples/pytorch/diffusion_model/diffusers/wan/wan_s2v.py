import argparse
import json
import logging
import os
import re
from functools import partial
from pathlib import Path

import torch
from neural_compressor.torch.quantization import AutoRoundConfig, convert, prepare

import wan
from auto_round.data_type.fp8 import quant_fp8_sym
from auto_round.data_type.mxfp import quant_mx_rceil
from auto_round.utils import get_block_names, get_module
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import merge_video_audio, save_video


def parse_args():
    parser = argparse.ArgumentParser(description="Wan s2v quantization and inference example")
    parser.add_argument("--model", required=True, type=str, help="Wan S2V checkpoint directory")
    parser.add_argument("--task", default="s2v-14B", choices=["s2v-14B"], type=str)
    parser.add_argument("--scheme", default="BF16", choices=["BF16", "FP8", "MXFP8"], type=str)
    parser.add_argument("--quantize", action="store_true", help="Quantize Wan S2V noise model with AutoRound")
    parser.add_argument("--inference", action="store_true", help="Run S2V inference")
    parser.add_argument("--output_dir", "--quantized_model", default="./tmp_autoround_s2v", type=str, help="Output dir for quantized model")

    parser.add_argument("--output_video_path", default="./wan_s2v_video", type=str)
    parser.add_argument("--manifest_path", default=None, type=str, help="Path to JSON with prompt/image/audio samples")

    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--image", default=None, type=str)
    parser.add_argument("--audio", default=None, type=str)

    parser.add_argument("--size", default="1280*720", type=str)
    parser.add_argument("--infer_frames", default=80, type=int)
    parser.add_argument("--num_clip", default=None, type=int)

    parser.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"], type=str)
    parser.add_argument("--sample_steps", default=None, type=int)
    parser.add_argument("--sample_shift", default=None, type=float)
    parser.add_argument("--sample_guide_scale", default=None, type=float)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--enable_tts", action="store_true")
    parser.add_argument("--tts_prompt_audio", default=None, type=str)
    parser.add_argument("--tts_prompt_text", default=None, type=str)
    parser.add_argument("--tts_text", default=None, type=str)
    parser.add_argument("--pose_video", default=None, type=str)
    parser.add_argument("--start_from_ref", action="store_true")
    parser.add_argument("--offload_model", action="store_true")

    parser.add_argument("--mxfp8_chunk_rows", default=2048, type=int)
    parser.add_argument("--disable_mxfp8_inplace_qdq", action="store_true")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def sanitize_filename(text):
    if not text:
        return "sample"
    clean = re.sub(r"[^0-9a-zA-Z._-]+", "_", text).strip("_")
    return clean[:80] if clean else "sample"


def build_samples(args):
    if not args.manifest_path:
        raise ValueError("S2V requires --manifest_path")

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = []
    if isinstance(manifest, dict):
        iterator = manifest.items()
    elif isinstance(manifest, list):
        iterator = [(str(i), item) for i, item in enumerate(manifest)]
    else:
        raise ValueError("Manifest must be a JSON object or list")

    for sample_id, sample in iterator:
        if not isinstance(sample, dict):
            continue
        prompt = sample.get("prompt", args.prompt)
        image = sample.get("image", args.image)
        audio = sample.get("audio", args.audio)

        if not prompt or not image or not audio:
            logging.warning("Skip sample %s: missing prompt/image/audio", sample_id)
            continue
        entries.append({"id": str(sample_id), "prompt": prompt, "image": image, "audio": audio})

    if not entries:
        raise ValueError("No valid samples found in manifest")
    return entries


def apply_activation_qdq(model, args):
    if args.scheme == "BF16":
        return

    if args.scheme == "FP8":
        logging.info("Enable FP8 activation QDQ for S2V linear layers")

        def act_qdq_forward(module, x, *fwd_args, **fwd_kwargs):
            qdq_x, _, _ = quant_fp8_sym(x, group_size=0)
            return module.orig_forward(qdq_x, *fwd_args, **fwd_kwargs)

        for _, module in model.named_modules():
            if module.__class__.__name__ == "Linear":
                module.orig_forward = module.forward
                module.forward = partial(act_qdq_forward, module)
        return

    logging.info(
        "Enable MXFP8 activation QDQ (inplace=%s, chunk_rows=%s)",
        not args.disable_mxfp8_inplace_qdq,
        args.mxfp8_chunk_rows,
    )

    def act_qdq_forward(module, x, *fwd_args, **fwd_kwargs):
        chunk_rows = max(1, int(args.mxfp8_chunk_rows))
        use_inplace = not args.disable_mxfp8_inplace_qdq

        if use_inplace and x.is_cuda:
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
            qdq_x = quant_mx_rceil(x, bits=8, group_size=32, data_type="mx_fp_rceil")[0]

        return module.orig_forward(qdq_x, *fwd_args, **fwd_kwargs)

    for block_names in get_block_names(model):
        for block_name in block_names:
            block = get_module(model, block_name)
            for _, module in block.named_modules():
                if module.__class__.__name__ == "Linear":
                    module.orig_forward = module.forward
                    module.forward = partial(act_qdq_forward, module)


def quantize_noise_model(model, args):
    if args.scheme == "BF16":
        raise ValueError("BF16 does not need quantization. Use --scheme FP8 or --scheme MXFP8.")

    layer_config = {}
    kwargs = {}
    if args.scheme == "FP8":
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Linear":
                layer_config[name] = {"bits": 8, "data_type": "fp", "group_size": 0, "sym": True}
    else:
        scheme = {
            "bits": 8,
            "group_size": 32,
            "data_type": "mx_fp",
        }

    os.makedirs(args.output_dir, exist_ok=True)
    qconfig = AutoRoundConfig(
        iters=0,
        disable_opt_rtn=True,
        layer_config=layer_config,
        export_format="fake",
        output_dir=args.output_dir,
        scheme=scheme,
    )

    logging.info("Prepare + convert S2V noise model (%s)", args.scheme)
    model = prepare(model, qconfig)
    model = convert(model)
    logging.info("S2V quantization done. Output saved to %s", args.output_dir)

def load_quantized_noise_model(wan_s2v, output_dir):
    from wan.modules.s2v.model_s2v import WanModel_S2V

    noise_model = WanModel_S2V.from_pretrained(
        output_dir,
        torch_dtype=torch.bfloat16
    )
    noise_model.eval()
    logging.info("Loading quantized noise_model from %s", output_dir)
    setattr(wan_s2v, "noise_model", noise_model)


def run_inference(wan_s2v, args, cfg):
    os.makedirs(args.output_video_path, exist_ok=True)

    samples = build_samples(args)
    logging.info("Start S2V generation, total samples: %s", len(samples))

    for sample in samples:
        prompt = sample["prompt"]
        image_path = sample["image"]
        audio_path = sample["audio"]
        base = f"{sample['id']}_{sanitize_filename(prompt)}.mp4"
        save_file = os.path.join(args.output_video_path, base)
        save_file_abs = os.path.abspath(save_file)

        if os.path.exists(save_file_abs):
            logging.info("Skip %s: video already exists: %s", sample["id"], save_file_abs)
            continue

        if not os.path.exists(image_path):
            logging.warning("Skip %s: image not found: %s", sample["id"], image_path)
            continue
        if not os.path.exists(audio_path) and not args.enable_tts:
            logging.warning("Skip %s: audio not found: %s", sample["id"], audio_path)
            continue

        video = wan_s2v.generate(
            input_prompt=prompt,
            ref_image_path=image_path,
            audio_path=audio_path,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=args.num_clip,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.infer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.seed,
            offload_model=args.offload_model,
            init_first_frame=args.start_from_ref,
        )

        save_video(
            tensor=video[None],
            save_file=save_file_abs,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        if args.enable_tts:
            merge_video_audio(video_path=save_file_abs, audio_path="tts.wav")
        else:
            merge_video_audio(video_path=save_file_abs, audio_path=audio_path)

        logging.info("Saved: %s", save_file_abs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    args = parse_args()
    setup_logging()
    if args.task not in WAN_CONFIGS:
        raise ValueError(f"Unsupported task: {args.task}")
    if args.size not in MAX_AREA_CONFIGS:
        raise ValueError(f"Unsupported --size {args.size}; valid keys: {list(MAX_AREA_CONFIGS.keys())}")

    cfg = WAN_CONFIGS[args.task]
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    logging.info("Create WanS2V pipeline from %s", args.model)
    wan_s2v = wan.WanS2V(
        config=cfg,
        checkpoint_dir=args.model,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )

    if args.quantize:
        quantize_noise_model(wan_s2v.noise_model, args)

    if args.inference:
        if args.scheme in ["FP8","MXFP8"]:
            load_quantized_noise_model(wan_s2v, args.output_dir)
            apply_activation_qdq(wan_s2v.noise_model, args)

        run_inference(wan_s2v, args, cfg)


if __name__ == "__main__":
    main()
