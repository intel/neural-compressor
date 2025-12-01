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

import torch

from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    convert,
    prepare,
)
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
import torch
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from auto_round import AutoRound
import json
import torchvision
import torch
import einops
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

parser = argparse.ArgumentParser(
    description="FramePack quantization.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--scheme", default="MXFP8", type=str, help="quantizaion scheme.")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--inference", action="store_true")
parser.add_argument("--output_dir", "--quantized_model_path", default="./tmp_autoround", type=str, help="the directory to save quantized model")
parser.add_argument("--dataset_location", type=str, help="path of cloned VBench repository which contains images and prompts for evaluation")
parser.add_argument("--output_video_path", default="./tmp_video", type=str, help="the directory to save generated videos")
parser.add_argument("--limit", default=-1, type=int, help="limit the number of prompts for evaluation")
parser.add_argument("--seed", default=31337, type=int, help="random seed")
parser.add_argument("--total_second_length", default=5, type=int, help="length of generated video")
parser.add_argument("--latent_window_size", default=9, type=int)
parser.add_argument("--steps", default=25, type=float, help="number of inference step")
parser.add_argument("--cfg", default=1.0, type=float, help="real guidance scale")
parser.add_argument("--gs", default=10.0, type=float, help="distilled guidance scale")
parser.add_argument("--rs", default=0.0, type=float, help="guidance rescale")
parser.add_argument("--gpu_memory_preservation", default=6, type=int)
parser.add_argument("--use_teacache", action="store_true", help="faster speed, but often makes hands and fingers slightly worse")
parser.add_argument("--mp4_crf", default=16, type=int, help="MP4 compression. Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs.")
parser.add_argument(
    "--dimension_list",
    nargs="+",
    choices=["subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality", "i2v_subject", "i2v_background", "camera_motion"],
    help="list of evaluation dimensions, usage: --dimension_list <dim_1> <dim_2>",
)
parser.add_argument("--ratio", default="16-9", type=str, help="aspect ratio of image")

args = parser.parse_args()
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

class VBenchDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        return img_path, label

    def __len__(self):
        return len(self.data)

@torch.no_grad()
def worker(
    text_encoder,
    text_encoder_2,
    image_encoder,
    vae,
    transformer,
    tokenizer,
    tokenizer_2,
    inputs,
    seed,
    total_second_length,
    latent_window_size,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    mp4_crf,
    output_video_path,
):
    dataset = VBenchDataset(inputs)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    for image_path, prompt in dataloader:
        image_path = image_path[0]
        prompt = prompt[0]
        cur_save_path = f"{output_video_path}/{prompt}-0.mp4"

        if os.path.exists(cur_save_path):
            continue

        input_image = Image.open(image_path).convert("RGB")
        input_image = np.array(input_image)
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding

        if not high_vram:
            fake_diffusers_current_device(text_encoder, local_rank)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=local_rank)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        # Processing input image

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding

        if not high_vram:
            load_model_as_complete(vae, target_device=local_rank)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=local_rank)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(f"latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}")

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=local_rank, preserved_memory_gb=gpu_memory_preservation)


            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d["denoised"]
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, "b c t h w -> (b h) (t w) c")

                current_step = d["i"] + 1
                hint = f"Sampling {current_step}/{steps}"
                desc = f"Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ..."
                print(hint, desc)
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler="unipc",
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=local_rank,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=local_rank, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=local_rank)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            print(f"Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}")

            if is_last_section:
                break

        b, c, t, h, w = history_pixels.shape

        per_row = b
        for p in [6, 5, 4, 3, 2]:
            if b % p == 0:
                per_row = p
                break

        history_pixels = torch.clamp(history_pixels.float(), -1., 1.) * 127.5 + 127.5
        history_pixels = history_pixels.detach().cpu().to(torch.uint8)
        video = einops.rearrange(history_pixels, "(m n) c t h w -> t (m h) (n w) c", n=per_row)
        torchvision.io.write_video(cur_save_path, video, fps=30, video_codec="h264", options={"crf": "10"})


if __name__ == "__main__":
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained("lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16).cpu()
    transformer.to(dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    transformer.eval()

    if args.quantize:
        setattr(transformer, "name_or_path", "lllyasviel/FramePackI2V_HY")

        qconfig = AutoRoundConfig(
            scheme=args.scheme,
            iters=0,
            export_format="fake",
            output_dir=args.output_dir,
        )
        transformer = prepare(transformer, qconfig)
        transformer = convert(transformer, qconfig)

    if args.inference:
        text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder", torch_dtype=torch.float16).cpu()
        text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder_2", torch_dtype=torch.float16).cpu()
        tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2")
        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="vae", torch_dtype=torch.float16).cpu()

        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="feature_extractor")
        image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16).cpu()

        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        image_encoder.eval()

        if not high_vram:
            vae.enable_slicing()
            vae.enable_tiling()

        transformer.high_quality_fp32_output_for_inference = True
        print("transformer.high_quality_fp32_output_for_inference = True")

        vae.to(dtype=torch.float16)
        image_encoder.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
        text_encoder_2.to(dtype=torch.float16)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        image_encoder.requires_grad_(False)

        if not os.path.exists(args.output_video_path):
            os.makedirs(args.output_video_path)

        init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        text_encoder.to(local_rank)
        text_encoder_2.to(local_rank)
        image_encoder.to(local_rank)
        vae.to(local_rank)
        transformer.to(local_rank)

        for dimension in args.dimension_list:
            # prepare inputs
            image_folder = os.path.join(args.dataset_location, f"vbench2_beta_i2v/data/crop/{args.ratio}")
            info_list = json.load(open(os.path.join(args.dataset_location, "vbench2_beta_i2v/vbench2_i2v_full_info.json"), "r"))
            inputs = [(os.path.join(image_folder, info["image_name"]), info["prompt_en"]) for info in info_list if dimension in info["dimension"]]
            inputs = inputs if args.limit < 0 else inputs[:args.limit]

            worker(
                  text_encoder,
                  text_encoder_2,
                  image_encoder,
                  vae,
                  transformer,
                  tokenizer,
                  tokenizer_2,
                  inputs,
                  args.seed,
                  args.total_second_length,
                  args.latent_window_size,
                  args.steps,
                  args.cfg,
                  args.gs,
                  args.rs,
                  args.gpu_memory_preservation,
                  args.use_teacache,
                  args.mp4_crf,
                  args.output_video_path,
            )

        destroy_process_group()
