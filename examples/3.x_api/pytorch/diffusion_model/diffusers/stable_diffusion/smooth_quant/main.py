import os
import logging
import tempfile
import shutil
import argparse
import pandas as pd
import time
import torch
import intel_extension_for_pytorch as ipex
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    deprecate, retrieve_timesteps, rescale_noise_cfg,
    PipelineImageInput, StableDiffusionXLPipelineOutput
)


class StableDiffusionXLPipelineSQ(StableDiffusionXLPipeline):
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = 'cpu'

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                )['sample']

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image.detach(), output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


parser = argparse.ArgumentParser()
parser.add_argument('--model-id', default="stabilityai/stable-diffusion-xl-base-1.0", type=str)
parser.add_argument('--precision', default='fp32', type=str)
parser.add_argument('--base-output-dir', default="./output", type=str)
parser.add_argument('--quantized-unet', default="./saved_results", type=str)
parser.add_argument("--int8", action="store_true", help="Load quantized model.")
parser.add_argument("--load", action="store_true")
parser.add_argument('--iters', default=5000, type=int, help="Num of image generated.")
parser.add_argument('--output-dir-name', default=None, type=str)
parser.add_argument('--output-dir-name-postfix', default=None, type=str)
parser.add_argument('--captions-fname', default="captions_5k.tsv", type=str)
parser.add_argument('--guidance', default=8.0, type=float)
parser.add_argument('--scheduler', default="euler", type=str)
parser.add_argument('--steps', default=20, type=int)
parser.add_argument('--negative-prompt', default="normal quality, low quality, worst quality, low res, blurry, nsfw, nude", type=str)
parser.add_argument('--latent-path', default="latents.pt", type=str)
parser.add_argument('--generator-seed', default=None, type=int)
parser.add_argument("--refiner", dest='refiner', action="store_true",
                    help="Whether to add a refiner to the SDXL pipeline."
                          "Applicable only with --model-id=xl")
parser.add_argument("--no-refiner", dest='refiner', action="store_false",
                    help="Whether to add a refiner to the SDXL pipeline."
                          "Applicable only with --model-id=xl")

args = parser.parse_args()

# Init the logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

if args.latent_path and args.generator_seed:
    raise ValueError(
        "Cannot specify both --latent-path and --generator-seed"
    )

if args.precision == "fp16":
    dtype = torch.float16
elif args.precision == "bf16":
    dtype = torch.bfloat16
else:
    dtype = torch.float32

# Initialize defaults
device = torch.device('cpu')
world_size = 1
rank = 0

# load frozen latent
latent_noise = None
if args.latent_path:
    logging.info(f"[{rank}] loading latent from: {args.latent_path}")
    latent_noise = torch.load(args.latent_path).to(dtype)

logging.info(f"[{rank}] args: {args}")
logging.info(f"[{rank}] world_size: {world_size}")
logging.info(f"[{rank}] device: {device}")

logging.info(f"[{rank}] using captions from: {args.captions_fname}")
df = pd.read_csv(args.captions_fname, sep='\t')
logging.info(f"[{rank}] {len(df)} captions loaded")

# split captions among ranks
df = df[rank::world_size]
logging.info(f"[{rank}] {len(df)} captions assigned")

# Build the pipeline
schedulers = {
    "ddpm": DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler"),
    "ddim": DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler"),
    "euler_anc": EulerAncestralDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler"),
    "euler": EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler"),
}
pipe = StableDiffusionXLPipelineSQ.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype,
    use_safetensors=True,
)
pipe = pipe.to(dtype)  # Ensure all modules are set as dtype
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

if args.refiner:
    refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(args.model_id,
                                                                    scheduler=schedulers[args.scheduler],
                                                                    safety_checker=None,
                                                                    add_watermarker=False,
                                                                    variant="fp16" if args.precision == 'fp16' else None,
                                                                    torch_dtype=dtype)

if args.int8 and args.load:
    from neural_compressor.torch.quantization import load
    example_inputs = {"sample": torch.randn((2, 4, 128, 128), dtype=dtype),
                "timestep": torch.tensor(951.0),
                "encoder_hidden_states": torch.randn((2, 77, 2048), dtype=dtype),
                "added_cond_kwargs": {'text_embeds':torch.randn((2, 1280), dtype=dtype),
                                    'time_ids': torch.tensor([[1024., 1024.,    0.,    0., 1024., 1024.],
                                                                [1024., 1024.,    0.,    0., 1024., 1024.]], dtype=dtype)},}
    q_unet = load(args.quantized_unet)
    for _ in range(2):
        q_unet(**example_inputs)
    print("Loaded Quantized Model")
    setattr(q_unet, "config", pipe.unet.config)
    pipe.unet = q_unet

pipe.set_progress_bar_config(disable=True)
logging.info(f"[{rank}] Pipeline initialized: {pipe}")

if args.refiner:
    refiner_pipe = refiner_pipe.to(device)
    refiner_pipe.set_progress_bar_config(disable=True)
    logging.info(f"[{rank}] Refiner pipeline initialized: {refiner_pipe}")

# Output directory
output_dir = args.output_dir_name or f"{args.model_id.replace('/','--')}__{args.scheduler}__{args.steps}__{args.guidance}__{args.precision}"
if args.output_dir_name_postfix is not None:
    output_dir = f"{output_dir}_{args.output_dir_name_postfix}"

output_dir = os.path.join(args.base_output_dir, output_dir)

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a temporary directory to atomically move the images
tmp_dir = tempfile.mkdtemp()

# Generate the images
for index, row in df.iterrows():
    image_id = row['image_id']
    caption_id = row['id']
    caption_text = row['caption']

    destination_path = os.path.join(output_dir, f"{caption_id}.png")

    if index >= args.iters:
        break

    # Check if the image already exists in the output directory
    if not os.path.exists(destination_path):
        # Generate the image
        print(index, caption_text)
        tic = time.time()
        image = pipe(prompt=caption_text,
                     negative_prompt="normal quality, low quality, worst quality, low res, blurry, nsfw, nude",
                     guidance_scale=8.0,
                     generator=torch.Generator(device=device).manual_seed(args.generator_seed) if args.generator_seed else None,
                     latents=latent_noise,
                     num_inference_steps=20).images[0]
        toc = time.time()
        print("Time taken : ",toc-tic)

        if args.refiner:
            image = refiner_pipe(caption_text,
                                 image=image).images[0]

        # Save the image
        image_path_tmp = os.path.join(tmp_dir, f"{caption_id}.png")
        image.save(image_path_tmp)
        shutil.move(image_path_tmp, destination_path)

        logging.info(f"[{rank}] Saved image {caption_id}: {caption_text}")
