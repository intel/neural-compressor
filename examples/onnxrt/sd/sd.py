# import torch
# from diffusers import StableDiffusionPipeline
#
# model_id = "CompVis/stable-diffusion-v1-4"
# device = "cuda"
#
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to(device)
#
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
#
# image.save("astronaut_rides_horse.png")


from typing import Callable, List, Optional, Union
import numpy as np
import torch
import inspect

import random


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed()

# from diffusers import StableDiffusionOnnxPipeline,OnnxRuntimeModel
#
# pipe = StableDiffusionOnnxPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     revision="onnx",
#     provider="CPUExecutionProvider",
#     ##provider="CUDAExecutionProvider",
# )
#
# unet = pipe.unet
# pipe.unet = OnnxRuntimeModel.from_pretrained("unet_quant_per_op_smooth1.0_100/")
# # unet.save_pretrained("sd_unet/")
#
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
# image.save("test_quant.png")
# tmp = 1
# exit()




class CalibDataloader:
    def __init__(self):
        self.batch_size = 1
        self.prompts = ["a photo of an astronaut riding a horse on mars",
                        ##"a photo of an astronaut riding a bike on mars",
                        ]
        from diffusers import StableDiffusionOnnxPipeline

        self.pipe = StableDiffusionOnnxPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            revision="onnx",
            provider="CPUExecutionProvider",

        )

        self.unet_inputs = []
        for prompt in self.prompts:
            self.get_unet_input(prompt)

    def __iter__(self):
        for data in self.unet_inputs:
            yield data

    def get_unet_input(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: Optional[float] = 0.0,
            generator: Optional[np.random.RandomState] = None,
            latents: Optional[np.ndarray] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
            callback_steps: Optional[int] = 1,
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if generator is None:
            generator = np.random

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self.pipe._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # get the initial random noise unless the user supplied it
        latents_dtype = text_embeddings.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps)

        latents = latents * np.float(self.pipe.scheduler.init_noise_sigma)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.pipe.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # timestep_dtype = next(
        #     (input.type for input in self.pipe.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        # )
        timestep_dtype = np.int64

        for i, t in enumerate(self.pipe.progress_bar(self.pipe.scheduler.timesteps)):
            # if i == 1:
            #     break  ##TODO for debug issue
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            self.unet_inputs.append(((latent_model_input, timestep, text_embeddings), 0))
            noise_pred = self.pipe.unet(sample=latent_model_input, timestep=timestep,
                                        encoder_hidden_states=text_embeddings)
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.pipe.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

            ##noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=text_embeddings)


import logging
import argparse

import numpy as np
import onnx

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARN)

if __name__ == "__main__":
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./sd_unet/model.onnx',
        help="Pre-trained mobilenet_v3 model on onnx file"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true', \
        default=False
    )
    parser.add_argument(
        '--tune',
        action='store_true', \
        default=False,
        help="whether quantize the model"
    )
    parser.add_argument(
        '--config',
        type=str,
        default="./examples/onnxrt/sd/unet.yaml",
        help="config yaml path"
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default="sd_quant_unet.onnx",
        help="output model path"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='performance',
        help="benchmark mode of performance or accuracy"
    )
    args = parser.parse_args()

    # from neural_compressor.experimental import Quantization, common
    # quantize = Quantization(args.config)
    # quantize.model = common.Model(args.model_path)
    # calib_loader = CalibDataloader()
    # quantize.calib_dataloader = calib_loader
    # q_model = quantize()
    # q_model.save(args.output_model)

    from neural_compressor import quantization, PostTrainingQuantConfig
    from neural_compressor.config import AccuracyCriterion
    # model = onnx.load(args.model_path)
    # accuracy_criterion = AccuracyCriterion()
    # accuracy_criterion.absolute = 0.01
    config = PostTrainingQuantConfig(
        ##accuracy_criterion=accuracy_criterion,
        quant_format="QDQ",
    )
    calib_loader = CalibDataloader()
    q_model = quantization.fit(args.model_path, config, calib_dataloader=calib_loader)

    q_model.save(args.output_model)
