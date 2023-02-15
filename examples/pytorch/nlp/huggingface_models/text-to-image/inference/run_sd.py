
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import intel_extension_for_pytorch as ipex
import time
import sys


# set args
if len(sys.argv) > 1 and sys.argv[1] not in ['fp32', 'float32']:
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = None

# load model
model_id = "runwayml/stable-diffusion-v1-5"
dpm = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=dpm, torch_dtype=torch.float)

# to channels last
pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
pipe.safety_checker = pipe.safety_checker.to(memory_format=torch.channels_last)
# to ipex
sample = torch.randn(2,4,64,64)
timestep = torch.rand(1)*999
encoder_hidden_status = torch.randn(2,77,768)
input_example = (sample, timestep, encoder_hidden_status)
pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=amp_dtype, inplace=True, sample_input=input_example)
pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=amp_dtype, inplace=True)
pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=amp_dtype, inplace=True)
pipe.safety_checker = ipex.optimize(pipe.safety_checker.eval(), dtype=amp_dtype, inplace=True)

prompt = "a photo of an astronaut riding a horse on mars"

# start
elapsed = time.time()
with torch.cpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
    image = pipe(prompt, num_inference_steps=20).images[0]  
elapsed = time.time() - elapsed
print("Inference latency: %.3f sec." % elapsed)

# save image
image.save("astronaut_rides_horse.png")

