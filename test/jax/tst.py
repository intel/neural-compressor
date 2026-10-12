import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["LOGLEVEL"] = "DEBUG"

from jax import numpy as jnp
from keras_hub.models import Gemma3CausalLM
from PIL import Image

from neural_compressor.jax import StaticQuantConfig, quantize_model
from neural_compressor.jax.utils.utility import print_model


def load_image(image_path, target_size):
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(target_size, Image.BILINEAR)
        pixels = jnp.array(img)

    return jnp.expand_dims(pixels, 0)


repo_root_path = f"{os.path.dirname(__file__)}/../.."
image_path = f"{repo_root_path}/examples/jax/keras/vit/colva_beach_sq.jpg"
target_size = (224, 224)
image = load_image(image_path, target_size)


def calib_fn(model):
    _ = model.generate(
        {
            "images": image,
            "prompts": "Guess the country where this picture was taken: <start_of_image>?",
        },
        max_length=250,
    )


gemma = Gemma3CausalLM.from_preset("/models/gemma3_instruct_4b-v1", dtype="bfloat16")
config = StaticQuantConfig(
    weight_dtype="fp8_e4m3",
    activation_dtype="fp8_e4m3",
    weight_scale_granularity="per_channel",
    const_scale=True,
    const_weight=True,
)
gemma_q = quantize_model(gemma, config, calib_fn, inplace=False)
print_model(gemma_q)
