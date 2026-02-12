import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import keras
import numpy as np
import tensorflow as tf

from neural_compressor.jax import quantize_model, StaticQuantConfig

parser = argparse.ArgumentParser("Quantize and save ViT model")
parser.add_argument(
    "-p",
    "--precision",
    default="fp8_e4m3",
    type=str,
    choices=["fp8_e4m3", "fp8_e5m2"],
    help="precision for the model",
)
parser.add_argument(
    "-m",
    "--model_path",
    default="vit.keras",
    type=str,
    help="path to the Keras model",
)
parser.add_argument(
    "-q",
    "--quantized_path",
    default="/tmp/vit_quantized.keras",
    type=str,
    help="path where to store quantized model. Notice: it has to have keras file extension (model.keras)",
)
args = parser.parse_args()
print("Arguments:", *vars(args).items(), sep="\n")

print("\nLoad original model from:", args.model_path)
vit_model = keras.models.load_model(args.model_path)
vit_model.summary()


print("Prepare quantization config")
config = StaticQuantConfig(weight_dtype=args.precision, activation_dtype=args.precision)


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(img, tf.uint8)


def calib_function(model):
    img = tf.expand_dims(load_image("./colva_beach_sq.jpg"), axis=0)
    model.predict(img)


print("Start quantization")
vit_model = quantize_model(vit_model, config, calib_function)
vit_model.summary()

print("Save quantized model to:", args.quantized_path)
keras.models.save_model(vit_model, args.quantized_path)
