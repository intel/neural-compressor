import os

os.environ["KERAS_BACKEND"] = "jax"
# os.environ["LOGLEVEL"] = "DEBUG"  # Uncomment to enable print_model

import argparse

import keras
import numpy as np
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions

from neural_compressor.jax import quantize_model, StaticQuantConfig
from neural_compressor.jax.utils.utility import print_model

parser = argparse.ArgumentParser("Quantize and use ViT model")
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
args = parser.parse_args()
print("Arguments:", *vars(args).items(), sep="\n")


print("\nLoad original model from:", args.model_path)
vit_model = keras.models.load_model(args.model_path)
print_model(vit_model)


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(img, tf.uint8)


def print_predictions(preds):
    preds = decode_predictions(preds, top=4)
    for i, sample_preds in enumerate(preds):
        print(f"Predictions for sample {i}:")
        for k, pred in enumerate(sample_preds):
            _, class_name, score = pred
            print(f"    top-{k+1}: class={class_name}, score={score:.4f}")


img = load_image("./colva_beach_sq.jpg")
img = tf.expand_dims(img, axis=0)

output = vit_model.predict(img)
print("\nOutput before quantization:")
print_predictions(output)

print("\nPrepare quantization config")
config = StaticQuantConfig(weight_dtype=args.precision, activation_dtype=args.precision)


def calib_function(model):
    model.predict(img)


print("\nStart quantization")
vit_model = quantize_model(vit_model, config, calib_function)
print_model(vit_model)

output = vit_model.predict(img)
print("\nOutput after quantization:")
print_predictions(output)
