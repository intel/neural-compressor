import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import keras
import numpy as np
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions

import neural_compressor.jax.quantization  # Required to load quantized model

parser = argparse.ArgumentParser("Run statically quantized ViT model")
parser.add_argument(
    "-q",
    "--quantized_path",
    default="/tmp/vit_quantized.keras",
    type=str,
    help="path to quantized model",
)
args = parser.parse_args()
print("Arguments:", *vars(args).items(), sep="\n")

print("Load quantized model from:", args.quantized_path)
vit_model = keras.models.load_model(args.quantized_path)
vit_model.summary()

print("Generate output using quantized model")


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

print("Output:")
print_predictions(output)
