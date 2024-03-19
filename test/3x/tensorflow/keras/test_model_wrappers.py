"""Tests for model."""

import os
import platform
import unittest

import numpy as np
import tensorflow as tf
from pkg_resources import parse_version

from neural_compressor.tensorflow.utils.model import Model


def build_keras():
    from tensorflow import keras

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    train_images = train_images.astype(np.float32) / 255.0

    # Create Keras model
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(28, 28), name="input"),
            keras.layers.Reshape(target_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation="softmax", name="output"),
        ]
    )

    # Compile model with optimizer
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # # Train model
    model.fit(x={"input": train_images[0:100]}, y={"output": train_labels[0:100]}, epochs=1)
    return model


class TestModelWrappers(unittest.TestCase):
    def test_keras_h5_model(self):
        if parse_version(tf.version.VERSION) < parse_version("2.3.0"):
            return
        keras_model = build_keras()
        keras_model.save("./simple_model.h5")
        # load from path
        model = Model("./simple_model.h5")

        self.assertGreaterEqual(len(model.output_node_names), 1)
        self.assertGreaterEqual(len(model.input_node_names), 1)
        os.makedirs("./keras_model", exist_ok=True)
        model.save("./keras_model")
        os.system("rm -rf simple_model.h5")
        os.system("rm -rf keras_model")

    def test_keras_saved_model(self):
        if parse_version(tf.version.VERSION) < parse_version("2.3.0"):
            return
        keras_model = build_keras()

        model = Model(keras_model)

        self.assertGreaterEqual(len(model.output_node_names), 1)
        self.assertGreaterEqual(len(model.input_node_names), 1)
        keras_model.save("./simple_model.keras")
        # load from path
        model = Model("./simple_model.keras")

        self.assertGreaterEqual(len(model.output_node_names), 1)
        self.assertGreaterEqual(len(model.input_node_names), 1)

        model.save("./keras_model.keras")
        os.system("rm -rf simple_model.keras")
        os.system("rm -rf keras_model.keras")

    def test_tf_qat_model(self):
        if parse_version(tf.version.VERSION) < parse_version("2.3.0"):
            return
        keras_model = build_keras()

        from neural_compressor.tensorflow.utils.model_wrappers import TensorflowQATModel

        model = TensorflowQATModel(keras_model)
        assert isinstance(model.model, tf.keras.Model)
        self.assertEqual(model.model_path, None)
        keras_model.save("./simple_model.keras")
        # load from path
        model = TensorflowQATModel("./simple_model.keras")
        assert isinstance(model.model, tf.keras.Model)
        self.assertEqual(model.model_path, "./simple_model.keras")

        model.save("./keras_model.keras")
        loaded_model = tf.keras.models.load_model("./keras_model.keras")
        assert isinstance(loaded_model, tf.keras.Model)

        model.save("keras_model.h5")
        loaded_model = tf.keras.models.load_model("keras_model.h5")
        assert isinstance(loaded_model, tf.keras.Model)

        os.system("rm -rf simple_model.keras")
        os.system("rm -rf keras_model.keras")
        os.remove("keras_model.h5")


if __name__ == "__main__":
    unittest.main()
