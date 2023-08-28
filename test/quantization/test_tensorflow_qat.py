import shutil
import unittest

import numpy as np
import tensorflow as tf
from pkg_resources import parse_version


def train_func():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model architecture.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )
    # Train the digit classification model
    model.compile(
        optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )

    model.fit(
        train_images,
        train_labels,
        epochs=1,
        validation_split=0.1,
    )

    _, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)

    print("Baseline test accuracy:", baseline_model_accuracy)
    model.save("baseline_model")


class Dataset(object):
    def __init__(self, batch_size=100):
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize the input image so that each pixel value is between 0 to 1.
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0
        self.train_labels = train_labels
        self.test_labels = test_labels

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        return self.test_images[idx], self.test_labels[idx]


class TestTensorflowQAT(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        train_func()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("baseline_model", ignore_errors=True)
        shutil.rmtree("trained_qat_model", ignore_errors=True)

    @unittest.skipIf(parse_version(tf.version.VERSION) < parse_version("2.3.0"), "version check")
    def test_qat(self):
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize the input image so that each pixel value is between 0 to 1.
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        from neural_compressor import QuantizationAwareTrainingConfig, training

        config = QuantizationAwareTrainingConfig()
        compression_manager = training.prepare_compression("./baseline_model", config)
        compression_manager.callbacks.on_train_begin()

        q_aware_model = compression_manager.model.model
        # `quantize_model` requires a recompile.
        q_aware_model.compile(
            optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
        )

        train_images_subset = train_images[0:1000]  # out of 60000
        train_labels_subset = train_labels[0:1000]

        q_aware_model.fit(train_images_subset, train_labels_subset, batch_size=500, epochs=1, validation_split=0.1)

        _, q_aware_model_accuracy = q_aware_model.evaluate(test_images, test_labels, verbose=0)

        print("Quant test accuracy:", q_aware_model_accuracy)

        compression_manager.callbacks.on_train_end()
        compression_manager.save("trained_qat_model")

    def test_quantize_recipe(self):
        from neural_compressor.adaptor.tf_utils.quantize_graph.qat.quantize_config import global_config
        from neural_compressor.adaptor.tf_utils.quantize_graph.qat.quantize_helper import (
            init_quantize_config,
            qat_clone_function,
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28)),
                tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10),
            ]
        )

        print("\n")
        print("The layer information of this fp32 mnist model:")
        model.summary()
        # custom setting to decide which layer to be quantized
        quantize_recipe = {
            "conv2d_1": {"quantize": False},
            "max_pooling2d_1": {"quantize": True},
        }
        init_quantize_config(model, quantize_recipe)
        q_model = tf.keras.models.clone_model(model, input_tensors=None, clone_function=qat_clone_function)
        global_config.clear()

        print("\n")
        print("The mnist model after applying QAT:")
        q_model.summary()
        assert (
            q_model.layers[1].name == "conv2d_1"
        ), "The Conv2D layer is incorrectly quantized, the quantize_recipe is ignored !"

    def test_quantize_wrapper(self):
        from neural_compressor.adaptor.tf_utils.quantize_graph.qat.quantize_config import global_config
        from neural_compressor.adaptor.tf_utils.quantize_graph.qat.quantize_helper import (
            init_quantize_config,
            qat_clone_function,
        )

        input_shape = (28, 28, 3)
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(
            inputs
        )
        x = tf.keras.layers.Add()([inputs, x])
        model = tf.keras.models.Model(inputs=inputs, outputs=x)

        init_quantize_config(model)
        q_model = tf.keras.models.clone_model(model, input_tensors=None, clone_function=qat_clone_function)
        global_config.clear()

        depthwise_conv2D_layer = q_model.layers[1]
        assert (
            depthwise_conv2D_layer.name == "quant_depthwise_conv2d"
        ), "The DepthwiseConv2D layer is not quantized as expected."
        depthwise_conv2D_layer.trainable = False
        assert (
            depthwise_conv2D_layer.trainable is False
        ), "The trainable attribute of this layer can not be correctly set."

        input_data = np.random.rand(1, 28, 28, 3)
        training = tf.keras.backend.learning_phase()
        output = depthwise_conv2D_layer(input_data, training=training)
        assert output is not None, "The layer can not be correctly inferenced."


if __name__ == "__main__":
    unittest.main()
