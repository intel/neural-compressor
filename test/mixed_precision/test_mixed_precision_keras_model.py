import os
import shutil
import unittest

import numpy as np
from tensorflow import keras

from neural_compressor import mix_precision
from neural_compressor.config import MixedPrecisionConfig
from neural_compressor.data import DataLoader, Datasets
from .neural_compressor.utils import logger


def build_sequential_model():
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

    # Print model architecture
    model.summary()

    # Compile model with optimizer
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.save("./models/saved_model")

    return


# Define a customized Metric function
from neural_compressor.metric import BaseMetric


class MyMetric(BaseMetric):
    def __init__(self, *args):
        self.pred_list = []
        self.label_list = []
        self.samples = 0

    def update(self, predict, label):
        self.pred_list.extend(np.argmax(predict, axis=1))
        self.label_list.extend(label)
        self.samples += len(label)

    def reset(self):
        self.pred_list = []
        self.label_list = []
        self.samples = 0

    def result(self):
        correct_num = np.sum(np.array(self.pred_list) == np.array(self.label_list))
        return correct_num / self.samples


class MyMetric_keras(MyMetric):
    def __init__(self, *args):
        super(MyMetric_keras, self).__init__(*args)


class TestMixedPrecisionWithKerasModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.environ["FORCE_FP16"] = "1"
        os.environ["FORCE_BF16"] = "1"
        build_sequential_model()

    @classmethod
    def tearDownClass(self):
        del os.environ["FORCE_FP16"]
        del os.environ["FORCE_BF16"]
        shutil.rmtree("./models", ignore_errors=True)
        shutil.rmtree("./nc_workspace", ignore_errors=True)

    def test_mixed_precision_with_keras_model(self):
        # use dummy dataset for UT test
        dataset = Datasets("tensorflow")["dummy"](shape=(10, 28, 28), low=0.0, high=1.0, label=True)

        dataloader = DataLoader(framework="tensorflow", dataset=dataset)

        config = MixedPrecisionConfig()
        q_model = mix_precision.fit(
            model="./models/saved_model", conf=config, eval_dataloader=dataloader, eval_metric=MyMetric()
        )

        # Optional, run quantized model
        import tensorflow as tf

        with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
            tf.compat.v1.import_graph_def(q_model.graph_def, name="")
            out = sess.run(["Identity:0"], feed_dict={"input:0": dataset.dataset})
            print("Inference is done.")

        found_cast = False
        for i in q_model.graph_def.node:
            if i.op == "Cast":
                found_cast = True
                break
        self.assertEqual(found_cast, True)

    def test_mixed_precision_with_keras_adaptor(self):
        # use dummy dataset for UT test
        dataset = Datasets("tensorflow")["dummy"](shape=(10, 28, 28), low=0.0, high=1.0, label=True)
        dataloader = DataLoader(framework="tensorflow", dataset=dataset)

        # add backend='itex' to run on keras adaptor
        config = MixedPrecisionConfig(backend="itex")

        bf16_model = mix_precision.fit(
            model="./models/saved_model", config=config, eval_dataloader=dataloader, eval_metric=MyMetric_keras()
        )

        bf16_policy = keras.mixed_precision.Policy("mixed_bfloat16")
        # bf16_model.model is an obj of tf.keras.Model
        model_policy = bf16_model.model.dtype_policy
        conv2d_layer_policy = bf16_model.model.get_layer("conv2d").dtype_policy

        self.assertEqual(model_policy.compute_dtype, bf16_policy.compute_dtype)
        logger.info("Passed check of keras model dtype for computation")
        self.assertEqual(conv2d_layer_policy.compute_dtype, bf16_policy.compute_dtype)
        logger.info("Passed check of keras layer dtype for computation")


if __name__ == "__main__":
    unittest.main()
