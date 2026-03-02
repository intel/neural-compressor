#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import shutil
import time
import unittest

import numpy as np

from neural_compressor.common import logger


def build_model():
    import tensorflow as tf
    from tensorflow.compat.v1 import graph_util

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(1)

    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    x = tf.compat.v1.placeholder(tf.float32, [1, 32, 32, 3], name="x")
    top_relu = tf.nn.relu(x)
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    x_pad = tf.pad(top_relu, paddings, "CONSTANT")
    conv_weights = tf.compat.v1.get_variable(
        "weight", [3, 3, 3, 3], initializer=tf.compat.v1.random_normal_initializer()
    )
    conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID", name="conv1")
    relu = tf.nn.relu(conv)
    relu6 = tf.nn.relu6(relu, name="op_to_store")
    out_name = relu6.name.split(":")[0]
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
        )

    graph_def.ParseFromString(output_graph_def.SerializeToString())
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return graph


class MyDataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = math.ceil(len(dataset) / self.batch_size)

    def __iter__(self):
        for _, (images, labels) in enumerate(self.dataset):
            images = np.expand_dims(images, axis=0)
            labels = np.expand_dims(labels, axis=0)
            yield (images, labels)

    def __len__(self):
        return self.length


class TestTF3xNewApi(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.graph = build_model()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("baseline_model", ignore_errors=True)

    def test_static_quant_from_dict_default(self):
        logger.info("test_static_quant_from_dict_default")
        from neural_compressor.tensorflow import get_default_static_quant_config, quantize_model
        from neural_compressor.tensorflow.utils import DummyDataset

        dataset = DummyDataset(shape=(100, 32, 32, 3), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset)
        fp32_model = self.graph
        qmodel = quantize_model(fp32_model, get_default_static_quant_config(), calib_dataloader)
        self.assertIsNotNone(qmodel)

        conv2d_quantized = False
        for node in qmodel.graph_def.node:
            if "Quantized" in node.op:
                conv2d_quantized = True
                break

        self.assertEqual(conv2d_quantized, True)

    def test_static_quant_from_dict_beginner(self):
        logger.info("test_static_quant_from_dict_beginner")
        from neural_compressor.tensorflow import quantize_model

        quant_config = {
            "static_quant": {
                "global": {
                    "weight_dtype": "fp32",
                    "act_dtype": "fp32",
                },
            }
        }
        from neural_compressor.tensorflow.utils import DummyDataset

        dataset = DummyDataset(shape=(100, 32, 32, 3), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset)
        fp32_model = self.graph

        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

        quantized = False
        for node in qmodel.graph_def.node:
            if "Quantize" in node.op:
                quantized = True
                break

        self.assertEqual(quantized, False)

    def test_static_quant_from_class_default(self):
        logger.info("test_static_quant_from_class_default")
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
        from neural_compressor.tensorflow.utils import DummyDataset

        dataset = DummyDataset(shape=(100, 32, 32, 3), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset)
        fp32_model = self.graph
        quant_config = StaticQuantConfig()
        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

        conv2d_quantized = False
        for node in qmodel.graph_def.node:
            if "Quantized" in node.op:
                conv2d_quantized = True
                break

        self.assertEqual(conv2d_quantized, True)

    def test_static_quant_from_class_beginner(self):
        logger.info("test_static_quant_from_class_beginner")
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
        from neural_compressor.tensorflow.utils import DummyDataset

        dataset = DummyDataset(shape=(100, 32, 32, 3), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset)
        fp32_model = self.graph
        quant_config = StaticQuantConfig(
            weight_dtype="fp32",
            act_dtype="fp32",
        )
        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

        quantized = False
        for node in qmodel.graph_def.node:
            if "Quantize" in node.op:
                quantized = True
                break

        self.assertEqual(quantized, False)

    def test_static_quant_from_dict_advance(self):
        logger.info("test_static_quant_from_dict_advance")
        from neural_compressor.tensorflow import quantize_model
        from neural_compressor.tensorflow.utils import DummyDataset

        dataset = DummyDataset(shape=(100, 32, 32, 3), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset)
        fp32_model = self.graph
        quant_config = {
            "static_quant": {
                "global": {
                    "weight_dtype": "int8",
                    "weight_sym": True,
                    "weight_granularity": "per_channel",
                    "act_dtype": "int8",
                    "act_sym": True,
                    "act_granularity": "per_channel",
                },
                "local": {
                    "conv1": {
                        "weight_dtype": "fp32",
                        "act_dtype": "fp32",
                    }
                },
            }
        }
        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

        conv2d_quantized = True
        for node in qmodel.graph_def.node:
            if node.name == "conv1" and "Quantize" not in node.op:
                conv2d_quantized = False
                break

        self.assertEqual(conv2d_quantized, False)

    def test_static_quant_from_dict_advance2(self):
        logger.info("test_static_quant_from_dict_advance2")
        from neural_compressor.tensorflow import quantize_model
        from neural_compressor.tensorflow.utils import DummyDataset

        dataset = DummyDataset(shape=(100, 32, 32, 3), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset)
        fp32_model = self.graph
        quant_config = {
            "static_quant": {
                "global": {
                    "weight_dtype": "int8",
                    "weight_sym": True,
                    "weight_granularity": "per_channel",
                    "act_dtype": "int8",
                    "act_sym": True,
                    "act_granularity": "per_channel",
                },
                "local": {
                    "conv1": {
                        "weight_algorithm": "kl",
                        "act_algorithm": "kl",
                    }
                },
            }
        }
        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

    def test_static_quant_from_class_advance(self):
        logger.info("test_static_quant_from_class_advance")
        from neural_compressor.tensorflow import StaticQuantConfig, quantize_model
        from neural_compressor.tensorflow.utils import DummyDataset

        dataset = DummyDataset(shape=(100, 32, 32, 3), label=True)
        calib_dataloader = MyDataLoader(dataset=dataset)
        fp32_model = self.graph
        quant_config = StaticQuantConfig(
            weight_dtype="int8",
            weight_sym=True,
            weight_granularity="per_channel",
            act_dtype="int8",
            act_sym=True,
            act_granularity="per_channel",
        )
        conv2d_config = StaticQuantConfig(
            weight_dtype="fp32",
            act_dtype="fp32",
        )
        quant_config.set_local("conv1", conv2d_config)
        qmodel = quantize_model(fp32_model, quant_config, calib_dataloader)
        self.assertIsNotNone(qmodel)

        conv2d_quantized = True
        for node in qmodel.graph_def.node:
            if node.name == "conv1" and "Quantize" not in node.op:
                conv2d_quantized = False
                break

        self.assertEqual(conv2d_quantized, False)

    def test_config_from_dict(self):
        logger.info("test_config_from_dict")
        from neural_compressor.tensorflow import StaticQuantConfig

        quant_config = {
            "static_quant": {
                "global": {
                    "weight_dtype": "int8",
                    "weight_sym": True,
                    "weight_granularity": "per_tensor",
                    "act_dtype": "int8",
                    "act_sym": True,
                    "act_granularity": "per_tensor",
                },
                "local": {
                    "conv1": {
                        "weight_dtype": "fp32",
                        "act_dtype": "fp32",
                    }
                },
            }
        }
        config = StaticQuantConfig.from_dict(quant_config["static_quant"])
        self.assertIsNotNone(config.local_config)

    def test_config_to_dict(self):
        logger.info("test_config_to_dict")
        from neural_compressor.tensorflow import StaticQuantConfig

        quant_config = StaticQuantConfig(
            weight_dtype="int8",
            weight_sym=True,
            weight_granularity="per_channel",
            act_dtype="int8",
            act_sym=True,
            act_granularity="per_channel",
        )
        conv2d_config = StaticQuantConfig(
            weight_dtype="fp32",
            act_dtype="fp32",
        )
        quant_config.set_local("conv1", conv2d_config)
        config_dict = quant_config.to_dict()
        self.assertIn("global", config_dict)
        self.assertIn("local", config_dict)


class TestQuantConfigForAutotune(unittest.TestCase):
    def test_expand_config(self):
        # test the expand functionalities, the user is not aware it
        from neural_compressor.tensorflow import StaticQuantConfig

        quant_configs = StaticQuantConfig(
            weight_dtype="int8",
            weight_sym=True,
            weight_granularity=["per_channel", "per_tensor"],
            act_dtype="int8",
            act_sym=True,
            act_granularity="per_channel",
        )

        expand_config_list = StaticQuantConfig.expand(quant_configs)
        self.assertEqual(expand_config_list[0].weight_granularity, "per_channel")
        self.assertEqual(expand_config_list[1].weight_granularity, "per_tensor")


if __name__ == "__main__":
    unittest.main()
