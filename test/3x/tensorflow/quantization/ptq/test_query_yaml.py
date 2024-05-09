#
#  -*- coding: utf-8 -*-
#
import os
import unittest

import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

import neural_compressor
from neural_compressor.tensorflow.algorithms.static_quant.tensorflow import TensorflowQuery
from neural_compressor.tensorflow.utils import disable_random, version1_gte_version2


def build_fake_framework_yaml():
    fake_yaml = """
---
-
  version:
    name: ['2.1.0', '2.2.0', '2.3.0', '2.4.0', '2.5.0', '2.6.0', '2.7.0']

  bf16: ['Conv2D', 'MatMul', 'ConcatV2', 'MaxPool', 'AvgPool', 'DepthwiseConv2dNative']

  int8: {
    'static': {
        'Conv2D': {
          'weight': {
                      'dtype': ['int8'],
                      'scheme': ['sym'],
                      'granularity': ['per_channel','per_tensor'],
                      'algorithm': ['minmax']
                      },
          'activation': {
                      'dtype': ['int8', 'uint8'],
                      'scheme': ['sym'],
                      'granularity': ['per_tensor'],
                      'algorithm': ['minmax', 'kl']
                      }
                  },
        'MatMul': {
          'weight': {
                      'dtype': ['int8'],
                      'scheme': ['sym'],
                      'granularity': ['per_tensor'],
                      'algorithm': ['minmax']
                      },
          'activation': {
                      'dtype': ['int8', 'uint8'],
                      'scheme': ['asym', 'sym'],
                      'granularity': ['per_tensor'],
                      'algorithm': ['minmax']
                      }
                  }
    },
    'dynamic': {
    }
  }

-
  version:
    name: ['default']

  bf16: ['Conv2D', 'MatMul', 'ConcatV2', 'MaxPool', 'AvgPool', 'DepthwiseConv2dNative']

  int8: {
    'static': {
        'Conv2D': {
          'weight': {
                      'dtype': ['int8'],
                      'scheme': ['sym'],
                      'granularity': ['per_channel','per_tensor'],
                      'algorithm': ['minmax']
                      },
          'activation': {
                      'dtype': ['int8', 'uint8'],
                      'scheme': ['sym'],
                      'granularity': ['per_tensor'],
                      'algorithm': ['minmax', 'kl']
                      }
                  },
        'BatchMatMul': {
          'weight': {
                      'dtype': ['int8', 'fp32'],
                      'scheme': ['sym'],
                      'granularity': ['per_tensor'],
                      'algorithm': ['minmax']
                      },
          'activation': {
                      'dtype': ['int8', 'fp32'],
                      'scheme': ['sym'],
                      'granularity': ['per_tensor'],
                      'algorithm': ['minmax']
                      }
                  },
        'BatchMatMulV2': {
          'weight': {
                      'dtype': ['int8'],
                      'scheme': ['sym'],
                      'granularity': ['per_tensor'],
                      'algorithm': ['minmax']
                      },
          'activation': {
                      'dtype': ['int8', 'uint8'],
                      'scheme': ['asym', 'sym'],
                      'granularity': ['per_tensor'],
                      'algorithm': ['minmax']
                      }
                  }
    },
    'dynamic': {
    }
  }
        """
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("fake_framework.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


class TestTFQueryYaml(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tf_yaml_path = neural_compressor.__path__[0] + "/tensorflow/algorithms/static_quant/tensorflow.yaml"

        with open(self.tf_yaml_path) as f:
            self.content = yaml.safe_load(f)
        self.query_handler = TensorflowQuery(local_config_file=self.tf_yaml_path)

    def test_unique_version(self):
        versions = [i["version"]["name"] for i in self.content]
        registered_version_name = []
        for i in versions:
            if isinstance(i, list):
                registered_version_name.extend(i)
            else:
                registered_version_name.append(i)

        self.assertEqual(len(registered_version_name), len(set(registered_version_name)))

    def test_int8_sequences(self):
        patterns = self.query_handler.get_eightbit_patterns()

        has_conv2d = bool("Conv2D" in patterns)
        has_matmul = bool("MatMul" in patterns)
        self.assertEqual(has_conv2d, True)
        self.assertEqual(has_matmul, True)
        self.assertGreaterEqual(len(patterns["Conv2D"]), 13)
        self.assertGreaterEqual(len(patterns["MatMul"]), 3)
        self.assertEqual(len(patterns["ConcatV2"]), 1)
        self.assertEqual(len(patterns["MaxPool"]), 1)
        self.assertEqual(len(patterns["AvgPool"]), 1)

    def test_convert_internal_patterns(self):
        internal_patterns = self.query_handler.generate_internal_patterns()
        self.assertEqual([["MaxPool"]] in internal_patterns, True)
        self.assertEqual([["ConcatV2"]] in internal_patterns, True)
        self.assertEqual([["AvgPool"]] in internal_patterns, True)
        self.assertEqual([["MatMul"], ("BiasAdd",), ("Relu",)] in internal_patterns, True)

    @disable_random()
    def test_grappler_cfg(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 30, 30, 1], name="input")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [2, 2, 1, 1], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv_bias = tf.compat.v1.get_variable("bias", [1], initializer=tf.compat.v1.random_normal_initializer())

        x = tf.nn.relu(x)
        conv = tf.nn.conv2d(x, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name="last")
        normed = (
            tf.keras.layers.BatchNormalization()(conv)
            if version1_gte_version2(tf.__version__, "2.16.1")
            else tf.compat.v1.layers.batch_normalization(conv)
        )

        relu = tf.nn.relu(normed)
        relu2 = tf.nn.relu(relu)
        pool = tf.nn.max_pool(relu2, ksize=1, strides=[1, 2, 2, 1], name="maxpool", padding="SAME")
        conv1 = tf.nn.conv2d(pool, conv_weights, strides=[1, 2, 2, 1], padding="SAME", name="last")
        conv_bias = tf.nn.bias_add(conv1, conv_bias)
        x = tf.nn.relu(conv_bias)
        final_node = tf.nn.relu(x, name="op_to_store")

        out_name = final_node.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            fp32_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.tensorflow import quantize_model
            from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

            dataset = DummyDataset(shape=(100, 30, 30, 1), label=True)
            calib_dataloader = BaseDataLoader(dataset)
            quant_config = {
                "static_quant": {
                    "global": {
                        "weight_dtype": "int8",
                        "weight_sym": True,
                        "weight_granularity": "per_tensor",
                        "weight_algorithm": "minmax",
                    },
                }
            }
            qmodel = quantize_model(fp32_graph_def, quant_config, calib_dataloader)

            disable_arithmetic = False
            for i in qmodel.graph_def.node:
                if i.name == "maxpool_eightbit_quantize_Relu_2" and i.input[0] == "Relu_2":
                    disable_arithmetic = True

            self.assertEqual(True, disable_arithmetic)


class TestFrameworkQueryYaml(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_framework_yaml()
        self.tf_yaml_path = os.path.join(os.getcwd() + "/fake_framework.yaml")

        with open(self.tf_yaml_path) as f:
            self.content = yaml.safe_load(f)
        self.query_handler = TensorflowQuery(local_config_file=self.tf_yaml_path)

    @classmethod
    def tearDownClass(self):
        os.remove("fake_framework.yaml")

    def test_version_fallback(self):
        if self.query_handler.version >= "2.1.0":
            self.assertEqual(True, "Conv2D" in self.query_handler.get_op_types()["int8"])
        else:
            self.assertEqual(True, "BatchMatMul" in self.query_handler.get_op_types()["int8"])


if __name__ == "__main__":
    unittest.main()
