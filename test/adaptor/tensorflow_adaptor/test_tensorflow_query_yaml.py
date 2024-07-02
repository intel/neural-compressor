#
#  -*- coding: utf-8 -*-
#
import os
import unittest

import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

from neural_compressor.adaptor.tensorflow import TensorflowQuery
from neural_compressor.adaptor.tf_utils.util import disable_random


def build_fake_yaml_on_grappler():
    fake_yaml = """
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          recipes:
            scale_propagation_concat: False
          model_wise:
            weight:
                granularity: per_tensor
                scheme: sym
                dtype: int8
                algorithm: minmax
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: basic
            accuracy_criterion:
              relative: 0.1
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        """
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("fake_yaml_grappler.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


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
        self.tf_yaml_path = os.path.join(os.getcwd() + "/../neural_compressor/adaptor/tensorflow.yaml")

        with open(self.tf_yaml_path) as f:
            self.content = yaml.safe_load(f)
        self.query_handler = TensorflowQuery(local_config_file=self.tf_yaml_path)
        build_fake_yaml_on_grappler()

    @classmethod
    def tearDownClass(self):
        os.remove("fake_yaml_grappler.yaml")

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
