#
#  -*- coding: utf-8 -*-
#
import unittest
import yaml
import os

from lpot.adaptor.tensorflow import TensorflowQuery


class TestTFQueryYaml(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tf_yaml_path = os.path.join(os.getcwd() + "/../lpot/adaptor/tensorflow.yaml")

        with open(self.tf_yaml_path) as f:
            self.content = yaml.safe_load(f)
        self.query_handler = TensorflowQuery(local_config_file=self.tf_yaml_path)

    def test_unique_version(self):
        registered_version_name = [i['version']['name'] for i in self.content]

        self.assertEqual(len(registered_version_name), len(set(registered_version_name)))

    def test_int8_sequences(self):
        patterns = self.query_handler.get_eightbit_patterns()

        has_conv2d = bool('Conv2D' in patterns)
        has_matmul = bool('MatMul' in patterns)
        self.assertEqual(has_conv2d, True)
        self.assertEqual(has_matmul, True)
        self.assertGreaterEqual(len(patterns['Conv2D']), 13)
        self.assertGreaterEqual(len(patterns['MatMul']), 3)
        self.assertEqual(len(patterns['ConcatV2']), 1)
        self.assertEqual(len(patterns['MaxPool']), 1)
        self.assertEqual(len(patterns['AvgPool']), 1)

    def test_convert_internal_patterns(self):
        internal_patterns = self.query_handler.generate_internal_patterns()
        self.assertEqual([['MaxPool']] in internal_patterns, True)
        self.assertEqual([['ConcatV2']] in internal_patterns, True)
        self.assertEqual([['AvgPool']] in internal_patterns, True)
        self.assertEqual([['MatMul'], ('BiasAdd',), ('Relu',)] in internal_patterns, True)

if __name__ == '__main__':
    unittest.main()
