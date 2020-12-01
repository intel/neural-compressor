#
#  -*- coding: utf-8 -*-
#
import unittest
import yaml
import os

from ilit.adaptor.tensorflow import TensorflowQuery

class TestTFQueryYaml(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tf_yaml_path = os.path.join(os.getcwd() + "/../ilit/adaptor/tensorflow.yaml")

        with open(self.tf_yaml_path) as f:
            self.content = yaml.safe_load(f)

    def test_unique_version(self):
        registered_version_name = [i['version']['name'] for i in self.content]

        self.assertEqual(len(registered_version_name), len(set(registered_version_name)))
    
    def test_model_wise_cfg(self):
        self.query_handler = TensorflowQuery(local_config_file=self.tf_yaml_path)
        model_wise_cfg = self.query_handler.get_model_wise_ability()

        conv2d_weigths_granularity = model_wise_cfg['weight']['granularity']
        self.assertEqual(conv2d_weigths_granularity, ['per_channel', 'per_tensor'])

if __name__ == '__main__':
    unittest.main()
