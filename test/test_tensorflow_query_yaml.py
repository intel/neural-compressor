#
#  -*- coding: utf-8 -*-
#
import unittest
import yaml
import os


class TestTFQueryYaml(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        tf_yaml_path = os.path.join(os.getcwd() + "/../ilit/adaptor/tensorflow.yaml")

        with open(tf_yaml_path) as f:
            self.content = yaml.safe_load(f)

    def test_unique_version(self):
        registered_version_name = [i['version']['name'] for i in self.content]

        self.assertEqual(len(registered_version_name), len(set(registered_version_name)))

if __name__ == '__main__':
    unittest.main()
