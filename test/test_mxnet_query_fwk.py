#
#  -*- coding: utf-8 -*-
#
import os
import sys
import unittest
import yaml
sys.path.append('..')

import mxnet as mx
import lpot

class TestMXNetQuery(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.yaml_path = os.path.join(os.getcwd() + "/../lpot/adaptor/mxnet.yaml")
        self.Queryhandler = lpot.adaptor.mxnet.MXNetQuery(self.yaml_path)
        self.version = mx.__version__

    def test_get_specified_version_cfg(self):
        with open(self.yaml_path) as f:
            content = yaml.safe_load(f)
            default_config = self.Queryhandler._get_specified_version_cfg(content)
            self.assertIsNotNone(default_config)

    def test_one_shot_query(self):
        self.Queryhandler._one_shot_query()
        self.assertIsNotNone(self.Queryhandler.cur_config)

    def test_get_version(self):
        Query_version = self.Queryhandler.get_version()
        self.assertEqual(mx.__version__, Query_version)

    def test_get_precisions(self):
        Query_precisions = self.Queryhandler.get_precisions()
        res = Query_precisions.split(',')
        self.assertEqual(len(res), len(set(res)))

    def test_get_op_types(self):
        Query_op_types = self.Queryhandler.get_op_types()
        self.assertEqual(len(Query_op_types), len(set(Query_op_types)))

    def test_get_fuse_patterns(self):
        Query_fusion_pattern = self.Queryhandler.get_fuse_patterns()
        self.assertEqual(len(Query_fusion_pattern), len(set(Query_fusion_pattern)))

    def test_get_quantization_capability(self):
        Query_quantization_capability = self.Queryhandler.get_quantization_capability()
        self.assertIsNotNone(Query_quantization_capability)

    def test_get_mixed_precision_combination(self):
        Query_mixed_precision = self.Queryhandler.get_mixed_precision_combination()
        self.assertNotIn(['int8', 'bf16'], Query_mixed_precision)

if __name__ == '__main__':
    unittest.main()
