"""Tests for 2.x config file"""
import unittest
import os
from neural_compressor.config import Config as conf
from neural_compressor.utils.constant import *
from neural_compressor.config import PostTrainingQuantConfig, BenchmarkConfig, Options
from neural_compressor.config import MixedPrecisionConfig, DotDict


def helper(content):
    with open('fake_conf.yaml', 'w', encoding="utf-8") as f:
        f.write(content)

class TestConfig(unittest.TestCase):
    def test_config(self):
        config = PostTrainingQuantConfig()
        self.assertEqual(config.recipes['smooth_quant'], False)
        self.assertEqual(config.recipes['fast_bias_correction'], False)
        self.assertEqual(config.recipes['weight_correction'], False)
        self.assertEqual(config.recipes['dedicated_qdq_pair'], False)
        self.assertEqual(config.recipes['add_qdq_pair_to_weight'], False)
        self.assertEqual(config.recipes['graph_optimization_level'], None)

class TestGeneralConf(unittest.TestCase):
    def test_config(self):
        cfg = PostTrainingQuantConfig()
        cfg.accuracy_criterion.tolerable_loss = 0.2
        a = conf(quantization=cfg)
        self.assertEqual(a.accuracy.tolerable_loss, 0.2)

        cfg.op_type_dict = {'Conv': {
                              'weight': {
                                  'dtype': ['fp32']},
                              'activation': {
                                  'dtype': ['fp32']}}
                            }
        cfg.op_name_dict = {"layer1.0.conv1": {
                              "activation": {
                                  "dtype": ["fp32"]},
                              "weight": {
                                  "dtype": ["fp32"]}},
                            }
        a = conf(quantization=cfg)
        self.assertEqual(a.quantization.op_type_dict['Conv']['weight']['dtype'], ['fp32'])
 
        cfg.tuning_criterion.strategy = 'mse'
        a = conf(quantization=cfg)
        self.assertEqual(a.tuning.strategy, 'mse')
        
        cfg = BenchmarkConfig()
        cfg.cores_per_instance = 4
        cfg.iteration = 100
        cfg.num_of_instance = 7
        a = conf(benchmark=cfg)
        self.assertEqual(a.benchmark.iteration, 100)

        cfg = Options()
        cfg.workspace = "workspace_path"
        a = conf(options=cfg)
        self.assertEqual(a.options.workspace, "workspace_path")

        cfg = MixedPrecisionConfig()
        a = conf(mixed_precision=cfg)
        self.assertEqual(a.mixed_precision.precision, ["bf16"])


if __name__ == "__main__":
    unittest.main()
