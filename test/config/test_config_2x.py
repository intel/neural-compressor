"""Tests for 2.x config file."""

import unittest

from neural_compressor import set_random_seed, set_resume_from, set_tensorboard, set_workspace
from neural_compressor.config import BenchmarkConfig, MixedPrecisionConfig, MXNet, PostTrainingQuantConfig
from neural_compressor.config import _Config as conf
from neural_compressor.config import options
from neural_compressor.utils.constant import *


def helper(content):
    with open("fake_conf.yaml", "w", encoding="utf-8") as f:
        f.write(content)


class TestConfig(unittest.TestCase):
    def test_config(self):
        config = PostTrainingQuantConfig()
        self.assertEqual(config.recipes["smooth_quant"], False)
        self.assertEqual(config.recipes["fast_bias_correction"], False)
        self.assertEqual(config.recipes["weight_correction"], False)
        self.assertEqual(config.recipes["dedicated_qdq_pair"], False)
        self.assertEqual(config.recipes["add_qdq_pair_to_weight"], False)
        self.assertEqual(config.recipes["graph_optimization_level"], None)


class TestGeneralConf(unittest.TestCase):
    def test_config(self):
        cfg = PostTrainingQuantConfig()
        cfg.op_type_dict = {"Conv": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}}
        cfg.op_name_dict = {
            "layer1.0.conv1": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}},
        }
        a = conf(quantization=cfg)
        self.assertEqual(a.quantization.op_type_dict["Conv"]["weight"]["dtype"], ["fp32"])

        cfg = BenchmarkConfig()
        cfg.cores_per_instance = 4
        cfg.iteration = 100
        cfg.num_of_instance = 7
        a = conf(benchmark=cfg)
        self.assertEqual(a.benchmark.iteration, 100)

        cfg = MixedPrecisionConfig()
        a = conf(mixed_precision=cfg)
        self.assertEqual(a.mixed_precision.precisions, ["bf16"])

        cfg = MXNet()
        cfg.precisions = "bf16"
        a = conf(mxnet=cfg)
        self.assertEqual(a.mxnet.precisions, ["bf16"])

        set_workspace("workspace_path")
        self.assertEqual(options.workspace, "workspace_path")

        set_random_seed(1)
        self.assertEqual(options.random_seed, 1)

        tmp_resume_from = options.resume_from
        set_resume_from("resume_from_path")
        self.assertEqual(options.resume_from, "resume_from_path")
        set_resume_from(tmp_resume_from)

        tmp_tensorboard = options.tensorboard
        set_tensorboard(True)
        self.assertEqual(options.tensorboard, True)
        set_tensorboard(tmp_tensorboard)


if __name__ == "__main__":
    unittest.main()
