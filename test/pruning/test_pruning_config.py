import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from neural_compressor.pruning import Pruning
from neural_compressor.config import WeightPruningConfig, GlobalPrunerConfig, LocalPruningConfig


def build_fake_yaml():
    fake_snip_yaml = """
    model:
      name: imagenet_prune
      framework: pytorch

    pruning:
      approach:
        weight_compression:
          target_sparsity: 0.9
          start_step: 0
          end_step: 10
          excluded_op_names: ["classifier"]
          max_sparsity_ratio_per_op: 0.95
          pruning_frequency: 1
          pruners:
            - !Pruner
                start_step: 0
                sparsity_decay_type: "cos"
                end_step: 10
                pruning_type: "magnitude"
                names: ['layer1.*']
                extra_excluded_op_names: ['layer2.*']
                pruning_scope: "global"
                pattern: "4x1"
                target_sparsity: 0.88

            - !Pruner
                start_step: 1
                end_step: 1
                target_sparsity: 0.5
                pruning_type: "snip_momentum"
                pruning_frequency: 2
                names: ['layer2.*']
                pruning_scope: local
                pattern: "2:4"

            - !Pruner
                start_step: 2
                end_step: 8
                target_sparsity: 0.8
                pruning_type: "snip"
                names: ['layer3.*']
                pruning_scope: "local"
                pattern: "16x1"
                sparsity_decay_type: "cube"

    """
    with open('fake_snip.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_snip_yaml)


class TestPytorchPruning(unittest.TestCase):
    model = torch.nn.Module()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake_snip.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)



    def test_pruning_class_config(self):
        dummy_pruner1 = Pruner(extra_excluded_op_names=["layer1"], reg_type="group_lasso", max_sparsity_ratio_per_op=0.9)
        dummy_pruner2 = Pruner(pruning_scope="local", criterion_reduce_type="max", target_sparsity=0.85)
        config = PruningConfig([dummy_pruner1, dummy_pruner2], target_sparsity=0.8, end_step=1)
        prune = Pruning(config)
        prune.model = self.model
        prune.on_train_begin()
        assert prune.pruners[0].config['extra_excluded_op_names'] == ["layer1"]
        assert prune.pruners[0].config['reg_type'] == "group_lasso"
        assert prune.pruners[0].config['max_sparsity_ratio_per_op'] == 0.9
        assert prune.pruners[0].config['target_sparsity'] == 0.8
        assert prune.pruners[0].config['end_step'] == 1
        assert prune.pruners[1].config["pruning_scope"] == "local"
        assert prune.pruners[1].config["criterion_reduce_type"] == "max"
        assert prune.pruners[1].config["target_sparsity"] == 0.85
        assert prune.pruners[1].config['end_step'] == 1


if __name__ == "__main__":
    unittest.main()
