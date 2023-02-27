import os
import shutil
import unittest

import torchvision

from neural_compressor.pruner.pruning import Pruning


def build_fake_yaml():
    fake_snip_yaml = """
    model:
      name: imagenet_prune
      framework: pytorch
    pruning:
      approach:
        weight_compression_v2:
          target_sparsity: 0.9
          start_step: 0
          end_step: 10
          excluded_op_names: ["classifier"]
          max_sparsity_ratio_per_op: 0.95
          pruning_frequency: 2
          pruners:
            - !PrunerV2
                target_sparsity: 0.3
                start_step: 0
                sparsity_decay_type: "cos"
                end_step: 2
                pruning_type: "magnitude"
                op_names: ['layer1.*']
                pruning_scope: "global"
                pattern: "3:5"
            - !PrunerV2
                start_step: 1
                end_step: 1
                target_sparsity: 0.5
                pruning_type: "snip_momentum"
                pruning_frequency: 2
                op_names: ['layer2.*']
                pruning_scope: "local"
                pattern: "4x1"
                reg_type: "group_lasso"
    """
    with open('fake_snip.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_snip_yaml)



class TestPytorchPruning(unittest.TestCase):
    model = torchvision.models.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake_snip.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_pruning_yaml_config(self):
        prune = Pruning("fake_snip.yaml")
        ##prune.generate_pruners()
        prune.update_config(start_step=1)
        prune.model = self.model
        prune.on_train_begin()
        assert prune.pruners[0].config['start_step'] == 1
        assert prune.pruners[0].config['sparsity_decay_type'] == 'cos'
        assert prune.pruners[0].config['end_step'] == 2
        assert prune.pruners[0].config['pruning_type'] == 'magnitude'
        assert prune.pruners[0].config['op_names'] == ['layer1.*']
        assert prune.pruners[0].config['pruning_scope'] == 'global'
        assert prune.pruners[0].config['pattern'] == '3:5'
        assert prune.pruners[0].config['excluded_op_names'] == ["classifier"]
        ##the following value is changed by pruning code
        assert prune.pruners[0].config['max_sparsity_ratio_per_op'] == 0.6
        assert prune.pruners[0].config['pruning_frequency'] == 2
        assert prune.pruners[1].config['target_sparsity'] == 0.5
        assert prune.pruners[1].config['pruning_scope'] == "local"
        assert prune.pruners[1].config['pattern'] == "4x1"
        assert prune.pruners[1].config['reg_type'] == "group_lasso"


if __name__ == "__main__":
    unittest.main()
