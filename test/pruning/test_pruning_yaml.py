import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from neural_compressor.data import Datasets
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.pruner.pruning import Pruning


def build_fake_yaml_basic():
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
          pruning_frequency: 1 
          sparsity_decay_type: "cos"
          pruners:
            - !PrunerV2
                start_step: 0      
                end_step: 10
                pruning_type: "magnitude"
                op_names: ['layer1.*']
                pruning_scope: "global"
                pattern: "4x1"
                parameters: {"reg_coeff":0.1}

            - !PrunerV2
                start_step: 1
                end_step: 1
                target_sparsity: 0.5
                pruning_type: "snip_momentum"
                pruning_frequency: 2
                op_names: ['layer2.*']
                pruning_scope: local
                pattern: "2:4"
                sparsity_decay_type: "exp"
            - !PrunerV2
                start_step: 2
                end_step: 8
                target_sparsity: 0.8
                pruning_type: "snip"
                op_names: ['layer3.*']
                pruning_scope: "local"
                pattern: "16x1"
                sparsity_decay_type: "cube"
    """
    with open('fake_snip.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_snip_yaml)



class TestPruningYaml(unittest.TestCase):
    model = torchvision.models.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml_basic()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake_snip.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_pruning_yaml(self):
        prune = Pruning("fake_snip.yaml")
        prune.update_config(start_step=1)
        prune.model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        prune.on_train_begin()
        prune.update_config(prune_frequency=1)
        for epoch in range(2):
            self.model.train()
            prune.on_epoch_begin(epoch)
            local_step = 0
            for image, target in dummy_dataloader:
                prune.on_step_begin(local_step)
                output = self.model(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                prune.on_before_optimizer_step()
                optimizer.step()
                prune.on_after_optimizer_step()
                prune.on_step_end()
                local_step += 1

            prune.on_epoch_end()
        prune.get_sparsity_ratio()
        prune.on_train_end()
        prune.on_before_eval()
        prune.on_after_eval()


if __name__ == "__main__":
    unittest.main()
