import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from neural_compressor.data import DATASETS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader

def build_fake_yaml():
    fake_yaml = """
    model:
      name: imagenet_prune
      framework: pytorch

    pruning:
      approach:
        weight_compression_pytorch:
          initial_sparsity: 0.0
          target_sparsity: 0.9
          start_step: 0
          end_step: 10
          excluded_names: ["classifier"]
              
          update_frequency_on_step: 1
          sparsity_decay_type: "exp"
          pruners:
            - !Pruner
                start_step: 0
                end_step: 10
                prune_type: "magnitude"
                names: ['layer1.*']
                extra_excluded_names: ['layer2.*']   
                prune_domain: "global"
                pattern: "tile_pattern_4x1"

            - !Pruner
                start_step: 1
                end_step: 1
                target_sparsity: 0.5
                prune_type: "snip_momentum"
                update_frequency: 2
                names: ['layer2.*']
                prune_domain: local
                pattern: "tile_pattern_2:4"
            - !Pruner
                start_step: 2
                end_step: 8
                target_sparsity: 0.8
                prune_type: "snip"
                names: ['layer3.*']
                prune_domain: "local"
                pattern: "tile_pattern_16x1"
                sparsity_decay_type: "cube"
    """
    with open('fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)



class TestPytorchPruning(unittest.TestCase):

    model = torchvision.models.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_pytorch_pruning(self):
        from neural_compressor.experimental.pytorch_pruner.pruning import Pruning
        prune = Pruning('fake.yaml')
        ##prune.generate_pruners()

        prune.model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        prune.on_train_begin()
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


if __name__ == "__main__":
    unittest.main()
