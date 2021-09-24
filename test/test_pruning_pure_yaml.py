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
      train:
        start_epoch: 0
        end_epoch: 4
        iteration: 10
        dataloader:
          batch_size: 30
          dataset:
            dummy:
              shape: [128, 3, 224, 224]
              label: True
        optimizer:
          SGD:
            learning_rate: 0.1
            momentum: 0.1
            nesterov: True
            weight_decay: 0.1
        criterion:
          CrossEntropyLoss:
            reduction: sum
      approach:
        weight_compression:
          initial_sparsity: 0.0
          target_sparsity: 0.97
          start_epoch: 0
          end_epoch: 4
          pruners:
            - !Pruner
                start_epoch: 1
                end_epoch: 3
                prune_type: basic_magnitude
                names: ['layer1.0.conv1.weight']

            - !Pruner
                target_sparsity: 0.6
                prune_type: gradient_sensitivity
                update_frequency: 2
                names: ['layer1.0.conv2.weight']
    evaluation:
      accuracy:
        metric:
          topk: 1
        dataloader:
          batch_size: 30
          dataset:
            dummy:
              shape: [128, 3, 224, 224]
              label: True
    """
    with open('fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)


class TestPruning(unittest.TestCase):

    model = torchvision.models.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_pruning_internal(self):
        from neural_compressor.experimental import Pruning, common
        prune = Pruning('fake.yaml')

        prune.model = common.Model(self.model)
        _ = prune()

if __name__ == "__main__":
    unittest.main()
