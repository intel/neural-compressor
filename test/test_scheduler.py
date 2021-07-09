import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from lpot.data import DATASETS
from lpot.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from lpot.experimental.scheduler import Scheduler

def build_fake_yaml():
    fake_yaml = """
    model:
      name: imagenet_prune
      framework: pytorch

    pruning:
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
                prune_type: basic_magnitude
                update_frequency: 2
                names: ['layer1.0.conv2.weight']
    evaluation:
      accuracy:
        metric:
          topk: 1
    """
    with open('fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

def build_fake_yaml2():
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
                start_epoch: 0
                end_epoch: 4
                target_sparsity: 0.6
                prune_type: basic_magnitude
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
    with open('fake2.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

class TestPruning(unittest.TestCase):

    model = torchvision.models.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()
        build_fake_yaml2()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        os.remove('fake2.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_pruning(self):
        from lpot.experimental import Pruning, common
        prune = Pruning('fake.yaml')
        scheduler = Scheduler()
        scheduler.model = common.Model(self.model)
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        def training_func_for_lpot(model):
            epochs = 16
            iters = 30
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                prune.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    prune.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    prune.on_batch_end()
                    if cnt >= iters:
                        break
                prune.on_epoch_end()

        prune.pruning_func = training_func_for_lpot
        prune.eval_dataloader = dummy_dataloader
        prune.train_dataloader = dummy_dataloader
        scheduler.append(prune)
        opt_model = scheduler()

    def test_pure_yaml_pruning(self):
        from lpot.experimental import Pruning, common
        prune = Pruning('fake2.yaml')
        scheduler = Scheduler()
        scheduler.model = common.Model(self.model)
        scheduler.append(prune)
        opt_model = scheduler()

if __name__ == "__main__":
    unittest.main()
