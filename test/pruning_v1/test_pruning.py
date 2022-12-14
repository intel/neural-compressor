import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from neural_compressor.config import Pruner, PruningConfig
from neural_compressor.data import Datasets
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.training import prepare_compression


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
          end_epoch: 2
          pruners:
            - !Pruner
                start_epoch: 1
                end_epoch: 2
                prune_type: basic_magnitude
                names: ['layer1.0.conv1.weight']

            - !Pruner
                target_sparsity: 0.6
                prune_type: basic_magnitude
                update_frequency: 2
                names: ['layer1.0.conv2.weight']
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

    def test_pruning(self):
        pruner1 = Pruner(start_epoch=1, end_epoch=2, names=['layer1.0.conv1.weight'])
        pruner2 = Pruner(target_sparsity=0.6, update_frequency=2, names=['layer1.0.conv2.weight'])
        conf = PruningConfig(pruners=[pruner1, pruner2], end_epoch=2)
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        compression_manager = prepare_compression(self.model, conf)
        model = compression_manager.model

        epochs = 2
        iters = 3
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        for nepoch in range(epochs):
            model.train()
            cnt = 0
            compression_manager.callbacks.on_epoch_begin(nepoch)
            for image, target in dummy_dataloader:
                compression_manager.callbacks.on_step_begin(cnt)
                print('.', end='')
                cnt += 1
                output = model(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                compression_manager.callbacks.on_step_end()
                if cnt >= iters:
                    break
            compression_manager.callbacks.on_epoch_end()

        model.save("./saved")

    def test_pruning_external(self):
        from neural_compressor.experimental import common
        from neural_compressor import Pruning
        from neural_compressor.conf.config import PruningConf
        pruners = [Pruner(1,3,names=['layer1.0.conv1.weight']),
            Pruner(target_sparsity=0.6,update_frequency=2,names=['layer1.0.conv2.weight'])]
        conf = PruningConfig(pruners)

        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        compression_manager = prepare_compression(self.model, conf)
        model = compression_manager.model

        epochs = 2
        iters = 3
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        for nepoch in range(epochs):
            model.train()
            cnt = 0
            compression_manager.callbacks.on_epoch_begin(nepoch)
            for image, target in dummy_dataloader:
                compression_manager.callbacks.on_step_begin(cnt)
                print('.', end='')
                cnt += 1
                output = model(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                compression_manager.callbacks.on_step_end()
                if cnt >= iters:
                    break
            compression_manager.callbacks.on_epoch_end()
        model.save("./saved")


if __name__ == "__main__":
    unittest.main()
