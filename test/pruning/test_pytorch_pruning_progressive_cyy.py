import sys
sys.path.append("./")
import os
import logging 
import shutil
import unittest
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from neural_compressor.experimental.pruning import Pruning

def build_fake_yaml_basic():
    fake_snip_yaml = """
    model:
      name: imagenet_prune
      framework: pytorch
    pruning:
      approach:
        weight_compression_pytorch:
          initial_sparsity: 0.0
          target_sparsity: 0.9
          start_step: 0
          end_step: 40
          excluded_names: ["classifier", "fp32"]
          update_frequency_on_step: 8
          sparsity_decay_type: "exp"
          pruners:
            - !Pruner
                start_step: 0
                sparsity_decay_type: "cos"
                end_step: 40
                prune_type: "magnitude_progressive"
                names: ['layer1.*']
                extra_excluded_names: ['layer2.*']
                prune_domain: "local"
                pattern: "tile_pattern_4x1"
            - !Pruner
                start_step: 0
                end_step: 40
                target_sparsity: 0.8
                prune_type: "snip_momentum_progressive"
                names: ['layer2.*']
                prune_domain: "global"
                pattern: "tile_pattern_16x1"
    """
    with open('fake_snip.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_snip_yaml)

def build_fake_yaml_channel():
    fake_channel_pruning_yaml = """
        model:
          name: imagenet_prune
          framework: pytorch
        pruning:
          approach:
            weight_compression_pytorch:
              initial_sparsity: 0.0
              target_sparsity: 0.9
              start_step: 0
              end_step: 40
              excluded_names: ["classifier", "fp32"]
              update_frequency_on_step: 8
              sparsity_decay_type: "exp"
              pruners:
                - !Pruner
                    start_step: 2
                    end_step: 38
                    prune_type: "magnitude_progressive"
                    names: ['layer1.*']
                    extra_excluded_names: ['layer2.*']
                    prune_domain: "local"
                    pattern: "channelx1"
                - !Pruner
                    start_step: 1
                    end_step: 1
                    target_sparsity: 0.5
                    prune_type: "pattern_lock"
                    update_frequency: 2
                    names: ['layer2.*']
                    prune_domain: local
                    pattern: "2:4"
                - !Pruner
                    start_step: 2
                    end_step: 38
                    target_sparsity: 0.8
                    prune_type: "snip_progressive"
                    names: ['layer3.*']
                    prune_domain: "global"
                    pattern: "1xchannel"
                    sparsity_decay_type: "cube"
        """

    with open('fake_channel_pruning.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_channel_pruning_yaml)


class TestPytorchPruning(unittest.TestCase):
 
    def __init__(self, *args, **kwargs):
        super(TestPytorchPruning, self).__init__(*args, **kwargs)
        LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper() 
        self._logger = logging.getLogger()
        self._logger.handlers.clear()
        self._logger.setLevel(LOGLEVEL)
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            "%Y-%m-%d %H:%M:%S")
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self._logger.addHandler(streamHandler)
        self._logger.propagate = False
    
    model = torchvision.models.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml_basic()
        build_fake_yaml_channel()


    @classmethod
    def tearDownClass(cls):
        os.remove('fake_channel_pruning.yaml')
        os.remove('fake_snip.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_pytorch_pruning_basic(self):
        prune = Pruning("fake_snip.yaml")
        ##prune.generate_pruners()
        prune.update_items_for_all_pruners(start_step=1)
        prune.model = self.model
        prune.prepare()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        try:
            from neural_compressor.experimental.data.datasets.dummy_dataset import DummyDataset
            dummy_dataset = DummyDataset(shape=(10, 3, 224, 224), low=0., high=1., label=True)
        except:
            x_train = np.random.uniform(low=0., high=1., size=tuple([10, 3, 224, 224]))
            y_train = np.random.randint(low=0, high=2, size=tuple([10]))
            x_train, y_train = torch.tensor(x_train, dtype=torch.float32), \
                                torch.tensor(y_train, dtype=torch.long)
            dummy_dataset = TensorDataset(x_train, y_train)
        dummy_dataloader = DataLoader(dummy_dataset)
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
        #prune.get_sparsity_ratio()
        prune.on_train_end()
        prune.on_before_eval()
        prune.on_after_eval()

    def test_pytorch_pruner_channel_pruning(self):
        prune = Pruning("fake_channel_pruning.yaml")
        ##prune.generate_pruners()
        prune.update_items_for_all_pruners(start_step=1)
        prune.model = self.model
        prune.prepare()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        try:
            from neural_compressor.experimental.data.datasets.dummy_dataset import DummyDataset
            dummy_dataset = DummyDataset(shape=(100, 3, 224, 224), low=0., high=1., label=True)
        except:
            x_train = np.random.uniform(low=0., high=1., size=tuple([100, 3, 224, 224]))
            y_train = np.random.randint(low=0, high=1, size=tuple([100]))
            x_train, y_train = torch.tensor(x_train, dtype=torch.float32), \
                                torch.tensor(y_train, dtype=torch.long)
            dummy_dataset = TensorDataset(x_train, y_train)
        dummy_dataloader = DataLoader(dummy_dataset)
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
        prune.on_train_end()
        prune.on_before_eval()
        prune.on_after_eval()

if __name__ == "__main__":
    unittest.main()


