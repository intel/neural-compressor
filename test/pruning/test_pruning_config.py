import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from neural_compressor.data import DATASETS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.pruning import Pruning

def build_fake_yaml_basic():
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
          excluded_names: ["classifier"]
          max_layer_sparsity_ratio: 0.95
          prune_frequency: 1
        #   sparsity_decay_type: "cos"
          pruners:
            - !Pruner
                start_step: 0
                sparsity_decay_type: "cos"
                end_step: 10
                prune_type: "magnitude"
                names: ['layer1.*']
                extra_excluded_names: ['layer2.*']
                prune_domain: "global"
                pattern: "4x1"

            - !Pruner
                start_step: 1
                end_step: 1
                target_sparsity: 0.5
                prune_type: "snip_momentum"
                prune_frequency: 2
                names: ['layer2.*']
                prune_domain: local
                pattern: "2:4"

            - !Pruner
                start_step: 2
                end_step: 8
                target_sparsity: 0.8
                prune_type: "snip"
                names: ['layer3.*']
                prune_domain: "local"
                pattern: "16x1"
                sparsity_decay_type: "cube"

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
            weight_compression:
              target_sparsity: 0.9
              start_step: 0
              end_step: 10
              excluded_names: ["classifier"]
              prune_frequency: 1
              sparsity_decay_type: "cos"
              pruners:
                - !Pruner
                    start_step: 5
                    end_step: 5
                    prune_type: "pattern_lock"
                    names: ['layer1.*']
                    extra_excluded_names: ['layer2.*']
                    prune_domain: "global"
                    pattern: "channelx1"

                - !Pruner
                    start_step: 1
                    end_step: 1
                    target_sparsity: 0.5
                    prune_type: "pattern_lock"
                    prune_frequency: 2
                    names: ['layer2.*']
                    prune_domain: local
                    pattern: "2:4"

                - !Pruner
                    start_step: 2
                    end_step: 8
                    target_sparsity: 0.8
                    prune_type: "snip"
                    names: ['layer3.*']
                    prune_domain: "local"
                    pattern: "1xchannel"

        """

    with open('fake_channel_pruning.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_channel_pruning_yaml)


class TestPytorchPruning(unittest.TestCase):

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
        prune.update_config(start_step=1)
        prune.model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        prune.on_train_begin()
        prune.update_config(reg_type='group_lasso', parameters={"reg_coeff": 1.1},\
                            max_layer_sparsity_ratio=0.99, min_layer_sparsity_ratio=0.71, \
                            sparsity_decay_type='cube', prune_frequency=2, prune_domain='local', target_sparsity=0.88)
        assert prune.pruners[0].config['prune_frequency'] == 2, "prune_frequency should be 1"
        assert prune.pruners[0].config['prune_domain'] == 'local', "prune_domain should be local"
        assert prune.pruners[0].config['target_sparsity'] == 0.88, "target_sparsity should be 0.88"
        assert prune.pruners[0].config['reg_type'] == 'group_lasso', "reg_type should be group_lasso"
        assert prune.pruners[0].config['sparsity_decay_type'] == "cube", "sparsity_decay_type should be cube"
        assert prune.pruners[0].config['min_layer_sparsity_ratio'] == 0.71, "min_layer_sparsity_ratio should be 0.71"
        assert prune.pruners[0].config['max_layer_sparsity_ratio'] == 0.99, "max_layer_sparsity_ratio should be 0.99"
        assert math.isclose(prune.pruners[0].config['reg_coeff'], 1.1), \
                                                f"reg_coeff should be 1.1 not {prune.pruners[0].config['reg_coeff']}"
        
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

    def test_pytorch_pruner_channel_pruning(self):
        prune = Pruning("fake_channel_pruning.yaml")
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


