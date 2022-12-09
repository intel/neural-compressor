import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from neural_compressor.data import DATASETS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.pruning import Pruning
from neural_compressor.config import WeightPruningConfig


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
          pruning_frequency: 3
          sparsity_decay_type: "exp"
          pruners:
            - !Pruner
                start_step: 0
                end_step: 10
                pruning_type: "magnitude"
                names: ['layer1.*']
                extra_excluded_op_names: ['layer2.*']
                pruning_scope: "global"
                target_sparsity: 0.5
                pattern: "5:8"

            - !Pruner
                start_step: 1
                end_step: 1
                target_sparsity: 0.5
                pruning_type: "snip_momentum"
                pruning_frequency: 2
                names: ['layer2.*']
                pruning_scope: local
                target_sparsity: 0.75
                pattern: "channelx1"
                sparsity_decay_type: "exp"

            - !Pruner
                start_step: 2
                end_step: 8
                target_sparsity: 0.8
                pruning_type: "snip"
                names: ['layer3.*']
                pruning_scope: "local"
                pattern: "8x2"
                sparsity_decay_type: "cube"
            - !Pruner
                start_step: 2
                end_step: 8
                target_sparsity: 0.1
                pruning_type: "gradient"
                names: ['fc']
                pruning_scope: "local"
                pattern: "1xchannel"
                sparsity_decay_type: "cube"

    """
    with open('fake_snip.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_snip_yaml)

class TestPruningPatterns(unittest.TestCase):
    model = torchvision.models.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake_snip.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_pruning_pattern(self):
        local_configs = [
            {
                "op_names": ['layer1.*'], 
                'target_sparsity': 0.5,
                "pattern": '5:8',
                "pruning_type": "magnitude"
            },
            {
                "op_names": ['layer2.*'],
                "pattern": '1xchannel',
                "pruning_scope": "global"
            },
            {
                "start_step": 2,
                "end_step": 20,
                "op_names": ['layer3.*'],
                'target_sparsity': 0.666666, 
                'pattern': '4x2',
                "pruning_type": "snip_progressive",
                "pruning_frequency": 5
            }
        ]
        config = WeightPruningConfig(
            local_configs, 
            target_sparsity=0.8,
            sparsity_decay_type="cos",
            excluded_op_names=["downsample.*"],
            pruning_scope= "local",
            min_sparsity_ratio_per_op=0.1
        )
        prune = Pruning(config)
        prune.update_config(start_step=1, end_step=10)
        prune.model = self.model

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        prune.on_train_begin()
        for epoch in range(5):
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


