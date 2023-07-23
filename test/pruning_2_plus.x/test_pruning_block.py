import unittest

import torch
import torchvision
import torch.nn as nn
import sys
sys.path.insert(0, './')
from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor import WeightPruningConfig
from transformers import (AutoModelForCausalLM)


class TestPruning(unittest.TestCase):
    # model = torchvision.models.resnet18()
    model = torchvision.models.vit_b_16()
    def test_pruning_basic(self):
        local_configs = [
            {
                "op_names": ['encoder_layer_1.mlp*'],
                "target_sparsity": 0.95,
                "pattern": 'channelx2',
                "pruning_type": "block_mask",
                "pruning_scope": "global",
                "criterion_type": "block_mask",
                "pruning_op_types": "Linear",
            },
            {
                "op_names": ['encoder_layer_2.mlp*'],
                "target_sparsity": 0.5,
                "pattern": '32x32',
                "pruning_op_types": "Linear",
                "pruning_type": "block_mask",
                "pruning_scope": "local",
            },
        ]
        config = WeightPruningConfig(
            local_configs,
            target_sparsity=0.8,
            start_step=1,
            end_step=10
        )

        criterion = nn.CrossEntropyLoss()
        from neural_compressor.compression.pruner import prepare_pruning
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(20, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        pruning = prepare_pruning(config, self.model, optimizer)
        

        for epoch in range(2):
            self.model.train()
            local_step = 0
            for image, target in dummy_dataloader:
                output = self.model(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                local_step += 1


if __name__ == "__main__":
    unittest.main()



