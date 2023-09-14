import unittest

import torch
import torch.nn as nn
import torchvision

from neural_compressor import WeightPruningConfig
from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader


class TestPruning(unittest.TestCase):
    model = torchvision.models.resnet18()

    def test_pruning_basic(self):
        local_configs = [
            {
                "op_names": ["layer1.*"],
                "target_sparsity": 0.6,
                "pattern": "8x2",
                "pruning_type": "magnitude_progressive",
                "false_key": "this is to test unsupported keys",
            },
            {
                "op_names": ["layer2.*"],
                "pruning_type": "snip_momentum",
                "target_sparsity": 0.5,
                "pruning_scope": "local",
                "pattern": "2:4",
            },
        ]
        config = WeightPruningConfig(local_configs, target_sparsity=0.8, start_step=1, end_step=4)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(10, 3, 224, 224), low=0.0, high=1.0, label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        from neural_compressor.compression.pruner import prepare_pruning

        pruning = prepare_pruning(self.model, config, optimizer)

        for epoch in range(4):
            self.model.train()
            local_step = 0
            for image, target in dummy_dataloader:
                output = self.model(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                local_step += 1

        assert self.model is not None


if __name__ == "__main__":
    unittest.main()
