import unittest

import torch
import torch.nn as nn
import torchvision

from neural_compressor.conf.pythonic_config import Config, WeightPruningConfig
from neural_compressor.data import Datasets
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.experimental.pruning_v2 import Pruning


class TestPruningPatterns(unittest.TestCase):
    model = torchvision.models.resnet18()

    def test_pruning_pattern(self):
        local_configs = [
            {"op_names": ["layer1.*"], "target_sparsity": 0.5, "pattern": "5:8", "pruning_type": "magnitude"},
            {"op_names": ["layer2.*"], "pattern": "1xchannel", "pruning_scope": "global"},
            {
                "start_step": 2,
                "end_step": 20,
                "op_names": ["layer3.*"],
                "target_sparsity": 0.666666,
                "pattern": "4x2",
                "pruning_type": "snip_progressive",
                "pruning_frequency": 5,
            },
        ]
        conf = WeightPruningConfig(
            local_configs,
            target_sparsity=0.8,
            sparsity_decay_type="cos",
            excluded_op_names=["downsample.*"],
            pruning_scope="local",
            min_sparsity_ratio_per_op=0.1,
        )
        config = Config(quantization=None, benchmark=None, pruning=conf, distillation=None)
        prune = Pruning(config)
        prune.update_config(start_step=1, end_step=10)
        prune.model = self.model

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(10, 3, 224, 224), low=0.0, high=1.0, label=True)
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
