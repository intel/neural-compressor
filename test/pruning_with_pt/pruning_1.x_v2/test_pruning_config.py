import unittest

import torch
import torch.nn as nn
import torchvision

from neural_compressor.conf.pythonic_config import Config, WeightPruningConfig
from neural_compressor.data import Datasets
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.experimental.pruning_v2 import Pruning


class TestPytorchPruning(unittest.TestCase):
    model = torchvision.models.resnet18()

    def test_pruning_class_config(self):
        local_configs = [
            {
                "op_names": ["layer1.*", "layer2.*"],
                "excluded_op_names": ["downsample.*"],
                "target_sparsity": 0.6,
                "pattern": "channelx1",
                "pruning_type": "snip_progressive",
                "pruning_scope": "local",
                "start_step": 0,
                "end_step": 10,
            },
            {"op_names": ["layer3.*"], "pruning_type": "pattern_lock"},
        ]
        conf = WeightPruningConfig(
            local_configs,
            pruning_frequency=2,
            target_sparsity=0.8,
        )
        config = Config(quantization=None, benchmark=None, pruning=conf, distillation=None)
        prune = Pruning(config)
        prune.model = self.model

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(12, 3, 224, 224), low=0.0, high=1.0, label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        prune.update_config(pruning_frequency=4)
        prune.on_train_begin()
        assert prune.pruners[0].config["pruning_frequency"] == 4
        assert prune.pruners[0].config["target_sparsity"] == 0.6
        assert prune.pruners[1].config["target_sparsity"] == 0.8
        assert prune.pruners[0].config["pattern"] == "channelx1"
        assert prune.pruners[1].config["pruning_type"] == "pattern_lock"

        for epoch in range(1):
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
