import unittest

import torch
import torchvision
import torch.nn as nn
import sys
sys.path.insert(0, '../../pruning_2.x/')
from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor import WeightPruningConfig
from neural_compressor.training import prepare_compression
from neural_compressor.utils import logger

class TestPytorchPruning(unittest.TestCase):
    model = torchvision.models.resnet18()

    def test_pruning_class_config(self):
        local_configs = [
            {
                "op_names": ['layer1.*', 'layer2.*'],
                "excluded_op_names": ['downsample.*'],
                'target_sparsity': 0.6,
                "pattern": 'channelx1',
                "pruning_type": "snip_progressive",
                "pruning_scope": "local",
                "start_step": 0,
                "end_step": 10
            },
            {
                "op_names": ['layer3.*'],
                "pruning_type": "pattern_lock"
            }
        ]
        config = WeightPruningConfig(
            local_configs,
            pruning_frequency=2,
            target_sparsity=0.8,
        )
        compression_manager = prepare_compression(model=self.model, confs=config)
        compression_manager.callbacks.on_train_begin()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(12, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        logger.info(compression_manager.callbacks.callbacks_list[0].pruners)
        assert compression_manager.callbacks.callbacks_list[0].pruners[0].config['pruning_frequency'] == 2
        assert compression_manager.callbacks.callbacks_list[0].pruners[0].config['target_sparsity'] == 0.6
        assert compression_manager.callbacks.callbacks_list[0].pruners[1].config['target_sparsity'] == 0.8
        assert compression_manager.callbacks.callbacks_list[0].pruners[0].config['pattern'] == "channelx1"
        assert compression_manager.callbacks.callbacks_list[0].pruners[1].config['pruning_type'] == 'pattern_lock'

        for epoch in range(1):
            self.model.train()
            compression_manager.callbacks.on_epoch_begin(epoch)
            local_step = 0
            for image, target in dummy_dataloader:
                compression_manager.callbacks.on_step_begin(local_step)
                output = self.model(image)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                compression_manager.callbacks.on_before_optimizer_step()
                optimizer.step()
                compression_manager.callbacks.on_after_optimizer_step()
                compression_manager.callbacks.on_step_end()
                local_step += 1

            compression_manager.callbacks.on_epoch_end()
        compression_manager.callbacks.on_train_end()
        compression_manager.callbacks.on_before_eval()
        compression_manager.callbacks.on_after_eval()


if __name__ == "__main__":
    unittest.main()
