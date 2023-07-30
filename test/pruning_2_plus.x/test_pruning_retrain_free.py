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
    # model = AutoModelForCausalLM.from_pretrained(
    #         "facebook/opt-125m",
    #         )
    def test_pruning_basic(self):
        local_configs = [
            {
                "op_names": ['encoder_layer_3.mlp*'],
                'target_sparsity': 0.9,
                'pattern': 'channelx1',
                "pruning_op_types": ["Linear"],
                "pruning_type": "retrain_free",
                "pruning_scope": "local",
                "pruning_frequency": 2,
            },
            {
                "op_names": ['encoder_layer_2.mlp*'],
                'target_sparsity': 0.4,
                'pattern': 'channelx2',
                "pruning_op_types": ["Linear"],
                "pruning_type": "retrain_free",
                "pruning_scope": "global",
                "pruning_frequency": 3,
            },
            {
                "op_names": ['encoder_layer_0.mlp*', "conv_proj"],
                'target_sparsity': 0.4,
                'pattern': 'channelx1',
                "pruning_op_types": ["Linear","Conv2d"],
                "pruning_type": "retrain_free",
                "pruning_scope": "global",
                "pruning_frequency": 3,
            }
        ]
        config = WeightPruningConfig(
            local_configs,
            target_sparsity=0.8,
            start_step=1,
            end_step=10
        )

        criterion = nn.CrossEntropyLoss()
        from neural_compressor.compression.pruner import prepare_pruning
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        
        # pruning = prepare_pruning(config, self.model)
        # for epoch in range(2):
        #     self.model.train()
        #     local_step = 0
        #     for image, target in dummy_dataloader:
        #         pruning.on_step_begin(local_step)
        #         output = self.model(image)
        #         loss = criterion(output, target)
        #         loss.backward()
        #         pruning.on_step_end()
        #         local_step += 1
        # pruning.on_train_end()

        pruning = prepare_pruning(config, self.model, dataloader=dummy_dataloader, loss_func=criterion)

if __name__ == "__main__":
    unittest.main()




