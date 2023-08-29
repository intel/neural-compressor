import unittest

import torch
import torch.nn as nn
import torchvision

from neural_compressor.conf.pythonic_config import Config, WeightPruningConfig
from neural_compressor.data import Datasets
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.experimental.pruning_v2 import Pruning

local_regs_config = [
    {
        "start_step": 0,
        "end_step": 10,
        "pruning_type": "magnitude",
        "op_names": ["layer1.*"],
        "excluded_op_names": ["layer2.*"],
        "pruning_scope": "global",
        "target_sparsity": 0.5,
        "pattern": "4x1",
        "reg_type": "group_lasso",
        "parameters": {"reg_coeff": 0.2},
    },
    {
        "start_step": 1,
        "end_step": 1,
        "target_sparsity": 0.5,
        "pruning_type": "snip_momentum",
        "pruning_frequency": 2,
        "op_names": ["layer2.*"],
        "pruning_scope": "local",
        "pattern": "1x1",
        "sparsity_decay_type": "exp",
        "reg_type": "group_lasso",
        "parameters": {"reg_coeff": 0.1},
    },
    {
        "start_step": 2,
        "end_step": 8,
        "pruning_type": "gradient",
        "pruning_frequency": 2,
        "op_names": ["fc"],
        "pruning_scope": "local",
        "target_sparsity": 0.75,
        "pattern": "1x1",
        "sparsity_decay_type": "cube",
        "reg_type": "group_lasso",
        "parameters": {"reg_coeff": 0.0},
    },
]

fake_snip_config = WeightPruningConfig(
    local_regs_config, target_sparsity=0.9, start_step=0, end_step=10, pruning_frequency=1, sparsity_decay_type="exp"
)


class TestPruningRegs(unittest.TestCase):
    model = torchvision.models.resnet18()

    def test_pruning_regs(self):
        config = Config(quantization=None, benchmark=None, pruning=fake_snip_config, distillation=None)
        prune = Pruning(config)
        prune.model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        prune.on_train_begin()
        prune.update_config(pruning_frequency=1)
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
        elementwise_over_matmul_gemm_conv, elementwise_over_all, blockwise_over_matmul_gemm_conv \
            = prune.get_sparsity_ratio()
        prune.on_train_end()
        prune.on_before_eval()
        prune.on_after_eval()

        # assert sparsity ratio
        self.assertAlmostEqual(elementwise_over_matmul_gemm_conv , 0.02244, delta=0.0001)
        self.assertAlmostEqual(elementwise_over_all, 0.02242, delta=0.0001)
        self.assertAlmostEqual(blockwise_over_matmul_gemm_conv, 0.02244, delta=0.0001)


if __name__ == "__main__":
    unittest.main()
