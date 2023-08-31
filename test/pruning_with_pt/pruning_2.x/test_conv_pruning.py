import sys
import unittest

import torch

sys.path.insert(0, "./")
from neural_compressor import WeightPruningConfig
from neural_compressor.training import prepare_compression


class TestPruning(unittest.TestCase):
    def test_conv1_prunig(self):
        local_config = [
            {
                "op_names": ["conv1.*"],
                "target_sparsity": 0.6,
                "pattern": "channelx1",
                "pruning_type": "snip",
                "pruning_scope": "local",
            },
            {"op_names": ["conv2.*"], "target_sparsity": 0.5, "pattern": "2:4", "pruning_scope": "global"},
        ]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(4, 4, 2)
                self.act = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv1d(4, 4, 2)
                self.linear = torch.nn.Linear(32, 3)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = self.conv2(out)
                out = out.view(1, -1)
                out = self.linear(out)
                return out

        model = Model()
        data = torch.rand((1, 4, 10))
        output = model(data)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        config = WeightPruningConfig(local_config, target_sparsity=0.8, start_step=1, end_step=10)
        compression_manager = prepare_compression(model=model, confs=config)
        compression_manager.callbacks.on_train_begin()
        for epoch in range(2):
            model.train()
            compression_manager.callbacks.on_epoch_begin(epoch)
            local_step = 0
            for _ in range(20):
                data, target = torch.rand((1, 4, 10), requires_grad=True), torch.empty(1, dtype=torch.long).random_(3)
                compression_manager.callbacks.on_step_begin(local_step)
                output = model(data)
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

    def test_hf_conv1_prunig(self):
        import transformers

        local_config = [
            {
                "op_names": ["conv1.*"],
                "target_sparsity": 0.6,
                "pattern": "channelx1",
                "pruning_type": "snip",
                "pruning_scope": "local",
            },
            {"op_names": ["conv2.*"], "target_sparsity": 0.5, "pattern": "2:4", "pruning_scope": "global"},
        ]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = transformers.Conv1D(4, 4)
                self.act = torch.nn.ReLU()
                self.conv2 = transformers.Conv1D(4, 4)
                self.linear = torch.nn.Linear(16, 3)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = self.conv2(out)
                out = out.view(1, -1)
                out = self.linear(out)
                return out

        model = Model()
        data = torch.rand((1, 4, 4))
        output = model(data)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        config = WeightPruningConfig(local_config, target_sparsity=0.8, start_step=1, end_step=10)
        compression_manager = prepare_compression(model=model, confs=config)
        compression_manager.callbacks.on_train_begin()
        for epoch in range(2):
            model.train()
            compression_manager.callbacks.on_epoch_begin(epoch)
            local_step = 0
            for _ in range(20):
                data, target = torch.rand((1, 4, 4), requires_grad=True), torch.empty(1, dtype=torch.long).random_(3)
                compression_manager.callbacks.on_step_begin(local_step)
                output = model(data)
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
