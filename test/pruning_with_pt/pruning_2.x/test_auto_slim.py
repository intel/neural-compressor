import sys
import unittest

import torch
import torch.nn as nn
import torchvision

from neural_compressor import WeightPruningConfig

# auto slim
from neural_compressor.compression.pruner import model_slim, parse_auto_slim_config
from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.training import prepare_compression


class TestPruning(unittest.TestCase):
    def test_pruning_basic(self):
        print("Run a Bert model")
        # create model, datasets, criterion and optimizer
        from transformers import BertForSequenceClassification

        model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-mini")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(10, 16), low=0.0, high=1.0, dtype="int64", label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        # case 1: without external dataloader
        prune_ffn2_sparsity = 0.5
        prune_mha_sparsity = 0.5
        auto_slim_configs = parse_auto_slim_config(
            model,
            ffn2_sparsity=prune_ffn2_sparsity,
            mha_sparsity=prune_mha_sparsity,
            pruning_scope="local",
        )
        # case 2: with external dataloader
        # get auto config for ffn and mha
        auto_slim_configs_2 = parse_auto_slim_config(
            model,
            dummy_dataloader,
            ffn2_sparsity=prune_ffn2_sparsity,
            mha_sparsity=prune_mha_sparsity,
            pruning_scope="local",
        )
        pruning_configs = []
        pruning_configs += auto_slim_configs_2
        configs = WeightPruningConfig(pruning_configs, start_step=1, end_step=25)
        # run mha and ffn pruning
        compression_manager = prepare_compression(model=model, confs=configs)
        compression_manager.callbacks.on_train_begin()
        # import pdb;pdb.set_trace()
        for epoch in range(3):
            model.train()
            compression_manager.callbacks.on_epoch_begin(epoch)
            local_step = 0
            for inp, target in dummy_dataloader:
                compression_manager.callbacks.on_step_begin(local_step)
                output = model(inp)
                loss = criterion(output.logits, target)
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

        # execute real slim process (remove weights)
        model = model_slim(model)


if __name__ == "__main__":
    unittest.main()
