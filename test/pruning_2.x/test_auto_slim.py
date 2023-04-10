import unittest

import torch
import torchvision
import torch.nn as nn
import sys
sys.path.insert(0, './')
from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor import WeightPruningConfig
from neural_compressor.training import prepare_compression
import random

class TestPruning(unittest.TestCase):
    def test_pruning_basic(self):
        # task1: check config generation functions
        print("test")
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        prune_ffn2_sparsity = 0.5
        prune_mha_sparsity = 0.5
        from neural_compressor.compression import parse_auto_slim_config
        auto_slim_configs = parse_auto_slim_config(
            model, 
            ffn2_sparsity = prune_ffn2_sparsity, 
            mha_sparsity = prune_mha_sparsity,
            pruning_scope="local",
        )
        assert auto_slim_configs[0]['op_names'].__len__() == 12 # ffn2
        assert auto_slim_configs[1]['op_names'].__len__() == 36 # mha qkv
        assert auto_slim_configs[2]['op_names'].__len__() == 12 # mha outputs

        # task2: check auto slim compression functions, input outputs should be the same.
        dummy_inputs = model.dummy_inputs['input_ids']
        print("Do the auto slim.")
        intm_size = model.config.intermediate_size
        head_size = model.config.hidden_size // model.config.num_attention_heads
        # generate a sparse model first
        for n, m in model.named_modules():
            if n in auto_slim_configs[0]['op_names']:
                ffn2_chn_sparsity = random.sample(list(range(intm_size)), int(intm_size * prune_ffn2_sparsity))
                _w = m.weight.clone()
                _w[:, ffn2_chn_sparsity] = 0
                setattr(m, 'weight', torch.nn.Parameter(_w.clone()))
            if n in auto_slim_configs[1]['op_names']:
                _w = m.weight.clone()
                _w[0:head_size, :] = 0
                setattr(m, 'weight', torch.nn.Parameter(_w.clone()))
            if n in auto_slim_configs[2]['op_names']:
                _w = m.weight.clone()
                _w[:, 0:head_size] = 0
                setattr(m, 'weight', torch.nn.Parameter(_w.clone()))
        # slim the model
        from neural_compressor.compression import model_slim
        outputs_before_slim = model(dummy_inputs)
        model = model_slim(model)
        outputs_after_slim = model(dummy_inputs)
        assert torch.sum(outputs_before_slim.logits - outputs_after_slim.logits).abs().item() < 1e-5


if __name__ == "__main__":
    unittest.main()
