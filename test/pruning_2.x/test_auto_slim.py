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

# auto slim
from neural_compressor.compression.pruner.model_slim import parse_auto_slim_config
from neural_compressor.compression.pruner.model_slim import model_slim

import random

class NaiveMLP(nn.Module):
    def __init__(self, hidden_size=16):
        super(NaiveMLP, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ac1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ac2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, 2, bias=True)
        self.ac3 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.ac1(x)
        x = self.linear2(x)
        x = self.ac2(x)
        x = self.linear3(x)
        x = self.ac3(x)
        return x

class TestPruning(unittest.TestCase):

    def test_pruning_basic(self):
        prune_ffn2_sparsity = 0.5
        prune_mha_sparsity = 0.5
        hidden_size = 16
        # print("Run a naive MLP model")
        model = NaiveMLP(hidden_size)
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, hidden_size), low=0., high=1., dtype='float32', label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        auto_slim_configs_0 = parse_auto_slim_config(
            model, 
            dummy_dataloader,
            ffn2_sparsity = prune_ffn2_sparsity, 
            pruning_scope="local",
        )
        assert auto_slim_configs_0[0]['op_names'].__len__() == 2 # ffn2
        for n, m in model.named_modules():
            if n in auto_slim_configs_0[0]['op_names']:
                ffn2_chn_sparsity = random.sample(list(range(hidden_size)), int(hidden_size * prune_ffn2_sparsity))
                _w = m.weight.clone()
                _w[:, ffn2_chn_sparsity] = 0
                setattr(m, 'weight', torch.nn.Parameter(_w.clone()))
        dummy_inputs = torch.randn([1, 16])
        outputs_before_slim = model(dummy_inputs)
        model = model_slim(model, dummy_dataloader)
        outputs_after_slim = model(dummy_inputs)
        assert torch.sum(outputs_before_slim - outputs_after_slim).abs().item() < 1e-5


        # task1: check config generation functions
        print("Run a Bert model")
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        # case 1: without external dataloader
        auto_slim_configs = parse_auto_slim_config(
            model, 
            ffn2_sparsity = prune_ffn2_sparsity, 
            mha_sparsity = prune_mha_sparsity,
            pruning_scope="local",
        )
        assert auto_slim_configs[0]['op_names'].__len__() == 12 # ffn2
        assert auto_slim_configs[1]['op_names'].__len__() == 36 # mha qkv
        assert auto_slim_configs[2]['op_names'].__len__() == 12 # mha outputs
        # case 2: with external dataloader
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 16), low=0., high=1., dtype='int64', label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        auto_slim_configs_2 = parse_auto_slim_config(
            model, 
            dummy_dataloader,
            ffn2_sparsity = prune_ffn2_sparsity, 
            mha_sparsity = prune_mha_sparsity,
            pruning_scope="local",
        )
        assert auto_slim_configs_2[0]['op_names'].__len__() == 12 # ffn2
        assert auto_slim_configs_2[1]['op_names'].__len__() == 36 # mha qkv
        assert auto_slim_configs_2[2]['op_names'].__len__() == 12 # mha outputs

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
        outputs_before_slim = model(dummy_inputs)
        model = model_slim(model)
        outputs_after_slim = model(dummy_inputs)
        assert torch.sum(outputs_before_slim.logits - outputs_after_slim.logits).abs().item() < 1e-5


if __name__ == "__main__":
    unittest.main()
