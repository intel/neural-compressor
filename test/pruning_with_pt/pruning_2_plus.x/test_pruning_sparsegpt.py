import unittest

from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor import WeightPruningConfig
from transformers import (AutoModelForCausalLM)


class TestPruning(unittest.TestCase):
    model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m"
            )
    def test_pruning_basic(self):
        local_configs = [
            {
                "op_names": ['5.fc', '5.attn'],
                "target_sparsity": 0.65,
                "pattern": '1x1',
                "pruning_type": "sparse_gpt",
                "pruning_op_types": ["Linear"],
            },
            {
                "op_names": ['7.fc', '7.attn'],
                "target_sparsity": 0.5,
                "pattern": '2:4',
                "pruning_op_types": ["Linear"],
                "pruning_type": "sparse_gpt",
            },
        ]
        config = WeightPruningConfig(
            local_configs,
            target_sparsity=0.5,
            start_step=1,
            end_step=10
        )

        from neural_compressor.compression.pruner import prepare_pruning
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 512), low=0., high=1., label=True, dtype='int64')
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        
        pruning = prepare_pruning(config, self.model, dataloader=dummy_dataloader, device='cpu')
        pruning.on_train_begin(dummy_dataloader)
        pruning.on_train_end()


if __name__ == "__main__":
    unittest.main()

