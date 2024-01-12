import sys
import unittest

sys.path.insert(0, "./")

from transformers import AutoModelForCausalLM

from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader


class TestPruning(unittest.TestCase):
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

    def test_pruning_basic(self):
        from neural_compressor.compression.pruner.wanda import prune_wanda
        from neural_compressor.compression.pruner.wanda.utils import get_module_list, get_tensor_sparsity_ratio

        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(10, 512), low=0.0, high=10.0, label=True, dtype="int64")
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        sparsity_ratio = 0.8
        prune_wanda(self.model, dummy_dataloader, sparsity_ratio, dsnot=False, use_variant=False)
        model_list = get_module_list(self.model)
        for block in model_list:
            self.assertAlmostEqual(get_tensor_sparsity_ratio(block.self_attn.q_proj.weight.data), sparsity_ratio, 2)
            self.assertAlmostEqual(get_tensor_sparsity_ratio(block.fc1.weight.data), sparsity_ratio, 2)


if __name__ == "__main__":
    unittest.main()
