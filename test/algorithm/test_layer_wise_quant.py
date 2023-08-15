import unittest
import sys
import shutil

sys.path.insert(0, './')
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from neural_compressor.adaptor.torch_utils.layer_wise_quant import load_shell
from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.utils.pytorch import load


class TestLayerWise(unittest.TestCase):
    def test_layer_wise(self):

        model_name_or_path = 'facebook/opt-125m'
        fp32_model = load_shell(model_name_or_path, AutoModelForCausalLM, torchscript=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        texts = [
            "Replace me by any text you'd like.", "This is a test sentence."
        ]
        tokenizer.pad_token = tokenizer.eos_token
        encoded_input = tokenizer(texts, return_tensors='pt', padding=True)

        class TestDataset(Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                return {
                    key: torch.tensor(val[idx])
                    for key, val in self.encodings.items()
                }

            def __len__(self):
                return len(self.encodings.input_ids)

        eval_dataset = TestDataset(encoded_input)
        eval_dataloader = DataLoader(eval_dataset, batch_size=8)

        conf = PostTrainingQuantConfig(
            calibration_sampling_size=8,
            recipes={
                "layer_wise_quant": True,
                "layer_wise_quant_args": {
                    "model_path": "facebook/opt-125m",
                    "smooth_quant": True,
                    "smooth_quant_alpha": 0.5
                }
            })

        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=eval_dataloader,
            eval_func=lambda x: 0.1,
        )
        ouput_dir = './saved_model'
        q_model.save(ouput_dir)
        load_model = load(ouput_dir,
                          AutoModelForCausalLM.from_pretrained(model_name_or_path))
        lm_weight = q_model._model.lm_head.module.weight()
        test_value = load_model.lm_head.module.weight().equal(lm_weight)
        self.assertTrue(test_value)
        shutil.rmtree(ouput_dir)

    def test_util(self):
        from neural_compressor.adaptor.torch_utils.layer_wise_quant.utils import (
            get_children,
            get_named_children,
            get_super_module_by_name
        )

        model_name_or_path = 'facebook/opt-125m'
        model = load_shell(model_name_or_path, AutoModelForCausalLM, torchscript=True)
        children = get_children(model)
        named_children = get_named_children(model)
        self.assertEqual(children, [v for k, v in named_children])
        module = get_super_module_by_name(model, named_children[0][0])
        self.assertTrue(hasattr(module, named_children[0][0].split('.')[-1]))


if __name__ == '__main__':
    unittest.main()
