import shutil
import unittest

import pytest
import torch
from optimum.intel import INCModelForCausalLM
from transformers import AutoTokenizer

from neural_compressor.transformers import GPTQConfig, RtnConfig


class TestQuantizationConfig(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "TheBlokeAI/Mixtral-tiny-GPTQ"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.prompt = "One day, the little girl"
        self.input_ids = self.tokenizer(self.prompt, return_tensors="pt")["input_ids"]

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("tmp_gptq")
        shutil.rmtree("tmp_rtn")

    def test_gptq(self):
        quantization_config = GPTQConfig(bits=4, sym=True, damp_percent=0.01, desc_act=True)
        user_model = INCModelForCausalLM.from_pretrained(self.model_name, quantization_config=quantization_config)
        output = user_model(self.input_ids)
        user_model.save_pretrained("tmp_gptq")
        loaded_model = INCModelForCausalLM.from_pretrained("tmp_gptq")
        loaded_output = loaded_model(self.input_ids)
        assert torch.allclose(output, loaded_output, atol=1e-2), "Compare failed!"

    def test_rtn(self):
        quantization_config = RtnConfig(bits=4)
        user_model = INCModelForCausalLM.from_pretrained(self.model_name, quantization_config=quantization_config)
        output = user_model(self.input_ids)
        user_model.save_pretrained("tmp_rtn")
        loaded_model = INCModelForCausalLM.from_pretrained("tmp_rtn")
        loaded_output = loaded_model(self.input_ids)
        assert torch.allclose(output, loaded_output, atol=1e-2), "Compare failed!"
