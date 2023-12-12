import unittest

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from neural_compressor.adaptor.torch_utils.autoround.autoround import AutoAdamRound, AutoOPTRound, AutoRound


class TestAutoRoundLinear(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "facebook/opt-125m"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=True
            ##low_cpu_mem_usage has impact to acc, changed the random seed?
        )
        self.model = self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @classmethod
    def test_signround(self):
        round = AutoRound(self.model, self.tokenizer, device="cpu", iters=5, seqlen=8, n_samples=1, group_size=7)
        round.quantize()

    @classmethod
    def test_Adamround(self):
        round = AutoOPTRound(self.model, self.tokenizer, device="cpu", iters=2, seqlen=8, n_samples=1, scheme="sym")
        round.quantize()


class TestAutoRoundConv1D(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "MBZUAI/LaMini-GPT-124M"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=True
            ##low_cpu_mem_usage has impact to acc, changed the random seed?
        )
        self.model = self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @classmethod
    def test_signround(self):
        round = AutoRound(self.model, self.tokenizer, device="cpu", iters=5, seqlen=8, n_samples=1, n_blocks=2)
        round.quantize()

    @classmethod
    def test_Adamround(self):
        round = AutoAdamRound(self.model, self.tokenizer, device="cpu", iters=5, seqlen=8, n_samples=1)
        round.quantize()
