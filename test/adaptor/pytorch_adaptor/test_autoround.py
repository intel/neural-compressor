import copy
import os
import shutil
import sys
import unittest

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_compressor.adaptor.torch_utils.autoround import (
    AutoAdamRound,
    AutoOPTRound,
    AutoRound,
    export_compressed_model,
)


class SimpleDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.randn([1, 30])


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestPytorchWeightOnlyAdaptor(unittest.TestCase):
    approach = "weight_only"

    @classmethod
    def setUpClass(self):
        self.dataloader = SimpleDataLoader()
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM", trust_remote_code=True
        )
        self.gptj_no_jit = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )
        self.llm_dataloader = LLMDataLoader()
        self.lm_input = torch.ones([1, 10], dtype=torch.long)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_RTN_int_quant(self):
        model = copy.deepcopy(self.gptj)
        out1 = model(self.lm_input)
        round = AutoRound
        optq_1 = round(model, self.tokenizer, n_samples=20, amp=False, seqlen=10)
        q_model, weight_config1 = optq_1.quantize()
        compressed_model = export_compressed_model(q_model, weight_config1)
        out2 = model(self.lm_input)
        out3 = compressed_model(self.lm_input)
        self.assertTrue(torch.all(torch.isclose(out1[0], out2[0], atol=1e-1)))
        self.assertFalse(torch.all(out1[0] == out2[0]))
        self.assertTrue(torch.all(torch.isclose(out2[0], out3[0], atol=1e-3)))
        self.assertTrue("transformer.h.0.attn.k_proj.qzeros" in compressed_model.state_dict().keys())

        # model = copy.deepcopy(self.gptj)
        # out6 = model(self.lm_input)
        # optq_2 = round(model, self.tokenizer, n_samples=20, amp=False, seqlen=10)
        # q_model, weight_config2 = optq_2.quantize()
        # out4 = q_model(self.lm_input)
        # out5 = model(self.lm_input)

        # self.assertTrue(torch.all(out1[0] == out6[0]))
        # self.assertTrue(torch.all(out4[0] == out5[0]))
        # self.assertTrue(torch.all(torch.isclose(out6[0], out5[0], atol=1e-1)))


if __name__ == "__main__":
    unittest.main()
