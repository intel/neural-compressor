import unittest

import torch

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()


def get_gpt_j():
    import transformers

    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        torchscript=True,
    )
    return tiny_gptj


class GPTQLLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(10):
            yield torch.ones([1, 512], dtype=torch.long)


class GPTQLLMDataLoaderList:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(10):
            yield (torch.ones([1, 512], dtype=torch.long), torch.ones([1, 512], dtype=torch.long))


class GPTQLLMDataLoaderDict:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(10):
            yield {
                "input_ids": torch.ones([1, 512], dtype=torch.long),
                "attention_mask": torch.ones([1, 512], dtype=torch.long),
            }


class TestGPTQ(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestGPTQ test: {self.id()}")

    def test_default_gptq(self):
        # Ported from test/adaptor/pytorch_adaptor/test_weight_only_adaptor.py
        # TestPytorchWeightOnlyAdaptor.test_GPTQ_fixed_length_quant
        from neural_compressor.torch import GPTQConfig, quantize

        # "gptq_args": {"percdamp": 0.01, "act_order": False, "use_max_length": True, "pad_max_length": 512},
        quant_config = GPTQConfig(weight_group_size=8, pad_max_length=512)
        quant_config.set_local("lm_head", GPTQConfig(weight_dtype="fp32"))
        logger.info(f"Test GPTQ with config {quant_config}")
        dataloader = GPTQLLMDataLoader()

        # case 1: tensor
        model_1 = get_gpt_j()
        input = torch.ones([1, 512], dtype=torch.long)
        out0 = model_1(input)
        q_model = quantize(model=model_1, quant_config=quant_config, calib_dataloader=dataloader)
        out1 = q_model(input)
        self.assertTrue(torch.allclose(out1[0], out0[0], atol=1e-02))


if __name__ == "__main__":
    unittest.main()
