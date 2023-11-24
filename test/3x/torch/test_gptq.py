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
    def __init__(self, length=512):
        self.batch_size = 1
        self.length = length

    def __iter__(self):
        for i in range(10):
            yield torch.ones([1, self.length], dtype=torch.long)


class GPTQLLMDataLoaderList(GPTQLLMDataLoader):
    def __iter__(self):
        for i in range(10):
            yield (torch.ones([1, self.length], dtype=torch.long), torch.ones([1, self.length], dtype=torch.long))


class GPTQLLMDataLoaderDict(GPTQLLMDataLoader):
    def __iter__(self):
        for i in range(10):
            yield {
                "input_ids": torch.ones([1, self.length], dtype=torch.long),
                "attention_mask": torch.ones([1, self.length], dtype=torch.long),
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

    def test_gptq(self):
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

    def _apply_gptq(self, input, model, quant_config, dataloader):
        logger.info(f"Test GPTQ with config {quant_config}")
        from neural_compressor.torch import quantize

        out0 = model(input)
        q_model = quantize(model=model, quant_config=quant_config, calib_dataloader=dataloader)
        out1 = q_model(input)
        self.assertTrue(torch.allclose(out1[0], out0[0], atol=1e-02))

    def test_more_gptq(self):
        import random
        from itertools import product

        from neural_compressor.torch import GPTQConfig

        # some tests were skipped to accelerate the CI
        input = torch.ones([1, 512], dtype=torch.long)
        # dataloader
        dataloader_collections = [GPTQLLMDataLoader, GPTQLLMDataLoaderList, GPTQLLMDataLoaderDict]
        gptq_options = {
            "weight_sym": [False, True],
            "weight_group_size": [8],
            "use_max_length": [False, True],
            "pad_max_length": [512],
        }
        for dataloader in dataloader_collections:
            for value in product(*gptq_options.values()):
                d = dict(zip(gptq_options.keys(), value))
                quant_config = GPTQConfig(**d)
                length = 512 if quant_config.use_max_length else random.randint(1, 1024)
                self._apply_gptq(
                    model=get_gpt_j(), input=input, quant_config=quant_config, dataloader=dataloader(length)
                )

    def test_gptq_advance(self):
        # Ported from test/adaptor/pytorch_adaptor/test_weight_only_adaptor.py
        # TestPytorchWeightOnlyAdaptor.test_GPTQ_fixed_length_quant
        from neural_compressor.torch import GPTQConfig, quantize

        # "gptq_args": {"percdamp": 0.01, "act_order": False, "use_max_length": True, "pad_max_length": 512},
        quant_config = GPTQConfig(weight_group_size=8, act_order=True, enable_mse_search=True, pad_max_length=512)
        quant_config.set_local("lm_head", GPTQConfig(weight_dtype="fp32"))
        logger.info(f"Test GPTQ with config {quant_config}")
        dataloader = GPTQLLMDataLoader()
        model_1 = get_gpt_j()
        input = torch.ones([1, 512], dtype=torch.long)
        out0 = model_1(input)
        q_model = quantize(model=model_1, quant_config=quant_config, calib_dataloader=dataloader)
        out1 = q_model(input)
        self.assertTrue(torch.allclose(out1[0], out0[0], atol=1e-02))


if __name__ == "__main__":
    unittest.main()
