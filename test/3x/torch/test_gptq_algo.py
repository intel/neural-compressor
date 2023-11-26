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


from tqdm import tqdm

from neural_compressor.torch.algorithms.gptq import move_input_to_device


def run_fn_for_gptq(model, dataloader_for_calibration, *args):
    logger.info("Collecting calibration inputs...")
    for batch in tqdm(dataloader_for_calibration):
        batch = move_input_to_device(batch, device=None)
        try:
            if isinstance(batch, tuple) or isinstance(batch, list):
                model(batch[0])
            elif isinstance(batch, dict):
                model(**batch)
            else:
                model(batch)
        except ValueError:
            pass
    return


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

        dataloader = GPTQLLMDataLoader()

        # case 1: tensor
        model_1 = get_gpt_j()
        input = torch.ones([1, 512], dtype=torch.long)
        out0 = model_1(input)
        device = None
        from neural_compressor.torch.algorithms.gptq import DataloaderPreprocessor

        dataloaderPreprocessor = DataloaderPreprocessor(
            dataloader_original=dataloader, use_max_length=False, pad_max_length=512, nsamples=128
        )
        dataloader_for_calibration = dataloaderPreprocessor.get_prepared_dataloader()

        quant_config = GPTQConfig(
            weight_group_size=8, dataloader_len=len(dataloader_for_calibration), pad_max_length=512
        )
        quant_config.set_local("lm_head", GPTQConfig(weight_dtype="fp32"))
        logger.info(f"Test GPTQ with config {quant_config}")
        q_model = quantize(
            model=model_1, quant_config=quant_config, run_fn=run_fn_for_gptq, run_args=dataloader_for_calibration
        )
        out1 = q_model(input)
        self.assertTrue(torch.allclose(out1[0], out0[0], atol=1e-02))

    def test_gptq_advance(self):
        # Ported from test/adaptor/pytorch_adaptor/test_weight_only_adaptor.py
        # TestPytorchWeightOnlyAdaptor.test_GPTQ_fixed_length_quant
        from neural_compressor.torch import GPTQConfig, quantize

        dataloader = GPTQLLMDataLoader()
        model_1 = get_gpt_j()
        input = torch.ones([1, 512], dtype=torch.long)
        out0 = model_1(input)

        device = None
        from neural_compressor.torch.algorithms.gptq import DataloaderPreprocessor

        dataloaderPreprocessor = DataloaderPreprocessor(
            dataloader_original=dataloader, use_max_length=False, pad_max_length=512, nsamples=128
        )
        dataloader_for_calibration = dataloaderPreprocessor.get_prepared_dataloader()

        quant_config = GPTQConfig(
            weight_group_size=8,
            dataloader_len=len(dataloader_for_calibration),
            act_order=True,
            enable_mse_search=True,
            pad_max_length=512,
        )
        quant_config.set_local("lm_head", GPTQConfig(weight_dtype="fp32"))
        logger.info(f"Test GPTQ with config {quant_config}")
        q_model = quantize(
            model=model_1, quant_config=quant_config, run_fn=run_fn_for_gptq, run_args=dataloader_for_calibration
        )
        out1 = q_model(input)
        self.assertTrue(torch.allclose(out1[0], out0[0], atol=1e-02))

    def _apply_gptq(self, input, model, quant_config, run_fn, run_args):
        logger.info(f"Test GPTQ with config {quant_config}")
        from neural_compressor.torch import quantize

        out0 = model(input)
        q_model = quantize(model=model, quant_config=quant_config, run_fn=run_fn, run_args=run_args)
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
        for dataloader_cls in dataloader_collections:
            for value in product(*gptq_options.values()):
                d = dict(zip(gptq_options.keys(), value))
                quant_config = GPTQConfig(**d)
                length = 512 if quant_config.use_max_length else random.randint(1, 1024)
                from neural_compressor.torch.algorithms.gptq import DataloaderPreprocessor

                dataloaderPreprocessor = DataloaderPreprocessor(
                    dataloader_original=dataloader_cls(length), use_max_length=False, pad_max_length=512, nsamples=128
                )
                dataloader_for_calibration = dataloaderPreprocessor.get_prepared_dataloader()
                quant_config.dataloader_len = len(dataloader_for_calibration)

                self._apply_gptq(
                    model=get_gpt_j(),
                    input=input,
                    quant_config=quant_config,
                    run_fn=run_fn_for_gptq,
                    run_args=dataloader_for_calibration,
                )


if __name__ == "__main__":
    unittest.main()
