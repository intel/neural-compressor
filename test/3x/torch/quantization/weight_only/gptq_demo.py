# ! Rename it

import pytest
import torch
import transformers
from tqdm import tqdm

from neural_compressor.torch.algorithms.weight_only.gptq import GPTQDataloaderPreprocessor, move_input_to_device
from neural_compressor.torch.quantization import GPTQConfig, quantize


class TestGPTQ:
    def test_GPTQ_fixed_length_quant(self):
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

        dataloader = GPTQLLMDataLoader()  # !!!! Failed
        dataloader_list = GPTQLLMDataLoaderList()
        dataloader_dict = GPTQLLMDataLoaderDict()

        quant_config = GPTQConfig()
        quant_config.set_local("lm_head", GPTQConfig(dtype="fp32"))

        gptq_use_max_length = False
        gptq_max_seq_length = 2048
        dataloaderPreprocessor = GPTQDataloaderPreprocessor(
            dataloader_original=dataloader_list,
            use_max_length=gptq_use_max_length,
            max_seq_length=gptq_max_seq_length,
        )
        dataloader_for_calibration = dataloaderPreprocessor.get_prepared_dataloader()

        def run_fn_for_gptq(model, dataloader_for_calibration, *args):
            for batch in tqdm(dataloader_for_calibration):
                batch = move_input_to_device(batch, device=model.device)
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

        user_model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )

        user_model = quantize(
            model=user_model, quant_config=quant_config, run_fn=run_fn_for_gptq, run_args=dataloader_for_calibration
        )


t = TestGPTQ()
t.test_GPTQ_fixed_length_quant()
