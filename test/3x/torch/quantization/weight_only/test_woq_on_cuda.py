import random

import pytest
import torch
import transformers
from tqdm import tqdm

from neural_compressor.common.utils import logger
from neural_compressor.torch.algorithms.weight_only.gptq import move_input_to_device
from neural_compressor.torch.quantization import GPTQConfig, get_default_rtn_config, quantize


class GPTQDataloaderPreprocessor:
    def __init__(self, dataloader_original, use_max_length=False, max_seq_length=2048, nsamples=128):
        self.dataloader_original = dataloader_original
        self.use_max_length = use_max_length
        self.max_seq_length = max_seq_length
        self.nsamples = nsamples
        self.dataloader = []
        self.is_ready = False

    def get_prepared_dataloader(self):
        if not self.is_ready:
            self.prepare_dataloader()
        return self.dataloader

    def prepare_dataloader(self):
        if self.use_max_length:
            # (Recommend) only take sequence whose length exceeds self.max_seq_length,
            # which preserves calibration's tokens are all valid
            # This is GPTQ official dataloader implementation
            self.obtain_first_n_samples_fulllength()
        else:
            # general selection, no padding, not GPTQ original implementation.
            self.obtain_first_n_samples()
        self.is_ready = True

    def obtain_first_n_samples(self, seed=0):
        """Get first nsample data as the real calibration dataset."""
        self.dataloader.clear()
        random.seed(seed)
        for batch in self.dataloader_original:
            # process data, depends on its data type.
            if len(self.dataloader) == self.nsamples:
                logger.info(f"Successfully collect {self.nsamples} calibration samples.")
                break
            # list, tuple
            if isinstance(batch, list) or isinstance(batch, tuple):
                if batch[0].shape[-1] > self.max_seq_length:
                    i = random.randint(0, batch[0].shape[-1] - self.max_seq_length - 1)
                    j = i + self.max_seq_length
                    batch_final = []
                    for item in batch:
                        if isinstance(item, torch.Tensor) and item.shape.__len__() == 2:
                            batch_final.append(item[:, i:j])
                        else:
                            batch_final.append(item)
                else:
                    batch_final = batch[:]
            # dict
            elif isinstance(batch, dict):
                try:
                    length = batch["input_ids"].shape[-1]
                except:
                    logger.warning("Please make sure your dict'like data contains key of 'input_ids'.")
                    continue
                batch_final = {}
                if length > self.max_seq_length:
                    i = random.randint(0, length - self.max_seq_length - 1)
                    j = i + self.max_seq_length
                    # may have to slice every sequence related data
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch_final[key] = batch[key][:, i:j]  # slice on sequence length dim
                        else:
                            batch_final[key] = batch[key]
                else:
                    batch_final = batch
            # tensor
            else:
                if batch.shape[-1] > self.max_seq_length:
                    i = random.randint(0, batch.shape[-1] - self.max_seq_length - 1)
                    j = i + self.max_seq_length
                    batch_final = batch[:, i:j]
                else:
                    batch_final = batch
            self.dataloader.append(batch_final)

        if len(self.dataloader) < self.nsamples:
            logger.warning(f"Try to use {self.nsamples} data, but entire dataset size is {len(self.dataloader)}.")

    def obtain_first_n_samples_fulllength(self, seed=0):
        self.dataloader.clear()
        random.seed(seed)
        unified_length = self.max_seq_length
        for batch in self.dataloader_original:
            if len(self.dataloader) == self.nsamples:
                logger.info(f"Successfully collect {self.nsamples} calibration samples.")
                break
            # list & tuple, gpt-j-6b mlperf, etc.
            if isinstance(batch, list) or isinstance(batch, tuple):
                if batch[0].shape[-1] == unified_length:
                    batch_final = batch[:]
                elif batch[0].shape[-1] > unified_length:
                    i = random.randint(0, batch[0].shape[-1] - unified_length - 1)
                    j = i + unified_length
                    batch_final = []
                    for item in batch:
                        if isinstance(item, torch.Tensor) and item.shape.__len__() == 2:
                            batch_final.append(item[:, i:j])
                        else:
                            batch_final.append(item)
                else:
                    # not match max length, not include in target dataset
                    continue
            # dict
            elif isinstance(batch, dict):
                try:
                    length = batch["input_ids"].shape[-1]
                except:
                    logger.warning("Please make sure your dict'like data contains key of 'input_ids'.")
                    continue
                batch_final = {}
                if length == self.max_seq_length:
                    batch_final = batch
                elif length > self.max_seq_length:
                    i = random.randint(0, length - self.max_seq_length - 1)
                    j = i + self.max_seq_length
                    # may have to slice every sequence related data
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch_final[key] = batch[key][:, i:j]  # slice on sequence length dim with same position
                        else:
                            batch_final[key] = batch[key]
                else:
                    # not match max length, not include in target dataset
                    continue
            # tensor
            else:
                if batch.shape[-1] == unified_length:
                    batch_final = batch
                elif batch.shape[-1] > unified_length:
                    i = random.randint(0, batch.shape[-1] - unified_length - 1)
                    j = i + unified_length
                    batch_final = batch[:, i:j]
                else:
                    # not match max length, not include in target dataset
                    continue
            self.dataloader.append(batch_final)
        if len(self.dataloader) < self.nsamples:  # pragma: no cover
            logger.warning(
                f"Trying to allocate {self.nsamples} data with fixed length {unified_length}, \
                but only {len(self.dataloader)} samples are found. Please use smaller 'self.max_seq_length' value."
            )


class TestGPTQ:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires a GPU")
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
        model_device = str(user_model.device)
        assert "cuda" in model_device, f"Model device is {model_device}"


class TestRTNQuant:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires a GPU")
    def test_rtn(self):
        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )
        self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
        model = self.tiny_gptj
        # record label for comparison
        self.label = model(self.example_inputs.to(model.device))[0]
        # test_default_config
        quant_config = get_default_rtn_config()
        q_model = quantize(model, quant_config)
        assert "cuda" in str(q_model.device), f"Expect qmodel device is cuda, got {q_model.device}"
