import random
import torch
from collections import UserDict
from packaging.version import Version
from neural_compressor.common import logger
from neural_compressor.torch.utils import get_torch_version

class DataloaderPreprocessor:
    def __init__(self, dataloader_original, use_max_length=False, max_seq_length=2048, nsamples=128) -> None:
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


def get_example_inputs(model, dataloader):
    version = get_torch_version()
    from neural_compressor.torch.algorithms.smooth_quant import move_input_to_device

    # Suggest set dataloader like calib_dataloader
    if dataloader is None:
        return None
    device = next(model.parameters()).device
    try:
        for idx, (input, label) in enumerate(dataloader):
            input = move_input_to_device(input, device)
            if isinstance(input, (dict, UserDict)):  # pragma: no cover
                assert version.release >= Version("1.12.0").release, "INC support IPEX version >= 1.12.0"
                if "label" in input.keys():
                    input.pop("label")
                if version.release <= Version("2.0.1").release:
                    return tuple(input.values())
                else:
                    return dict(input)
            if isinstance(input, (list, tuple)):
                return tuple(input)
            if isinstance(input, torch.Tensor):
                return input
            break
    except Exception as e:  # pragma: no cover
        for idx, input in enumerate(dataloader):
            input = move_input_to_device(input, device)
            if isinstance(input, (dict, UserDict)):  # pragma: no cover
                assert version.release >= Version("1.12.0").release, "INC support IPEX version >= 1.12.0"
                if "label" in input.keys():
                    input.pop("label")
                if version.release <= Version("2.0.1").release:
                    return tuple(input.values())
                else:
                    return dict(input)
            if isinstance(input, list) or isinstance(input, tuple):
                return tuple(input)
            if isinstance(input, torch.Tensor):
                return input
            break
    if idx == 0:
        assert False, "Please checkout the example_inputs format."
