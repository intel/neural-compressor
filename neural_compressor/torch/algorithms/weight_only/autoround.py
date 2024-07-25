# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AutoRound quantization."""
import copy
import json
import time
from typing import Union

import torch
from auto_round import AutoRound  # pylint: disable=E0401
from auto_round.export.export_to_itrex.export import pack_model  # pylint: disable=E0401

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.utils import get_accelerator, logger

from .utility import CapturedDataloader, InputCaptureModule


class AutoRoundQuantizer(Quantizer):
    """AutoRound Quantizer."""

    def __init__(
        self,
        quant_config: dict = {},
        enable_full_range: bool = False,  ##for symmetric, TODO support later
        batch_size: int = 8,
        amp: bool = True,
        device: str = None,
        lr_scheduler=None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        enable_quanted_input: bool = True,
        enable_minmax_tuning: bool = True,
        lr: float = None,
        minmax_lr: float = None,
        low_gpu_mem_usage: bool = False,
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        sampler: str = "rand",
        seed: int = 42,
        nblocks: int = 1,
        gradient_accumulate_steps: int = 1,
        not_use_best_mse: bool = False,
        dynamic_max_gap: int = -1,
        data_type: str = "int",
        scale_dtype: str = "fp16",
        quant_block_list: list = None,
        act_bits: int = 32,
        act_group_size: int = None,
        act_sym: bool = None,
        act_dynamic: bool = True,
        low_cpu_mem_usage: bool = False,
        **kwargs,
    ):
        """Init a AutQRoundQuantizer object.

        Args:
            quant_config (dict): Configuration for weight quantization (default is None).
            quant_config={
                        'layer1':##layer_name
                        {
                            'data_type': 'int',
                            'bits': 4,
                            'group_size': 32,
                            'sym': False,
                            'act_data_type': None,
                            'act_bits': 32,
                            'act_sym': None,
                            'act_dynamic': True,
                        }
                        ...,
                    }
                keys:
                    data_type (str): The data type to be used (default is "int").
                    bits (int): Number of bits for quantization (default is 4).
                    group_size (int): Size of the quantization group (default is 128).
                    sym (bool): Whether to use symmetric quantization. (default is None).
            bits (int): Number of bits for quantization (default is 4).
            group_size (int): Size of the quantization group (default is 128).
            sym (bool): Whether symmetric quantization is to be used (default is False).
            enable_full_range (bool): Whether to enable full range quantization (default is False).
            batch_size (int): Batch size for training (default is 8).
            amp (bool): Whether to use automatic mixed precision (default is True).
            device: The device to be used for tuning (default is "auto").
            lr_scheduler: The learning rate scheduler to be used.
            dataset (str): The default dataset name (default is "NeelNanda/pile-10k").
            enable_quanted_input (bool): Whether to use the output of the previous quantized block as
                                            the input for the current block (default is True).
            enable_minmax_tuning (bool): Whether to enable weight min-max tuning (default is True).
            lr (float): The learning rate (default is None, will be set to 1.0/iters).
            minmax_lr (float): The learning rate for min-max tuning
                                    (default is None, it will be set to lr automatically).
            low_gpu_mem_usage (bool): Whether to use low GPU memory (default is True).
            iters (int): Number of iterations (default is 200).
            seqlen (int): Data length of the sequence for tuning (default is 2048).
            nsamples (int): Number of samples (default is 128).
            sampler (str): The sampling method (default is "rand").
            seed (int): The random seed (default is 42).
            nblocks (int): Number of blocks (default is 1).
            gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
            not_use_best_mse (bool): Whether to use mean squared error (default is False).
            dynamic_max_gap (int): The dynamic maximum gap (default is -1).
            data_type (str): The data type to be used (default is "int").
            scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
                have different choices.
            quant_block_list (list): A list whose elements are list of block's layer names to be quantized.
            act_bits (int): Number of bits for activation quantization. Default is 32.
            act_group_size (int): Group size for activation quantization. Default is None.
            act_sym (bool): Whether to use symmetric activation quantization. Default is None.
            act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.

        Returns:
            The quantized model.
        """
        super().__init__(quant_config)
        self.tokenizer = None
        self.enable_full_range = enable_full_range
        self.batch_size = batch_size
        self.amp = amp
        self.device = get_accelerator(kwargs.pop("device", "auto")).current_device_name()
        self.lr_scheduler = lr_scheduler
        self.enable_quanted_input = enable_quanted_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.iters = iters
        self.seqlen = seqlen
        self.nsamples = nsamples
        self.sampler = sampler
        self.seed = seed
        self.nblocks = nblocks
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.data_type = data_type
        self.scale_dtype = scale_dtype
        self.quant_block_list = quant_block_list
        self.act_bits = act_bits
        self.act_group_size = act_group_size
        self.act_sym = act_sym
        self.act_dynamic = act_dynamic
        self.low_cpu_mem_usage = low_cpu_mem_usage

    def prepare(self, model: torch.nn.Module, *args, **kwargs):
        """Prepares a given model for quantization.

        Args:
            model (torch.nn.Module): The model to be prepared.

        Returns:
            A prepared model.
        """
        prepare_model = InputCaptureModule(model)
        return prepare_model

    def convert(self, model: torch.nn.Module, *args, **kwargs):
        """Convert the prepared model to a quantized model.

        Args:
            model (torch.nn.Module): the prepared model

        Returns:
            The quantized model.
        """
        dataloader = CapturedDataloader(model.args_list, model.kwargs_list)
        model = model.orig_model
        rounder = AutoRound(
            model=model,
            tokenizer=None,
            dataset=dataloader,
            layer_config=self.quant_config or {},
            enable_full_range=self.enable_full_range,
            batch_size=self.batch_size,
            amp=self.amp,
            device=self.device,
            lr_scheduler=self.lr_scheduler,
            enable_quanted_input=self.enable_quanted_input,
            enable_minmax_tuning=self.enable_minmax_tuning,
            lr=self.lr,
            minmax_lr=self.minmax_lr,
            low_gpu_mem_usage=self.low_gpu_mem_usage,
            iters=self.iters,
            seqlen=self.seqlen,
            nsamples=self.nsamples,
            sampler=self.sampler,
            seed=self.seed,
            nblocks=self.nblocks,
            gradient_accumulate_steps=self.gradient_accumulate_steps,
            not_use_best_mse=self.not_use_best_mse,
            dynamic_max_gap=self.dynamic_max_gap,
            data_type=self.data_type,
            scale_dtype=self.scale_dtype,
            quant_block_list=self.quant_block_list,
            act_bits=self.act_bits,
            act_group_size=self.act_group_size,
            act_sym=self.act_sym,
            act_dynamic=self.act_dynamic,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )
        model, weight_config = rounder.quantize()
        model.autoround_config = weight_config
        model = pack_model(model, weight_config, device=self.device, inplace=True)
        return model


def get_dataloader(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", seed=42, bs=8, nsamples=128):
    """Generate a DataLoader for calibration using specified parameters.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        seqlen (int): The exact sequence length. samples < seqlen will be dropped,
                      samples longer than seqlen will be truncated
        dataset_name (str, optional): The name of the dataset or datasets separated by commas.
                                     Defaults to "NeelNanda/pile-10k".
        split (str, optional): The data split to use. Defaults to None.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        bs (int, optional): The batch size. Defaults to 4.
        nsamples (int, optional): The total number of samples to include. Defaults to 128.

    Returns:
        DataLoader: The DataLoader for the calibrated dataset.
    """
    from auto_round.calib_dataset import get_dataloader  # pylint: disable=E0401

    dataloader = get_dataloader(
        tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", seed=seed, bs=bs, nsamples=nsamples
    )
    return dataloader
