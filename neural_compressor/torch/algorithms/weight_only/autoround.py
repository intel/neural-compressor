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

import time

import torch
from auto_round import AutoRound  # pylint: disable=E0401
from auto_round.calib_dataset import CALIB_DATASETS  # pylint: disable=E0401
from auto_round.utils import get_block_names  # pylint: disable=E0401

from neural_compressor.torch.algorithms import Quantizer
from neural_compressor.torch.utils import logger


class AutoRoundQuantizer(Quantizer):
    def __init__(
        self,
        model,
        weight_config: dict = {},
        enable_full_range: bool = False,
        batch_size: int = 8,
        amp: bool = True,
        device=None,
        lr_scheduler=None,
        use_quant_input: bool = True,
        enable_minmax_tuning: bool = True,
        lr: float = None,
        minmax_lr: float = None,
        low_gpu_mem_usage: bool = True,
        iters: int = 200,
        seqlen: int = 2048,
        n_samples: int = 512,
        sampler: str = "rand",
        seed: int = 42,
        n_blocks: int = 1,
        gradient_accumulate_steps: int = 1,
        not_use_best_mse: bool = False,
        dynamic_max_gap: int = -1,
        scale_dtype="fp32",
    ):
        """Init a AutQRoundQuantizer object.

        Args:
        model: The PyTorch model to be quantized.
        weight_config (dict): Configuration for weight quantization (default is an empty dictionary).
        weight_config={
                    'layer1':##layer_name
                    {
                        'data_type': 'int',
                        'bits': 4,
                        'group_size': 32,
                        'sym': False,
                    }
                    ...
                }
            keys:
                data_type (str): The data type to be used (default is "int").
                bits (int): Number of bits for quantization (default is 4).
                group_size (int): Size of the quantization group (default is 128).
                sym (bool): Whether to use symmetric quantization. (default is None).
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True). Automatically detect and set.
        device: The device to be used for tuning (default is None). Automatically detect and set.
        lr_scheduler: The learning rate scheduler to be used.
        use_quant_input (bool): Whether to use quantized input data (default is True).
        enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
        lr (float): The learning rate (default is 0.005).
        minmax_lr (float): The learning rate for min-max tuning (default is None).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is True).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        n_samples (int): Number of samples (default is 512).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).
        n_blocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        scale_dtype (str): The data type of quantization scale to be used (default is "float32"), different kernels
                            have different choices.
        """

        self.model = model
        self.tokenizer = None
        self.weight_config = weight_config
        self.enable_full_range = enable_full_range
        self.batch_size = batch_size
        self.amp = amp
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.use_quant_input = use_quant_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.iters = iters
        self.seqlen = seqlen
        self.n_samples = n_samples
        self.sampler = sampler
        self.seed = seed
        self.n_blocks = n_blocks
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.data_type = "int"
        self.scale_dtype = scale_dtype

    def quantize(self, model: torch.nn.Module, *args, **kwargs):
        run_fn = kwargs.get("run_fn", None)
        run_args = kwargs.get("run_args", None)
        assert run_fn is not None, (
            "Can't find run_func. Please provide run_func to quantize API "
            "or overwrite quantize member function in your Quantizer class."
        )
        model = self.prepare(model)
        if run_args:
            run_fn(model, *run_args)
        else:
            run_fn(model)
        model = self.convert(model)
        return model

    def prepare(self, model: torch.nn.Module, *args, **kwargs):
        """Prepares a given model for quantization.
        Args:
            model (torch.nn.Module): The model to be prepared.

        Returns:
            A prepared model.
        """
        self.rounder = AutoRoundProcessor(
            model=model,
            tokenizer=None,
            weight_config=self.weight_config,
            enable_full_range=self.enable_full_range,
            batch_size=self.batch_size,
            amp=self.amp,
            device=self.device,
            lr_scheduler=self.lr_scheduler,
            use_quant_input=self.use_quant_input,
            enable_minmax_tuning=self.enable_minmax_tuning,
            lr=self.lr,
            minmax_lr=self.minmax_lr,
            low_gpu_mem_usage=self.low_gpu_mem_usage,
            iters=self.iters,
            seqlen=self.seqlen,
            n_samples=self.n_samples,
            sampler=self.sampler,
            seed=self.seed,
            n_blocks=self.n_blocks,
            gradient_accumulate_steps=self.gradient_accumulate_steps,
            not_use_best_mse=self.not_use_best_mse,
            dynamic_max_gap=self.dynamic_max_gap,
            data_type=self.data_type,
            scale_dtype=self.scale_dtype,
        )
        self.rounder.prepare()
        return model

    def convert(self, model: torch.nn.Module, *args, **kwargs):
        model, weight_config = self.rounder.convert()
        model.autoround_config = weight_config
        return model


@torch.no_grad()
def get_autoround_default_run_fn(
    model,
    tokenizer,
    dataset_name="NeelNanda/pile-10k",
    n_samples=512,
    seqlen=2048,
    seed=42,
    bs=8,
    dataset_split: str = "train",
    dataloader=None,
):
    """Perform calibration for quantization.

    This method calibrates the model for quantization by processing a specified
    number of samples from the calibration dataset. It ensures that the data is
    properly formatted and feeds it to the model. If the number of samples processed
    is less than the specified number, it logs a warning. If no samples are processed,
    it logs an error and exits.

    Args:
        n_samples (int): The number of samples to use for calibration.
    """
    if dataloader is None:
        get_dataloader = CALIB_DATASETS.get(dataset_name, CALIB_DATASETS["NeelNanda/pile-10k"])
        dataloader = get_dataloader(
            tokenizer,
            seqlen,
            seed=seed,
            bs=bs,
            split=dataset_split,
            dataset_name=dataset_name,
        )
    total_cnt = 0
    for data in dataloader:
        if data is None:
            continue
        if isinstance(data, torch.Tensor):
            data_new = data.to(model.device)
            input_ids = data_new
        else:
            data_new = {}
            for key in data.keys():
                data_new[key] = data[key].to(model.device)
            input_ids = data_new["input_ids"]
        # if input_ids.shape[-1] < seqlen:
        #     continue
        if total_cnt + input_ids.shape[0] > n_samples:
            input_ids = input_ids[: n_samples - total_cnt, ...]
        try:
            if isinstance(data_new, torch.Tensor):
                model(data_new)
            elif isinstance(data_new, dict):
                model(**data_new)
            else:
                # Handle cases where data_new is neither a Tensor nor a dict
                raise NotImplementedError(f"Handling not implemented for data type {type(data)}")
        except Exception as error:
            logger.error(error)
        total_cnt += input_ids.shape[0]
        if total_cnt >= n_samples:
            break
    if total_cnt == 0:
        logger.error(
            "no data has been cached, please provide more data with sequence length >= {} in the ".format(seqlen)
            + "dataloader or decease the sequence length."
        )
        exit()
    elif total_cnt < n_samples:
        logger.warning(
            "Insufficient number of samples collected may affect the quantification. "
            "Effective samples size: {}, Target sample size: {}".format(total_cnt, n_samples)
        )


class AutoRoundProcessor(AutoRound):

    def prepare(self):
        """Quantize the model and return the quantized model along with weight configurations.

        Returns:
        The quantized model and weight configurations.
        """
        # logger.info("cache block input")
        self.start_time = time.time()
        self.block_names = get_block_names(self.model)
        if len(self.block_names) == 0:
            logger.warning("could not find blocks, exit with original model")
            return
        if self.amp:
            self.model = self.model.to(self.amp_dtype)
        if not self.low_gpu_mem_usage:
            self.model = self.model.to(self.device)
        # inputs = self.cache_block_input(block_names[0], self.n_samples)

        # cache block input
        self.inputs = {}
        self.tmp_block_name = self.block_names[0]
        self._replace_forward()

    def convert(self):
        # self.calib(self.n_samples)
        self._recover_forward()
        inputs = self.inputs[self.tmp_block_name]
        del self.tmp_block_name

        del self.inputs
        if "input_ids" in inputs.keys():
            dim = int((hasattr(self.model, "config") and "chatglm" in self.model.config.model_type))
            total_samples = inputs["input_ids"].shape[dim]
            self.n_samples = total_samples
            if total_samples < self.train_bs:
                self.train_bs = total_samples
                logger.warning(f"force the train batch size to {total_samples} ")
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()
        self.qdq_weight_round(
            self.model,
            inputs,
            self.block_names,
            n_blocks=self.n_blocks,
            device=self.device,
        )
        for n, m in self.model.named_modules():
            if n in self.weight_config.keys():
                if hasattr(m, "scale"):
                    self.weight_config[n]["scale"] = m.scale
                    self.weight_config[n]["zp"] = m.zp
                    if self.group_size <= 0:
                        self.weight_config[n]["g_idx"] = torch.tensor(
                            [0 for i in range(m.weight.shape[1])], dtype=torch.int32, device="cpu"
                        )
                    else:
                        self.weight_config[n]["g_idx"] = torch.tensor(
                            [i // self.group_size for i in range(m.weight.shape[1])], dtype=torch.int32, device="cpu"
                        )
                    delattr(m, "scale")
                    delattr(m, "zp")
                else:
                    self.weight_config[n]["data_type"] = "float"
                    if self.amp_dtype == torch.bfloat16:
                        self.weight_config[n]["data_type"] = "bfloat"
                    self.weight_config[n]["bits"] = 16
                    self.weight_config[n]["group_size"] = None
                    self.weight_config[n]["sym"] = None

        end_time = time.time()
        cost_time = end_time - self.start_time
        logger.info(f"quantization tuning time {cost_time}")
        ## dump a summary
        quantized_layers = []
        unquantized_layers = []
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(self.supported_types)):
                if self.weight_config[n]["bits"] == 16:
                    unquantized_layers.append(n)
                else:
                    quantized_layers.append(n)
        summary_info = (
            f"Summary: quantized {len(quantized_layers)}/{len(quantized_layers) + len(unquantized_layers)} in the model"
        )
        if len(unquantized_layers) > 0:
            summary_info += f",  {unquantized_layers} have not been quantized"

        logger.info(summary_info)
        if len(unquantized_layers) > 0:
            logger.info(f"Summary: {unquantized_layers} have not been quantized")

        self.quantized = True
        self.model = self.model.to(self.model_orig_dtype)
        return self.model, self.weight_config
