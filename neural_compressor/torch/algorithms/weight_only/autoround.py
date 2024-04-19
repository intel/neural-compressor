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

import torch
from auto_round import AutoRound  # pylint: disable=E0401
from auto_round.calib_dataset import CALIB_DATASETS  # pylint: disable=E0401

from neural_compressor.torch.utils import logger


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


class InputCaptureModule(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.data_pairs = []
        self.device = "cpu"

    def forward(self, *args, **kwargs):
        if kwargs and len(args) == 0:
            # Handle cases where input data is a dict
            self.data_pairs.append(kwargs)
        elif args and len(args) == 1:
            # Handle cases where input data is a Tensor
            self.data_pairs.append(args[0])
        else:
            logger.error("Handle cases where input data is neither a Tensor nor a dict")


def recover_dataloader_from_calib_fn(run_fn, run_args):
    input_capture_model = InputCaptureModule()
    input_capture_model.eval()
    run_fn(input_capture_model, *run_args)
    dataloader = torch.utils.data.DataLoader(input_capture_model.data_pairs)
    return dataloader


def autoround_quantize(
    model,
    weight_config: dict = {},
    enable_full_range: bool = False,  ##for symmetric, TODO support later
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
    scale_dtype="fp16",
    run_fn=None,
    run_args=None,
):
    """The entry point of the autoround weight-only quantization.
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
    run_fn: a calibration function for calibrating the model. Defaults to None.
    run_args: positional arguments for `run_fn`. Defaults to None.

    Returns:
        The quantized model.
    """
    if run_fn is None or run_fn == get_autoround_default_run_fn:
        assert run_args is not None, "Please provide tokenizer for AutoRound default calibration."
        run_fn = get_autoround_default_run_fn
    dataloader = recover_dataloader_from_calib_fn(run_fn, run_args)

    rounder = AutoRound(
        model=model,
        tokenizer=None,
        bits=4,
        group_size=128,
        sym=False,
        weight_config=weight_config,
        enable_full_range=enable_full_range,  ##for symmetric, TODO support later
        batch_size=batch_size,
        amp=amp,
        device=device,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
        use_quant_input=use_quant_input,
        enable_minmax_tuning=enable_minmax_tuning,
        lr=lr,
        minmax_lr=minmax_lr,
        low_gpu_mem_usage=low_gpu_mem_usage,
        iters=iters,
        seqlen=seqlen,
        n_samples=n_samples,
        sampler=sampler,
        seed=seed,
        n_blocks=n_blocks,
        gradient_accumulate_steps=gradient_accumulate_steps,
        not_use_best_mse=not_use_best_mse,
        dynamic_max_gap=dynamic_max_gap,
        data_type="int",
        scale_dtype=scale_dtype,
        run_fn=run_fn,
        run_args=run_args,
    )
    qdq_model, weight_config = rounder.quantize()
    return qdq_model, weight_config
