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
"""This utility is for block-wise calibration of LLMs."""

import gc
from functools import partial

import torch

from neural_compressor.torch.utils import (
    fetch_module,
    forward_wrapper,
    get_accelerator,
    get_non_persistent_buffers,
    load_non_persistent_buffers,
    logger,
    set_module,
)

cur_accelerator = get_accelerator()


# copied from neural_compressor.torch.algorithms.weight_only.utility
def get_block_prefix(model):
    """Get prefix and number of attention blocks of transformer models.

    Args:
        model (torch.nn.Module): input model

    Returns:
        block_prefix(str): block_list name in model
        block_num(int): number of block in block_list
    """
    module_types = [torch.nn.ModuleList]
    for n, m in model.named_modules():
        if type(m) in module_types:
            block_prefix = n
            block_num = len(m)
            logger.debug(f"block_prefix: {block_prefix}, block_num: {block_num} ")
            break
    assert block_num > 0, "block num shouldn't be zero!"
    return block_prefix, block_num


def replace_forward(model):
    """Replace forward to get the input args and kwargs of first block for AWQ algorithm.

    Args:
        model (torch.nn.Module): input model.

    Raises:
        ValueError: to avoid inference of rest parts in model.

    Returns:
        torch.nn.Module: model with replaced forward.
    """
    # Step 1: replace block_forward to collect block inputs and avoid entire inference
    setattr(model, "total_block_args", [])
    setattr(model, "total_block_kwargs", [])

    def forward(layer, *args, **kwargs):
        # update total_hidden_states, total_block_kwargs, per batch
        model.total_block_args.append(list(args))
        model.total_block_kwargs.append(kwargs)
        raise ValueError

    block_prefix, block_num = get_block_prefix(model)
    block_list = fetch_module(model, block_prefix)
    first_block = block_list[0]
    first_block.forward_orig = first_block.forward
    first_block.forward = partial(forward, first_block)

    # Step 2: replace model_forward to avoid ValueError
    model.forward_orig = model.forward
    model_forward_cache = model.forward

    def model_forward(model, *args, **kwargs):
        nonlocal model_forward_cache
        try:
            model_forward_cache(*args, **kwargs)
        except ValueError:
            pass

    model.forward = partial(model_forward, model)
    return model


def recover_forward(model):
    """Recover model and block forward for AWQ algorithm.

    Args:
        model (torch.nn.Module): input model.

    Returns:
        torch.nn.Module: model with recovered forward.
    """
    model.forward = model.forward_orig

    block_prefix, _ = get_block_prefix(model)
    block_list = fetch_module(model, block_prefix)
    first_block = block_list[0]
    first_block.forward = first_block.forward_orig
    return model


def block_wise_calibration(model, dataloader=None, data=None, inference_dtype=torch.bfloat16):
    """Calibration model on hpu block-by-block to reduce device memory usage.

    Args:
        model (torch.nn.Module): prepared model.
        dataloader (obj): dataloader.
        data (obj): one data.
    """

    def get_block_kwargs(model, dataloader=None, data=None, device="cpu"):
        """Get the input args and kwargs of the first block.

        Args:
            model (torch.nn.Module): prepared model.
            dataloader (obj): dataloader.
            data (obj): one data.
            device (str, optional): target device. Defaults to "cpu".

        Returns:
            total_block_args: list of input args of each batch.
            total_block_kwargs: list of input kwargs of each batch.
        """
        with torch.no_grad():
            model = replace_forward(model)
            mapped_state_dict = model.state_dict()  # using memory mapping
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Embedding):
                    m = m.to(device).to(inference_dtype)
                    set_module(model, n, m)
            if dataloader is not None:
                for i, data in enumerate(dataloader):
                    logger.info(f"Collecting block inputs from batch: {i}")
                    forward_wrapper(model, data)
                    cur_accelerator.synchronize()
            if data is not None:
                forward_wrapper(model, data)
            model = recover_forward(model)
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.Embedding):
                    m = m.to("meta")  # clean device memory
                    set_module(model, n, m)
            gc.collect()
            cur_accelerator.synchronize()
            total_block_args = getattr(model, "total_block_args", [])
            total_block_kwargs = getattr(model, "total_block_kwargs", [])
            model.load_state_dict(mapped_state_dict, assign=True)  # recover memory mapping
        return total_block_args, total_block_kwargs

    def calibrate_block_and_update_inputs(block, total_block_args, total_block_kwargs, device="cpu"):
        """Calibrate current block and update args and kwargs for the next block.

        Args:
            raw_block (torch.nn.Module): block module for calibration
            total_block_args: list of input args of each batch.
            total_block_kwargs: list of input kwargs of each batch.
            device (str, optional): target device. Defaults to "cpu".

        Returns:
            total_block_args: list of input args of each batch for the next block.
            total_block_kwargs: list of input kwargs of each batch for the next block.
        """
        with torch.no_grad():
            mapped_state_dict = block.state_dict()  # using memory mapping
            block = block.to(device).to(inference_dtype)
            # get block outputs
            for block_id, (args, kwargs) in enumerate(zip(total_block_args, total_block_kwargs)):
                out = block(*args, **kwargs)
                if isinstance(out, tuple):  # pragma: no cover
                    out = out[0]
                # update inputs of next block
                if len(total_block_args[block_id]) > 0:
                    total_block_args[block_id][0].copy_(out)
                elif "hidden_states" in total_block_kwargs[block_id]:
                    total_block_kwargs[block_id]["hidden_states"].copy_(out)
                cur_accelerator.synchronize()
            block = block.to("meta")  # clean device memory
            gc.collect()
            cur_accelerator.synchronize()
            block.load_state_dict(mapped_state_dict, assign=True)  # recover memory mapping
        return total_block_args, total_block_kwargs

    # Get non_persistent_buffers
    non_persistent_buffers = get_non_persistent_buffers(model)
    # Get prefix and number of attention blocks of transformer models.
    device = cur_accelerator.name()
    block_prefix, block_num = get_block_prefix(model)
    # Use prefix to fetch all attention blocks. For llama, it's LlamaDecoderLayer module.
    block_list = fetch_module(model, block_prefix)
    logger.info("Getting inputs of the first block")

    total_block_args, total_block_kwargs = get_block_kwargs(model, dataloader=dataloader, data=data, device=device)
    for i, block in enumerate(block_list):
        logger.info("Calibrating block: {}".format(i))
        # Calibrate current block and update args and kwargs for the next block.
        total_block_args, total_block_kwargs = calibrate_block_and_update_inputs(
            block, total_block_args, total_block_kwargs, device=device
        )
    # Load non_persistent_buffers
    load_non_persistent_buffers(model, non_persistent_buffers)
    # Clean host and device memory
    del total_block_args, total_block_kwargs
    gc.collect()
    cur_accelerator.synchronize()
    logger.info("Calibration is done!")
