#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for pytorch framework."""

import argparse
import os

from intel_extension_for_transformers.tools.utils import is_ipex_available

from neural_compressor.common.utils import CpuInfo, LazyImport, logger

CONFIG_NAME = "best_configure.yaml"
ENGINE_MODEL_NAME = "model.bin"
ENGINE_MODEL_CONFIG = "conf.yaml"
ENCODER_NAME = "encoder_model.bin"
DECODER_NAME = "decoder_model.bin"
DECODER_WITH_PAST_NAME = "decoder_with_past_model.bin"
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
QUANT_CONFIG = "quantize_config.json"
SPARSITY_CONFIG = "sparsity_config.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

if is_ipex_available():
    import intel_extension_for_pytorch as ipex
torch = LazyImport("torch")


def distributed_init(
    backend="gloo",
    world_size=1,
    rank=-1,
    init_method=None,
    master_addr="127.0.0.1",
    master_port="12345",
):
    """Init the distribute environment."""
    rank = int(os.environ.get("RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", world_size))
    if init_method is None:
        master_addr = os.environ.get("MASTER_ADDR", master_addr)
        master_port = os.environ.get("MASTER_PORT", master_port)
        init_method = "env://{addr}:{port}".format(addr=master_addr, port=master_port)
    torch.distributed.init_process_group(backend, init_method=init_method, world_size=world_size, rank=rank)


def remove_label(input):
    if "labels" in input:  # for GLUE
        input.pop("labels")
    elif "start_positions" in input and "end_positions" in input:  # for SQuAD
        # pragma: no cover
        input.pop("start_positions")
        input.pop("end_positions")
    return input


def _build_inc_dataloader(dataloader):
    # transformer issue #1
    # for transformers 4.31.0: accelerate dataloader
    # *** ValueError: batch_size attribute should not be set
    # after DataLoaderShard is initialized
    class INCDataLoader:
        __iter__ = dataloader.__iter__
        __len__ = dataloader.__len__

        def __init__(self) -> None:
            self.dataloader = dataloader
            self.batch_size = dataloader.total_batch_size
            self.dataset = dataloader.dataset

    return INCDataLoader()


"""Utility."""

import importlib
import sys
import warnings

import torch

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

try:
    import habana_frameworks.torch.hpu as hthpu

    is_hpu_available = True
except ImportError:
    is_hpu_available = False


def supported_gpus():
    return ["flex", "max", "arc"]


def is_intel_gpu_available():
    import intel_extension_for_pytorch as ipex

    return hasattr(torch, "xpu") and torch.xpu.is_available()


_ipex_available = importlib.util.find_spec("intel_extension_for_pytorch") is not None
_ipex_version = "N/A"
if _ipex_available:
    try:
        _ipex_version = importlib_metadata.version("intel_extension_for_pytorch")
    except importlib_metadata.PackageNotFoundError:
        _ipex_available = False


def is_ipex_available():
    return _ipex_available


_autoround_available = importlib.util.find_spec("auto_round") is not None
_autoround_version = "N/A"
if _autoround_available:
    try:
        _autoround_version = importlib_metadata.version("auto_round")
    except importlib_metadata.PackageNotFoundError:
        _autoround_available = False


def is_autoround_available():
    return _autoround_available


_neural_compressor_available = importlib.util.find_spec("neural_compressor") is not None
_neural_compressor_version = "N/A"

if _neural_compressor_available:
    try:
        _neural_compressor_version = importlib_metadata.version("neural_compressor")
    except importlib_metadata.PackageNotFoundError:
        _neural_compressor_available = False


def is_neural_compressor_avaliable():
    return _neural_compressor_available


def get_device_type():
    if torch.cuda.is_available():
        device = "cuda"
    elif is_hpu_available:
        device = "hpu"
    elif is_ipex_available() and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    return device
