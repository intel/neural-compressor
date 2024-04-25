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

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import torch

from neural_compressor.torch.utils import Mode


class Quantizer(ABC):
    """The base quantizer for one step algorithm quantizers."""

    def __init__(self, tune_cfg: OrderedDict = {}):
        self.tune_cfg = tune_cfg

    @abstractmethod
    def quantize(self, model: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:
        raise NotImplementedError("{} doesn't implement `quantize` function.".format(self.__class__.__name__))

    def prepare(self, model: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:
        raise NotImplementedError(
            "{} doesn't implement `prepare` function. "
            "Please implement `prepare` function or "
            "switch to call `from neural_compressor.torch.quantization import quantize` to do quantization.".format(
                self.__class__.__name__
            )
        )

    def convert(self, model: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:
        raise NotImplementedError(
            "{} doesn't implement `convert` function. "
            "Please implement `convert` function or "
            "switch to call `from neural_compressor.torch.quantization import quantize` to do quantization.".format(
                self.__class__.__name__
            )
        )

    def execute(self, model, mode, *args: Any, **kwargs: Any):
        if mode == Mode.PREPARE:
            model = self.prepare(model, *args, **kwargs)
        elif mode == Mode.CONVERT:
            model = self.convert(model, *args, **kwargs)
        elif mode == Mode.QUANTIZE:
            model = self.quantize(model, *args, **kwargs)
        return model
