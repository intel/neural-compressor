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
"""Intel Neural Compressor PyTorch Quantizer API."""

import copy
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from neural_compressor.common.utils import Mode


class Quantizer(ABC):
    """The base quantizer for all algorithm quantizers.

    The `Quantizer` unifies the interfaces across various quantization algorithms, including GPTQ, RTN, etc.
    Given a float model, `Quantizer` apply the quantization algorithm to the model according to the `quant_config`.

    To implement a new quantization algorithm,, inherit from `Quantizer` and implement the following methods:
        - `prepare`: prepare a given model for convert.
        - `convert`: convert a prepared model to a quantized model.
    Note: `quantize` and `execute` are optional for new quantization algorithms.
    """

    def __init__(self, quant_config: Optional[Any] = None):
        """Init a Quantizer object.

        Args:
            quant_config : Specifies how to apply the algorithm on the given model.
            The format of `quant_config` can be defined by `Quantized` itself.
            For example, `quant_config` can be a dictionary as below:
                quant_config={
                'fc2':{
                    'dtype': 'int',
                    'bits': 4,
                    'group_size': 32,
                    'scheme': 'sym'
                    }}
        """
        self.quant_config = quant_config

    @abstractmethod
    def prepare(self, model: torch.nn.Module, *args: Any, **kwargs: Any):
        """Prepares a given model for quantization.

        Insert observers into the model so that it can monitor the input and output tensors during calibration.

        Args:
            model (torch.nn.Module): The model to be prepared.

        Returns:
            A prepared model.
        """
        raise NotImplementedError("{} doesn't implement `prepare` function. ".format(self.__class__.__name__))

    @abstractmethod
    def convert(self, model: torch.nn.Module, *args: Any, **kwargs: Any):
        """Converts a prepared model to a quantized model.

        Args:
            model (torch.nn.Module): The prepared model to be converted.

        Returns:
            A quantized model.
        """
        raise NotImplementedError("{} doesn't implement `convert` function. ".format(self.__class__.__name__))

    def quantize(self, model: torch.nn.Module, *args: Any, **kwargs: Any):
        """Quantizes a given float model.

        Args:
            model (torch.nn.Module): The float model to be quantized.

        Returns:
            A quantized model.
        """
        model = self.prepare(model, *args, **kwargs)

        run_fn = kwargs.get("run_fn", None)
        if run_fn is not None:
            run_args = kwargs.get("run_args", None)
            if run_args:
                run_fn(model, *run_args)
            else:
                run_fn(model)

        model = self.convert(model, *args, **kwargs)

        return model

    def execute(self, model: torch.nn.Module, mode, *args: Any, **kwargs: Any):
        """Execute according to mode.

        Args:
            model (torch.nn.Module): The model to be executed.
            mode (Mode): The mode of current phase, including 'prepare', 'convert' and 'quantize'.
        """
        if mode == Mode.PREPARE:
            model = self.prepare(model, *args, **kwargs)
        elif mode == Mode.CONVERT:
            model = self.convert(model, *args, **kwargs)
        elif mode == Mode.QUANTIZE:
            if not isinstance(self.quant_config, dict):
                user_cfg = copy.deepcopy(self.quant_config).to_dict()
            else:
                user_cfg = copy.deepcopy(self.quant_config)
            if "recipe_cfgs" in user_cfg:  # keep quantize API for smoothquant
                run_fn = kwargs.get("run_fn", None)
                example_inputs = kwargs.get("example_inputs", None)
                inplace = kwargs.get("inplace", True)
                model = self.quantize(model, self.quant_config, run_fn, example_inputs, inplace)
            else:
                model = self.quantize(model, *args, **kwargs)
        return model
