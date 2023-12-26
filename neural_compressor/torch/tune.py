# Copyright (c) 2023 Intel Corporation
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

from typing import Callable, Optional, Tuple

import torch

from neural_compressor.common.base_tune import BaseQuantizer, BaseTuningConfig, Tuner
from neural_compressor.common.logger import Logger

logger = Logger().get_logger()


def get_default_tuning_config():
    # TODO (Yi) support it in the next PR
    return None


class TorchQuantizer(BaseQuantizer):
    def __init__(
        self, model: torch.nn.Module, run_fn: Optional[Callable] = None, run_args: Optional[Tuple] = None
    ) -> None:
        super().__init__(model)
        self.run_fn = run_fn
        self.run_args = run_args

    def prepare(self, quant_config):
        """Prepare a copy of the model for quantization."""
        pass

    def calibrate(self):
        """Run the prepared model on the calibration dataset."""
        pass

    def convert(self):
        "Convert a calibrated model to quantized model."
        pass

    def quantize(self, quant_config):
        """The entry to quantize a model."""
        logger.info(f"apply quant_config: {quant_config}.")
        # TODO, decompose the quantization process
        from neural_compressor.torch import quantize

        q_model = quantize(model=self.model, quant_config=quant_config, run_fn=self.run_fn, run_args=self.run_args)
        return q_model


def autotune(model, tune_config, run_fn=None, run_args=None):
    tuner = Tuner(tune_config=tune_config)
    quantizer = TorchQuantizer(model, run_fn, run_args)
    best_qmodel = tuner.search(quantizer=quantizer)
    return best_qmodel


class TuningConfig(BaseTuningConfig):
    def __init__(self, tuning_order=None, timeout=0, max_trials=100):
        super().__init__(tuning_order, timeout, max_trials)
