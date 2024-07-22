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

# Some code snippets are taken from the X86InductorQuantizer tutorial.
# https://pytorch.org/tutorials/prototype/pt2e_quant_x86_inductor.html
"""The quantizer using PT2E path."""

from typing import Any

import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.fx.graph_module import GraphModule

from neural_compressor.common.utils import logger
from neural_compressor.torch.algorithms.base_algorithm import Quantizer
from neural_compressor.torch.algorithms.pt2e_quant import half_precision_rewriter as hp_rewriter
from neural_compressor.torch.algorithms.pt2e_quant.utility import create_xiq_quantizer_from_pt2e_config


class W8A8PT2EQuantizer(Quantizer):
    """The W8A8 quantizer using PT2E."""

    is_dynamic = False

    def __init__(self, quant_config=None):
        """Initialize the quantizer."""
        super().__init__(quant_config)

    @staticmethod
    def update_quantizer_based_on_quant_config(quant_config=None) -> X86InductorQuantizer:
        """Updates the quantizer based on the given quantization configuration.

        Args:
            quant_config (dict): The quantization configuration. Defaults to None.

        Returns:
            X86InductorQuantizer: The updated quantizer object.
        """
        if not quant_config:
            quantizer = X86InductorQuantizer()
            quantizer.set_global(
                xiq.get_default_x86_inductor_quantization_config(is_dynamic=W8A8PT2EQuantizer.is_dynamic)
            )
        else:
            quantizer = create_xiq_quantizer_from_pt2e_config(quant_config, is_dynamic=W8A8PT2EQuantizer.is_dynamic)
        return quantizer

    def prepare(self, model: GraphModule, example_inputs=None, inplace=True, *args, **kwargs) -> GraphModule:
        """Prepares the model for calibration.

        Create the `quantizer` according to the `quant_config`, and insert the observers accordingly.

        Args:
            model (GraphModule): The model to be prepared for calibration.
            example_inputs (tuple, optional): Example inputs to be used for calibration. Defaults to None.
            inplace (bool, optional): Whether to modify the model in-place or return a new prepared model.
                Defaults to True.

        Returns:
            GraphModule: The prepared model.
        """
        quant_config = self.quant_config
        assert model._exported, "The model should be exported before preparing it for calibration."
        quantizer = self.update_quantizer_based_on_quant_config(quant_config)
        prepared_model = prepare_pt2e(model, quantizer)
        return prepared_model

    def convert(self, model: GraphModule, *args: Any, **kwargs: Any) -> GraphModule:
        """Convert the calibrated model into qdq mode.

        Args:
            model (GraphModule): The prepared model.

        Returns:
            GraphModule: The converted quantized model.
        """
        fold_quantize = kwargs.get("fold_quantize", False)
        converted_model = convert_pt2e(model, fold_quantize=fold_quantize)
        logger.warning("Converted the model in qdq mode, please compile it to accelerate inference.")
        if self.quant_config:
            self.half_precision_transformation(converted_model, self.quant_config)
        return converted_model

    def half_precision_transformation(self, model, config):
        """Applies half-precision transformation to the given model in-place.

        Args:
            model: The model to apply the transformation to.
            config: The configuration for the transformation.
        """
        half_precision_node_set = hp_rewriter.get_half_precision_node_set(model, config)
        logger.info("Try to convert %d nodes to half precision.", len(half_precision_node_set))
        hp_rewriter.transformation(model, half_precision_node_set)
