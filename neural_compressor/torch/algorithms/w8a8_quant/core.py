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

# Note - The `W8A8StaticQuantizer` is aligned with with the torch.ao's unified quantization API.
# https://github.com/pytorch-labs/ao/blob/5401df093564825c06691f4c2c10cdcf1a32a40c/torchao/quantization/unified.py#L15-L26
# Some code snippets are taken from the X86InductorQuantizer tutorial.
# https://pytorch.org/tutorials/prototype/pt2e_quant_x86_inductor.html


from typing import Any, Tuple

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.fx.graph_module import GraphModule

from neural_compressor.common.utils import logger


class W8A8StaticQuantizer:

    @staticmethod
    def update_quantizer_based_on_quant_config(quantizer: X86InductorQuantizer, quant_config) -> X86InductorQuantizer:
        # TODO: add the logic to update the quantizer based on the quant_config
        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
        return quantizer

    def prepare(
        self, model: torch.nn.Module, quant_config, example_inputs: Tuple[Any], *args: Any, **kwargs: Any
    ) -> GraphModule:
        """Prepare the model for calibration.

        There are two steps in this process:
            1) export the eager model into model with Aten IR.
            2) create the `quantizer` according to the `quant_config`, and insert the observers accordingly.
        """
        assert isinstance(example_inputs, tuple), f"Expected `example_inputs` to be a tuple, got {type(example_inputs)}"
        # Set the model to eval mode
        model = model.eval()

        # 1) Capture the FX Graph to be quantized
        with torch.no_grad():
            # Note 1: `capture_pre_autograd_graph` is also a short-term API, it will be
            # updated to use the official `torch.export` API when that is ready.
            exported_model = capture_pre_autograd_graph(model, example_inputs)

        # 2) create the `quantizer` according to the `quant_config`, and insert the observers accordingly.
        quantizer = X86InductorQuantizer()
        quantizer = self.update_quantizer_based_on_quant_config(quantizer, quant_config)
        prepared_model = prepare_pt2e(exported_model, quantizer)
        return prepared_model

    def convert(self, model: torch.fx.GraphModule, *args: Any, **kwargs: Any) -> GraphModule:
        """Convert the calibrated model into qdq mode."""
        fold_quantize = kwargs.get("fold_quantize", False)
        converted_model = convert_pt2e(model, fold_quantize=fold_quantize)
        logger.warning("Converted the model in qdq mode, please compile it to accelerate inference.")
        return converted_model


class TestW8A8StaticQuantizer:
    def test_quantizer_on_llm(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
        example_inputs = (input_ids,)
        quant_config = None
        w8a8_static_quantizer = W8A8StaticQuantizer()
        # prepare
        prepare_model = w8a8_static_quantizer.prepare(model, quant_config, example_inputs=example_inputs)
        # calibrate
        for i in range(2):
            prepare_model(*example_inputs)
        # convert
        converted_model = w8a8_static_quantizer.convert(prepare_model)
        # inference
        from torch._inductor import config

        config.freezing = True
        opt_model = torch.compile(converted_model)
        out = opt_model(*example_inputs)
        assert out is not None
