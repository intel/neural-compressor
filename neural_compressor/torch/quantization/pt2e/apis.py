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

# Steps for W8A8 static/dynamic PTQ and QAT

# 1. Load the eager model and prepare example inputs
# 2. Capture graph at Aten IR
# 2. Prepare graph (insert observers)
# 3. Calibrate/Train
# 4. Convert graph (remove observer ops and insert q/dq pairs)
# 5. Lowering graph (fuse q/qd pairs to q op)

from typing import Any, Tuple

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torchvision import models


def get_model_and_example_inputs(model_name) -> Tuple[torch.nn.Module, Tuple[Any]]:
    # Create the Eager Model

    model = models.__dict__[model_name](pretrained=True)

    # Set the model to eval mode
    model = model.eval()

    # Create the data, using the dummy data here as an example
    traced_bs = 50
    x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)

    example_inputs = (x,)
    return model, example_inputs


################### Quantization Code ###################
is_dynamic = False
is_qat = False

# 1. Prepare the eager model and example inputs
model_name = "resnet18"
model, example_inputs = get_model_and_example_inputs(model_name)

# 2. Export model at aten IR
exported_model = capture_pre_autograd_graph(model, args=example_inputs)


# 3. Insert observers into exported model
quantizer = X86InductorQuantizer()
quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=is_qat, is_dynamic=is_dynamic))
if is_qat:
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
else:
    # static/dynamic
    prepared_model = prepare_pt2e(exported_model, quantizer)

# 4. Calibrate/Train
prepared_model(*example_inputs)

# 5. Convert
converted_model = convert_pt2e(prepared_model)

# 6. Lowering
optimized_model = torch.compile(converted_model)
