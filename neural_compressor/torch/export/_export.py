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

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch._export import capture_pre_autograd_graph
from torch.fx.graph_module import GraphModule

from neural_compressor.common.utils import logger
from neural_compressor.torch.utils import TORCH_VERSION_2_2_2, get_torch_version, is_ipex_imported

__all__ = ["export", "export_model_for_pt2e_quant"]


def export_model_for_pt2e_quant(
    model: torch.nn.Module,
    example_inputs: Tuple[Any],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
) -> Optional[GraphModule]:
    """Export the eager model into model with Aten IR."""
    assert isinstance(example_inputs, tuple), f"Expected `example_inputs` to be a tuple, got {type(example_inputs)}"
    # Set the model to eval mode
    model = model.eval()
    exported_model = None
    try:
        with torch.no_grad():
            # Note 1: `capture_pre_autograd_graph` is also a short-term API, it will be
            # updated to use the official `torch.export` API when that is ready.
            cur_version = get_torch_version()
            if cur_version <= TORCH_VERSION_2_2_2:  # pragma: no cover
                logger.warning(
                    (
                        "`dynamic_shapes` is not supported in the current version(%s) of PyTorch,"
                        "If you want to use `dynamic_shapes` to export model, "
                        "please upgrade to 2.3.0 or later."
                    ),
                    cur_version,
                )
                exported_model = capture_pre_autograd_graph(model, args=example_inputs)
            else:
                exported_model = capture_pre_autograd_graph(  # pylint: disable=E1123
                    model, args=example_inputs, dynamic_shapes=dynamic_shapes
                )
            exported_model._exported = True
            logger.info("Exported the model to Aten IR successfully.")
    except Exception as e:
        logger.error(f"Failed to export the model: {e}")

    return exported_model


def export(
    model: torch.nn.Module,
    example_inputs: Tuple[Any],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
) -> Optional[GraphModule]:
    if not is_ipex_imported():
        return export_model_for_pt2e_quant(model, example_inputs, dynamic_shapes)
    else:
        # TODO, add `export` for ipex
        pass
