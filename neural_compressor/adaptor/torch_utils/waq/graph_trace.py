#
# -*- coding: utf-8 -*-
#
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
#
try:
    from neural_compressor.utils.utility import LazyImport

    torch = LazyImport("torch")
    from neural_compressor.utils import logger
except:
    import logging

    import torch

    logger = logging.getLogger()
from collections import UserDict

from .utils import get_module, move_input_to_device


def get_parent(node, all_parents=False):
    if node.inputs() is None:
        return None
    elif len(list(node.inputs())) == 0:
        return None
    if not all_parents:
        return list(node.inputs())[0].node()
    else:
        return list(node.inputs())


class GraphTrace:
    """"""

    def __init__(self):
        self.supported_torch_module_to_aten = {
            "Linear": "aten::linear",
            "Conv2d": "aten::_convolution",
            "ConvTranspose2d": "aten::_convolution",
            "LayerNorm": "aten::layer_norm",
            "BatchNorm2d": "aten::batch_norm",
            "GroupNorm": "aten::group_norm",
            "InstanceNorm2d": "aten::instance_norm",
            "LlamaRMSNorm": "aten::mul",
            "T5LayerNorm": "aten::mul",
            "LPLayerNorm": "aten::layer_norm",  ##mpt_chat
        }

        ##TODO potential bug, need to check only have one bug
        ##TODO, must satisfy af(x)=f(ax),current skip layer may be incomplete
        self.skip_ops_to_find_absorb = ["aten::to", "aten::relu", "aten::leaky_relu", "aten::hardtanh"]

        self.could_absorb_layers = [
            "aten::layer_norm",
            "aten::batch_norm",
            "aten::linear",
            "aten::_convolution",
            "aten::group_norm",
            "aten::instance_norm",
            "aten::mul",
        ]  ##TODO,support more norm

    def trace(self, model, dummy_input):
        traced_model = None
        optimize_numerics = False
        orig_device = str(next(model.parameters()).device)
        if orig_device != "cpu" and orig_device != "meta":  # pragma: no cover
            model = model.to("cpu")
            dummy_input = move_input_to_device(dummy_input, "cpu")
        if isinstance(dummy_input, dict) or isinstance(dummy_input, UserDict):
            try:
                traced_model = torch.jit.trace(
                    model, example_kwarg_inputs=dict(dummy_input), strict=False, check_trace=False
                )
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except Exception as e:
                logger.warning(e)
                logger.warning("Jit trace in GraphTrace failed, absorb layer detection is skipped")
        else:
            try:
                traced_model = torch.jit.trace(model, dummy_input, strict=False)
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except:
                try:
                    traced_model = torch.jit.trace(model, dummy_input[0], strict=False)
                    traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
                except Exception as e:
                    logger.warning(e)
                    logger.warning("Jit trace in GraphTrace failed, absorb layer detection is skipped")
        model = model.to(orig_device)
        return traced_model

    def get_nodes(self, traced_model, op_types=["Linear"]):
        if isinstance(op_types, str):
            op_types = [op_types]
        nodes = []
        for node in traced_model.graph.nodes():
            node_type = node.kind()
            for op_type in op_types:
                if node_type == op_type:
                    nodes.append((node, op_type))
                    break
        return nodes

    def get_prev_absorb_layer(self, nodes):
        prev_absorb_layer = []
        for node in nodes:
            parent = get_parent(node)
            while 1:
                if parent.kind() in self.skip_ops_to_find_absorb:
                    parent = get_parent(parent)
                    continue
                if parent.kind() in self.could_absorb_layers:
                    parent_out_kinds = []
                    for val_user in list(parent.outputs())[0].uses():
                        next_node = val_user.user
                        parent_out_kinds.append(next_node.kind())
                    parent_out_kinds = set(parent_out_kinds)
                    parent_out_kinds.discard("aten::size")

                    if parent_out_kinds == parent_out_kinds.intersection(self.could_absorb_layers):
                        prev_absorb_layer.append(parent)
                    elif parent_out_kinds.intersection(self.skip_ops_to_find_absorb):
                        res = self.skip_op_absorb_helper(parent)
                        prev_absorb_layer.append(parent) if res else prev_absorb_layer.append(None)
                    else:  # When parent to multiple ops, sq transformation could be wrong.
                        prev_absorb_layer.append(None)
                else:
                    prev_absorb_layer.append(None)
                break
        return prev_absorb_layer

    def skip_op_absorb_helper(self, parent_node):
        for val_user in list(parent_node.outputs())[0].uses():
            next_node = val_user.user
            if next_node.kind() == "aten::size":
                continue
            elif next_node.kind() in self.could_absorb_layers:
                continue
            elif next_node.kind() in self.skip_ops_to_find_absorb:
                node_res = self.skip_op_absorb_helper(next_node)
                if not node_res:
                    return False
            else:
                return False
        return True

    def mapping_torch_module_to_aten(self, op_types):
        res = []
        for op in op_types:
            if op not in self.supported_torch_module_to_aten.keys():
                logger.warning(f"{op} is not supported in smooth quant, ignoring...")
                continue
            res.append(self.supported_torch_module_to_aten[op])
        res = list(set(res))
        return res

    def _check_valid_conv(self, module):
        """Remove group conv except depthwise conv
        :param module:

        :return:
        """
        if not isinstance(module, torch.nn.Conv2d):
            return True
        if module.groups > 1:
            if module.in_channels == module.out_channels and module.groups == module.in_channels:
                return True
            else:
                return False
        return True

    def get_absorb_to_layer(self, model, example_input, op_types, skip_unsupported_layers=True):
        traced_model = self.trace(model, example_input)
        if traced_model is None:
            return None, None

        aten_op_types = self.mapping_torch_module_to_aten(op_types)
        nodes_types = self.get_nodes(traced_model, aten_op_types)
        nodes = [node_type[0] for node_type in nodes_types]
        nodes_prev_absorb = self.get_prev_absorb_layer(nodes)
        absorb_to_layer = {}
        no_absorb_layers = []
        for index, absorb in enumerate(nodes_prev_absorb):
            if absorb is None:
                no_absorb_layers.append(".".join(nodes[index].scopeName().split("/")[-1].split(".")[1:]))
                continue
            node = nodes[index]
            layer_name = ".".join(node.scopeName().split("/")[-1].split(".")[1:])
            absorb_name = ".".join(absorb.scopeName().split("/")[-1].split(".")[1:])
            if layer_name == "" or absorb_name == "":
                continue
            if absorb_name in absorb_to_layer.keys():
                absorb_to_layer[absorb_name].append(layer_name)
            else:
                absorb_to_layer[absorb_name] = [layer_name]
        if skip_unsupported_layers:
            absorb_to_layer = self.remove_unsupported_layers(model, absorb_to_layer, no_absorb_layers)
        return absorb_to_layer, no_absorb_layers

    def remove_unsupported_layers(self, model, absorb_to_layer, no_absorb_layers):
        res = {}
        for key in absorb_to_layer.keys():
            absorb_layer = get_module(model, key)
            layer_type = absorb_layer.__class__.__name__
            if layer_type not in self.supported_torch_module_to_aten.keys():
                no_absorb_layers.extend(absorb_to_layer[key])
                continue
            supported = True
            for layer_name in absorb_to_layer[key]:
                layer = get_module(model, layer_name)
                layer_type = layer.__class__.__name__
                if (layer_type not in self.supported_torch_module_to_aten.keys()) or not self._check_valid_conv(layer):
                    supported = False
                    no_absorb_layers.extend(absorb_to_layer[key])
                    break
            if supported:
                res[key] = absorb_to_layer[key]
        return res
