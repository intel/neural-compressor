#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""Tensorflow scaling model weights and activations for Smooth Quantization."""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes, tensor_util

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer

logger = logging.getLogger("neural_compressor")


class SmoothQuantScaler:
    """A class for scaling model weights using Smooth Quantization method.

    Args:
        model: Tensorflow model to be scaled
        dataloader: Tensorflow dataloader for the dataset
        alpha: float, the scaling factor
        scales_per_op: bool, each op will have an individual scale or
                       ops with the same input will share a scale
    """

    def __init__(self, model, dataloader, alpha, scales_per_op):
        """Initialization."""
        self.model = model
        self.dataloader = dataloader
        self.alpha = alpha
        self.scales_per_op = scales_per_op
        self.mul_list = []
        self.g_analyzer = GraphAnalyzer()
        self.g_analyzer.graph = self.model

    def _adjust_activation(self, scale, input_node_name, output_node_name, w_i):
        """Insert the Mul node after the activation before the weight node.

        Args:
            scale: smooth scale with the shape (ic,)
            input_node_name: the parent input node
            output_node_name: the concrete output weight node name
            w_i: distinguish between different output weight nodes on different branches when naming
        """
        from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper

        node_suffix = str(w_i)
        mul_const_node = Helper.create_constant_node(input_node_name + "/scale_mul" + node_suffix, scale, tf.float32)
        mul_node = Helper.create_node(
            "Mul",
            input_node_name + "_mul" + node_suffix,
            [input_node_name + "/scale_mul" + node_suffix, input_node_name],
        )
        Helper.set_attr_dtype(mul_node, "T", dtypes.float32)
        self.mul_list.append(mul_node.name)
        self.g_analyzer.add_node(mul_node, input_node_name, [output_node_name])
        self.g_analyzer.add_node(mul_const_node, None, [input_node_name + "_mul" + node_suffix])

    def _adjust_weight(self, scale, weight_node, original_weight):
        """In-place adjust weight by scale.

        Args:
            scale: smooth scale with the shape (ic,)
            weight_node: reference to the original const weight node
            original_weight: numpy value of the original const weight node
        """
        # scale: (ic,)
        original_shape = original_weight.shape
        if len(original_shape) == 4:  # (fh, hw, ic, oc)
            W = np.transpose(original_weight, [0, 1, 3, 2])  # move input channel to last dimension
            W *= scale
            W = np.transpose(W, [0, 1, 3, 2])  # move input channel back
            weight_node.attr["value"].tensor.CopyFrom(tensor_util.make_tensor_proto(W))
        elif len(original_shape) == 2:  # (ic, oc) if transpose_a == transpose_b == false
            W = np.transpose(original_weight, [1, 0])
            W *= scale
            W = np.transpose(W, [1, 0])
            weight_node.attr["value"].tensor.CopyFrom(tensor_util.make_tensor_proto(W))

    def transform(self, max_vals_per_channel, sq_weight_tensors, sq_weights_nodes, sq_weight_node_names):
        """Apply scaling to weights and activations based on the maximum values per channel.

        Args:
            max_vals_per_channel (dict): A dictionary containing the maximum values per channel for each input node.
            sq_weight_tensors (dict): A dictionary containing the name -> weight tensors mapping for each input node.
            sq_weights_nodes (dict): A dictionary containing the name -> constant nodes mapping for each input node.
            sq_weight_node_names (dict): A dictionary from weight node name to the its concrete output node name.

        Returns:
            tuple: A tuple containing the modified model and a list of the inserted multiplication nodes.
        """
        logger.info("Start scaling on model graph for Smooth Quantization.")
        if self.scales_per_op:
            # 1. obtain the smooth scale per op
            # 2. adjust weight
            # 3. adjust activation
            for idx, input_node_name in enumerate(max_vals_per_channel):
                A_max_per_in_channel = max_vals_per_channel[input_node_name]
                W_dict = sq_weight_tensors[input_node_name]
                # Use the const nodes before to get weight values
                W_const_node_dict = sq_weights_nodes[input_node_name]
                # Get the concrete weight node as the output of Mul insertion
                for w_i, W_name in enumerate(W_dict):
                    W = W_dict[W_name]
                    if len(W.shape) == 4:
                        # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                        # weight: [filter_height, filter_width, in_channels, out_channels]
                        # activation: NHWC, also batch_shape + [in_height, in_width, in_channels]
                        tensor = np.abs(np.transpose(W, [0, 1, 3, 2]))
                        # reduce weight max to (in_channel, ), aligned with activation max
                        W_max_per_in_channel = np.max(np.reshape(tensor, (-1, tensor.shape[-1])), axis=0)
                    elif len(W.shape) == 2:  # matmul
                        # reduce weight max to (in_channel, ), aligned with activation max
                        tensor = np.abs(W)
                        W_max_per_in_channel = np.max(tensor, axis=1)
                    else:  # pragma: no cover
                        assert False, "not supported"
                    cur_const_node = W_const_node_dict[W_name]
                    try:
                        scale = np.power(A_max_per_in_channel, self.alpha) / np.power(
                            W_max_per_in_channel, (1 - self.alpha)
                        )
                    except ValueError as e:  # pragma: no cover
                        logger.info(e)
                        logger.info("Skip smoothing the node: {}".format(cur_const_node.name))
                        continue
                    # clip the scales that are too small
                    scale = np.clip(scale, a_min=1e-5, a_max=1e8)
                    # skip smoothing the op where scale has elements that less than 1
                    # if np.any(scale < 1):
                    #     logger.info("skip smooth quant: {}".format(input_node_name))
                    #     continue
                    self._adjust_weight(scale, cur_const_node, W)
                    self._adjust_activation(1 / scale, input_node_name, sq_weight_node_names[cur_const_node.name], w_i)
        else:
            pass
        sq_graph_def = self.g_analyzer.dump_graph()
        sq_graph_def.library.CopyFrom(self.model.graph_def.library)
        self.model.graph_def = sq_graph_def
        return self.model, self.mul_list


class SmoothQuantScalerLLM(SmoothQuantScaler):
    """A class for scaling model weights for TF LLM models using Smooth Quantization method.

    Args:
        graph_def: graph_def of the model to be scaled
        alpha: float, the scaling factor
        scales_per_op: bool, each op will have an individual scale or
                       ops with the same input will share a scale
        op_types:
    """

    def __init__(self, graph_def, alpha, scales_per_op, op_types):
        """Initialization."""
        self.graph_def = graph_def
        self.alpha = alpha
        self.scales_per_op = scales_per_op
        self.op_types = op_types

        self.graph_info = None
        self.mul_list = []
        self.sq_weight_scale_dict = {}

    def _parse_weight_dict(self, max_vals_per_channel, sq_weight_tensor_dict):
        """Parse weight related dictionaries to two required dictionaries.

        Args:
            max_vals_per_channel (dict): A dictionary containing the maximum values per channel.
            sq_weight_tensor_dict (dict): A dictionary containing tensor of weights.

        Returns:
            sq_weight_tensors: A dictionary whose structure is like {input_node_name: weight_tensor}}.
            sq_weights_node_names: A dictionary whose structure is like {input_node_name: weight_node_name}}.
        """
        sq_weight_tensors = {}
        sq_weight_node_names = {}
        for input_node_name in max_vals_per_channel:
            curr_weight_tensors = []
            curr_weights_node_names = []
            next_node_names = self.graph_info[input_node_name].outputs
            for node_name in next_node_names:
                curr_node = self.graph_info[node_name].node
                if curr_node.op not in self.op_types:
                    continue
                if len(curr_node.input) >= 2:
                    weight_name = curr_node.input[1]
                    weight_tensor = sq_weight_tensor_dict[weight_name]
                    curr_weight_tensors.append(weight_tensor)
                    curr_weights_node_names.append(weight_name)
            sq_weight_tensors[input_node_name] = curr_weight_tensors
            sq_weight_node_names[input_node_name] = curr_weights_node_names
        return sq_weight_tensors, sq_weight_node_names

    def transform(self, max_vals_per_channel, sq_weight_tensor_dict, sq_target_node_names):
        """Apply scaling to weights and activations based on the maximum values per channel.

        Args:
            max_vals_per_channel (dict): A dictionary containing the maximum values per channel for each input node.
            sq_weight_tensor_dict (dict): A dictionary whose structure is like {input_node_name: weight_tensor}.
            sq_target_node_names (dict): A dictionary whose structure is like {weight_node_name: target_node_name}.
        """
        self.g_analyzer = GraphAnalyzer()
        self.g_analyzer.graph = self.graph_def
        self.graph_info = self.g_analyzer.parse_graph()
        sq_weight_tensors, sq_weight_node_names = self._parse_weight_dict(max_vals_per_channel, sq_weight_tensor_dict)
        logger.info("Start scaling on model graph for Smooth Quantization.")
        if self.scales_per_op:
            # 1. obtain the smooth scale per op
            # 2. adjust weight
            # 3. adjust activation
            for _, input_node_name in enumerate(max_vals_per_channel):
                activation_max_per_in_channel = max_vals_per_channel[input_node_name]
                W_lst = sq_weight_tensors[input_node_name]  # VQK weight value
                # Use the const nodes before to get weight values, VQK ReadVariable
                W_node_name_lst = sq_weight_node_names[input_node_name]
                # Get the concrete weight node as the output of Mul insertion, QKV ReadVariable
                for w_i, W in enumerate(W_lst):
                    if len(W.shape) == 4:
                        # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                        # weight: [filter_height, filter_width, in_channels, out_channels]
                        # activation: NHWC, also batch_shape + [in_height, in_width, in_channels]
                        tensor = np.abs(np.transpose(W, [0, 1, 3, 2]))
                        # reduce weight max to (in_channel, ), aligned with activation max
                        W_max_per_in_channel = np.max(np.reshape(tensor, (-1, tensor.shape[-1])), axis=0)
                    elif len(W.shape) == 2:  # matmul
                        # reduce weight max to (in_channel, ), aligned with activation max
                        tensor = np.abs(W)
                        W_max_per_in_channel = np.max(tensor, axis=1)
                    else:  # pragma: no cover
                        assert False, "not supported"
                    cur_weight_node_name = W_node_name_lst[w_i]
                    try:
                        scale = np.power(activation_max_per_in_channel, self.alpha) / np.power(
                            W_max_per_in_channel, (1 - self.alpha)
                        )
                    except ValueError as e:  # pragma: no cover
                        logger.info(e)
                        logger.info("Skip smoothing the node: {}".format(cur_weight_node_name))
                        continue
                    # clip the scales that are too small
                    scale = np.clip(scale, a_min=1e-5, a_max=1e8)
                    # skip smoothing the op where scale has elements that less than 1
                    # if np.any(scale < 1):
                    #     logger.info("skip smooth quant: {}".format(input_node_name))
                    #     continue
                    self.sq_weight_scale_dict[cur_weight_node_name] = scale
                    self._adjust_activation(1 / scale, input_node_name, sq_target_node_names[cur_weight_node_name], w_i)
        else:
            pass
        sq_graph_def = self.g_analyzer.dump_graph()
        sq_graph_def.library.CopyFrom(self.graph_def.library)
        return sq_graph_def, self.sq_weight_scale_dict, self.mul_list
