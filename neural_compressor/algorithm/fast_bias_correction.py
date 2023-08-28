#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
"""Build FastBiasCorrection algorithm class."""

import numpy as np

from ..utils import logger
from .algorithm import Algorithm, algorithm_registry


@algorithm_registry(algorithm_type="fast_bias_correction", location="post_quantization")
class FastBiasCorrection(Algorithm):
    """FastBiasCorrection algorithm class."""

    def __init__(self, threshold=2.0, channel_axis=1):
        """Initialize FastBiasCorrection class.

        Args:
            threshold (float, optional): threshold. Defaults to 2.0.
            channel_axis (int, optional): channel_axis. Defaults to 1.
        """
        self.threshold = threshold
        self.channel_axis = channel_axis
        self.quantization_cfg = None

    def __call__(self, origin_model, q_model, adaptor, dataloader, iterations):
        """Return the processed model via FastBiasCorrection algorithm.

        Args:
            origin_model: origin_model
            q_model: q_model
            adaptor: adaptor
            dataloader: dataloader
            iterations: iterations

        Returns:
            model : The processed model
        """
        # (TODO) assume int8 model also use fp32 op list
        # in adaptor fp32 op will be mapped to corresponding int8 op
        graph_info = origin_model.graph_info
        op_list = [op_name for op_name, op_type in graph_info.items() if "conv" in op_type.lower()]
        iteration_list = list(range(1, iterations + 1))
        fp32_data = adaptor.inspect_tensor(
            origin_model.graph_def,
            dataloader,
            op_list=op_list,
            iteration_list=iteration_list,
            inspect_type="all",
            save_to_disk=False,
            save_path="",
            quantization_cfg=self.quantization_cfg,
        )
        q_data = adaptor.inspect_tensor(
            q_model.graph_def,
            dataloader,
            op_list=op_list,
            iteration_list=iteration_list,
            inspect_type="all",
            save_to_disk=False,
            save_path="",
            quantization_cfg=self.quantization_cfg,
        )

        fp32_weights = fp32_data["weight"]
        q_weights = q_data["weight"]
        fp32_activations_list = fp32_data["activation"]
        q_activations_list = q_data["activation"]

        def take_out_array(value_dict):
            value_list = []
            for name, value in value_dict.items():
                if isinstance(value, dict):
                    value = take_out_array(value)
                value_list.append(value)
            return np.concatenate(value_list)

        # node_mapping = adaptor.mapping(q_model, fp32_model)
        fp32_activations = {}
        for i, _ in enumerate(iteration_list):
            for name, value in fp32_activations_list[i].items():
                if isinstance(name, tuple):
                    name = name[0]
                if name in fp32_activations:
                    fp32_activations[name] = np.concatenate((fp32_activations[name], take_out_array(value)))
                else:
                    fp32_activations[name] = take_out_array(value)

        q_activations = {}
        for i, _ in enumerate(iteration_list):
            for name, value in q_activations_list[i].items():
                if isinstance(name, tuple):
                    name = name[0]
                if name in q_activations:
                    q_activations[name] = np.concatenate((q_activations[name], take_out_array(value)))
                else:
                    q_activations[name] = take_out_array(value)
        tensor_dict = {}
        # for fp32_op, q_op in node_mapping.items():
        for fp32_op in op_list:
            # (TODO) assume adaptor will map the fp32_op to q_op, so directly assign here
            q_op = fp32_op
            # (TODO) assume fp32 op output and weight all mapped from the first node name
            # fp32 op and quantized op should all have bias
            if fp32_op not in fp32_weights or not len(fp32_weights[fp32_op]) == 2:
                continue

            fp32_weight, fp32_weight_name = None, ""
            fp32_bias, fp32_bias_name = None, ""
            for name, value in fp32_weights[fp32_op].items():
                if len(value.shape) > 1:
                    fp32_weight = value
                    fp32_weight_name = name
                if len(value.shape) == 1:
                    fp32_bias = value
                    fp32_bias_name = name

            q_weight, q_weight_name = None, ""
            q_bias, q_bias_name = None, ""
            for name, value in q_weights[q_op].items():
                if len(value.shape) > 1:
                    q_weight = value
                    q_weight_name = name
                if len(value.shape) == 1:
                    q_bias = value
                    q_bias_name = name

            # (TODO) assume use conv output first tensor
            fp32_output = fp32_activations[fp32_op]
            q_output = q_activations[q_op]

            bias_shift = fp32_output - q_output
            # transpose the channel_axis to first
            bias_shape = list(range(len(bias_shift.shape)))
            transpose_bias_shape = [bias_shape.pop(self.channel_axis)]
            transpose_bias_shape.extend(bias_shape)
            bias_shift = bias_shift.transpose(transpose_bias_shape)
            bias_shift = bias_shift.reshape(bias_shift.shape[0], -1)
            bias_shift = np.mean(bias_shift, axis=1)

            tensor_dict[q_bias_name] = fp32_bias + bias_shift

        if len(tensor_dict) > 0:
            adaptor.set_tensor(q_model, tensor_dict)

        return q_model
