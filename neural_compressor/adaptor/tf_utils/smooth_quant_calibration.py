#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
"""Tensorflow model calibration process for Smooth Quantization."""

import logging
import os
from collections import OrderedDict, UserDict

import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util

from .quantize_graph_common import QuantizeGraphHelper
from .util import iterator_sess_run

logger = logging.getLogger("neural_compressor")
debug = bool(logger.level == logging.DEBUG)


class SmoothQuantCalibration:
    """A class for performing smooth quantization calibration on a Tensorflow model.

    Args:
        model (Model): The Tensorflow wrapper model to be calibrated.
        dataloader (DataLoader): The data loader for the calibration dataset.
        iterations (int): The number of iterations to run the calibration process.
        op_types (List[str]): The types of operations to be quantized.
        percentile (float): The percentile of calibration to remove outliers.
        black_nodes (List[str]): A list of node names to be ignored during calibration.
    """

    def __init__(self, model, dataloader, iterations, op_types, percentile, black_nodes):
        """Initializes a SmoothQuantCalibration object."""
        self.model = model
        self.dataloader = dataloader
        self.iterations = iterations
        # self.iterations = 3
        self.op_types = op_types
        self.percentile = percentile
        self.black_nodes = black_nodes
        self._sq_input_node_names = []
        self._sq_output_tensor_dict = {}
        self._sq_weight_node_names = {}  # mapping from its weight node name to the concrete output node name

    def _inference_for_calibration(self, model):
        """Run the calibration on the input graph.

        Args:
            model(TensorflowBaseModel): input TensorflowBaseModel
        """
        # ITEX optimization has broken INC calibration process.
        # INC needs turn off ITEX optimization pass in calibration stage.
        # TODO ITEX will provide API to replace setting environment variable.
        os.environ["ITEX_REMAPPER"] = "0"
        sess = model.sess
        iter_op = model.iter_op
        input_tensor = model.input_tensor
        output_tensor = [item + ":0" for item in self._sq_input_node_names]
        # TF table initialization: https://github.com/tensorflow/tensorflow/issues/8665
        node_names = [node.name for node in sess.graph.as_graph_def().node]
        if "init_all_tables" in node_names:  # pragma: no cover
            init_table_op = sess.graph.get_operation_by_name("init_all_tables")
            sess.run(init_table_op)

        logger.info("Start sampling on calibration dataset for Smooth Quantization.")
        if hasattr(self.dataloader, "__len__") and len(self.dataloader) == 0:  # pragma: no cover
            feed_dict = {}
            for output_idx, output in enumerate(
                sess.run(output_tensor, feed_dict)
                if iter_op == []
                else iterator_sess_run(sess, iter_op, feed_dict, output_tensor, self.iterations)
            ):
                self._sq_output_tensor_dict.setdefault(self._sq_input_node_names[output_idx], []).append(output)
        for idx, (inputs, labels) in enumerate(self.dataloader):
            if len(input_tensor) == 1:
                feed_dict = {}
                if (
                    isinstance(inputs, dict) or isinstance(inputs, OrderedDict) or isinstance(inputs, UserDict)
                ):  # pragma: no cover
                    for name in inputs:
                        for tensor in input_tensor:
                            pos = tensor.name.rfind(":")
                            t_name = tensor.name if pos < 0 else tensor.name[:pos]
                            if name == t_name:
                                feed_dict[tensor] = inputs[name]
                                break
                else:
                    feed_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
            else:  # pragma: no cover
                assert len(input_tensor) == len(inputs), "inputs len must equal with input_tensor"
                feed_dict = {}
                if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) or isinstance(inputs, UserDict):
                    for name in inputs:
                        for tensor in input_tensor:
                            pos = tensor.name.rfind(":")
                            t_name = tensor.name if pos < 0 else tensor.name[:pos]
                            if name in [tensor.name, t_name]:
                                feed_dict[tensor] = inputs[name]
                                break
                else:
                    # sometimes the input_tensor is not the same order with inputs
                    # we should check and pair them
                    def check_shape(tensor, data):
                        # scalar or 1 dim default True
                        if (
                            tensor.shape is None
                            or tensor.shape.dims is None
                            or len(tensor.shape.dims) == 1
                            or not hasattr(data, "shape")
                        ):
                            return True
                        tensor_shape = tuple(tensor.shape)
                        data_shape = tuple(data.shape)
                        for tensor_dim, data_dim in zip(tensor_shape, data_shape):
                            if tensor_dim is not None and tensor_dim != data_dim:
                                return False
                        return True

                    disorder_tensors = []
                    disorder_inputs = []
                    for idx, sort_tensor in enumerate(input_tensor):
                        sort_input = inputs[idx]
                        if check_shape(sort_tensor, sort_input):
                            feed_dict.update({sort_tensor: sort_input})
                        else:
                            disorder_tensors.append(sort_tensor)
                            disorder_inputs.append(sort_input)
                    for i, dis_tensor in enumerate(disorder_tensors):
                        for j, dis_input in enumerate(disorder_inputs):
                            if check_shape(dis_tensor, dis_input):
                                feed_dict.update({dis_tensor: dis_input})
                                break
            for output_idx, output in enumerate(
                sess.run(output_tensor, feed_dict)
                if iter_op == []
                else iterator_sess_run(sess, iter_op, feed_dict, output_tensor, self.iterations)
            ):
                self._sq_output_tensor_dict.setdefault(self._sq_input_node_names[output_idx], []).append(output)
            if idx + 1 == self.iterations:
                break
        os.environ["ITEX_REMAPPER"] = "1"

    def _generate_calibration_data(self):
        """Generate the calibration data."""
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(
            self.model.graph_def, self.model.input_node_names, self.model.output_node_names
        )

        for node in sorted_graph.node:
            if node.op not in self.op_types or node.name in self.black_nodes:
                continue
            # Fix retval already been set issue
            if "while" in node.input[0]:  # pragma: no cover
                continue
            self._sq_input_node_names.append(node.input[0])
            self._sq_weight_node_names[node.input[1]] = node.name

        self._inference_for_calibration(self.model)

    def _get_maxval_per_channel(self, tensor_data, percentile):
        """Get the max values per input channel.

        Args:
            tensor_data: The input tensors
            percentile: The percentile of calibration to remove outliers

        Returns:
            The max values per input channel
        """
        permute_datas = []
        for data in tensor_data:  # iteration_num * (N, H, W, C)
            if len(data.shape) == 3:  # pragma: no cover
                # TODO  matmul batchsize*seq*inchannel
                tensor = np.abs(np.reshape(data, (-1, data.shape[-1])))
                permute_datas.append(tensor)
            elif len(data.shape) == 4:  # already NHWC
                # tensor = np.transpose(data, [0, 3, 1, 2])
                tensor = data
                tensor = np.abs(np.reshape(tensor, (-1, tensor.shape[-1])))
                permute_datas.append(tensor)
            elif len(data.shape) == 2:  # (?, ic)
                permute_datas.append(np.abs(data))
            else:  # pragma: no cover
                assert False, "not supported"
        permute_datas = np.concatenate(permute_datas, axis=0)
        permute_datas = permute_datas.reshape(-1, permute_datas.shape[-1])
        # try:
        #     np.percentile(permute_datas, percentile, axis=0)
        # except FloatingPointError:
        #     indexes = [i for i,e in enumerate(np.percentile(permute_datas, percentile, axis=0)) if np.isnan(e)][0]
        #     np.seterr(all='warning')
        max_per_channels = np.percentile(permute_datas, percentile, axis=0)
        # max_per_channels = np.max(permute_datas, axis=0)
        max_per_channels = max_per_channels.astype(np.single)
        return max_per_channels

    def __call__(self):
        """Generates calibration data and calculate the maximum values per channel.

        Returns:
            max_vals_per_channel (dict): A dictionary containing the maximum values per channel.
            shape_infos (dict): A dictionary containing the shape information.
        """
        self._generate_calibration_data()
        max_vals_per_channel = {}
        for key in self._sq_output_tensor_dict.keys():
            max_val_per_channel = self._get_maxval_per_channel(
                self._sq_output_tensor_dict[key], percentile=self.percentile
            )
            max_vals_per_channel[key] = max_val_per_channel
        return max_vals_per_channel, self._sq_weight_node_names
