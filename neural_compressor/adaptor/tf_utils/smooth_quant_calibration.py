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

import copy
import logging
import os
import tempfile
import time
from collections import OrderedDict, UserDict

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2
from tensorflow.python.framework import dtypes, tensor_util
from tensorflow.python.saved_model import load, tag_constants

from neural_compressor import Model
from neural_compressor.utils.utility import CaptureOutputToFile

from .graph_util import GraphAnalyzer
from .graph_util import GraphRewriterHelper as Helper
from .quantize_graph_common import QuantizeGraphHelper
from .util import iterator_sess_run, parse_saved_model, reconstruct_saved_model

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
    """

    def __init__(self, model, dataloader, iterations, op_types, percentile):
        """Initializes a SmoothQuantCalibration object."""
        self.model = model
        self.dataloader = dataloader
        self.iterations = iterations
        # self.iterations = 3
        self.op_types = op_types
        self.percentile = percentile
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
            if node.op not in self.op_types:
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
            sq_weight_node_names (dict): A dictionary mapping from weight names to target node names.
        """
        self._generate_calibration_data()
        max_vals_per_channel = {}
        for key in self._sq_output_tensor_dict.keys():
            max_val_per_channel = self._get_maxval_per_channel(
                self._sq_output_tensor_dict[key], percentile=self.percentile
            )
            max_vals_per_channel[key] = max_val_per_channel
        return max_vals_per_channel, self._sq_weight_node_names


class SmoothQuantCalibrationLLM(SmoothQuantCalibration):
    """A class for performing smooth quantization calibration on a Tensorflow LLM model.

    Args:
        model (str): A path to the original Tensorflow model.
        iterations (int): The number of iterations to run the calibration process.
        op_types (List[str]): The types of operations to be quantized.
        percentile (float): The percentile of calibration to remove outliers.
        eval_func (function):  The function to inference the model.
        temp_path (str): The temporary path to store median model.
        weight_name_mapping (): A function that convert weight tensor name in autotrackable to node name in graph_def
    """

    def __init__(self, model_path, dataloader, iterations, op_types, percentile, temp_path, weight_name_mapping):
        """Initializes a SmoothQuantCalibrationLLM object."""
        self.func = None
        self.graph_def = None
        self.frozen_func = None
        self._saved_model = None
        self.model = model_path
        self.dataloader = dataloader
        self.iterations = iterations
        self.op_types = op_types
        self.percentile = percentile
        self.temp_path = temp_path
        self.weight_name_mapping = weight_name_mapping
        self.print_node_list = []
        self._sq_input_node_names = []
        self._sq_target_node_names = {}
        self._sq_output_tensor_dict = {}
        self._sq_weight_tensor_dict = {}

    def _parse_calibration_logs(self, tmp_dump_file):
        """Parse calibration logs for llm saved_model."""
        valid_data = []
        with open(tmp_dump_file) as file:
            for i in file.readlines():
                if i.startswith(";"):
                    valid_data.append(i.strip())

        for activation in valid_data:
            activation = activation.split(" ")
            data = []
            activation_name = ""
            per_channel = []
            for idx, s in enumerate(activation):
                if idx == 0:
                    per_channel.append(float(s.rsplit(":")[-1].strip("[")))
                    activation_name = s.rsplit(":")[0][1:-9]
                elif s.find("][") != -1:
                    pairs = [float(i) for i in s.split("][")]
                    per_channel.append(pairs[0])
                    data.append(per_channel)
                    per_channel = [pairs[1]]
                elif s.find("]]") != -1:
                    per_channel.append(float(s.strip("]")))
                    data.append(per_channel)
                else:
                    per_channel.append(float(s))

            if activation_name not in self._sq_output_tensor_dict:
                self._sq_output_tensor_dict[activation_name] = [np.array(data)]
            else:
                self._sq_output_tensor_dict[activation_name].append(np.array(data))

    def _insert_print_for_activation(self, graph_def):
        """Insert print node in the graph to do the calibration for llm saved_model."""
        cur_graph = GraphAnalyzer()
        cur_graph.graph = graph_def

        graph_info = cur_graph.parse_graph()
        for cur_list in self.print_node_list:
            pre_node_name = cur_list[0]
            post_node_name = cur_list[-1]
            insert_node_pairs = []
            top_node = graph_info[pre_node_name].node
            if top_node.op == "ConcatV2":
                for i in range(top_node.attr["N"].i):
                    insert_node_pairs.append([top_node.input[i], post_node_name])
            elif top_node.op in ("BatchMatMul", "BatchMatMulV2"):
                insert_node_pairs.append([top_node.input[0], post_node_name])
                if graph_info[top_node.input[1]].node.op != "Const":
                    insert_node_pairs.append([top_node.input[1], post_node_name])
            elif top_node.op in ("Conv2DBackpropInput", "Conv3DBackpropInputV2"):
                insert_node_pairs.append([top_node.input[2], post_node_name])
            else:
                refresh_pre_node_name = graph_info[pre_node_name].node.input[0]
                # Check the Conv2D could be fused with previous Pad or not.
                # If so, we need to update the pre-node name correspondingly.
                refresh_pre_node = graph_info[Helper.node_name_from_input(refresh_pre_node_name)].node
                if refresh_pre_node.op == "Pad" and top_node.op in ("Conv2D", "Conv3D"):
                    insert_node_pairs.append([refresh_pre_node_name, post_node_name])
                    refresh_pre_node_name = refresh_pre_node.input[0]

                insert_node_pairs.append([refresh_pre_node_name, post_node_name])

            output_names = []
            for node_pair_names in insert_node_pairs:
                for index, each_node_name in enumerate(node_pair_names):
                    name_with_sig = each_node_name
                    node_name_prefix = name_with_sig.replace(":", "__port__").replace("^", "__hat__")
                    print_node = Helper.create_node(
                        "Print",
                        node_name_prefix + "_print__{}".format(index),
                        [each_node_name + ":0", each_node_name + ":0"],
                    )

                    if index == 0:
                        msg = ";{}__print__:".format(each_node_name)
                        # workaround for swish_f32, attribute T is not in the op definition
                        if "swish_f32" in graph_info[pre_node_name].node.name:
                            src_dt = attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum)
                        else:
                            src_dt = graph_info[pre_node_name].node.attr["T"]
                    else:
                        break

                    print_node.attr["T"].CopyFrom(src_dt)

                    print_node.attr["message"].s = msg.encode()
                    print_node.attr["first_n"].i = -1
                    print_node.attr["summarize"].i = 102400000

                    attr_u = [dtypes.as_dtype(src_dt.type).as_datatype_enum]
                    print_node.attr["U"].list.CopyFrom(attr_value_pb2.AttrValue.ListValue(type=attr_u))
                    post_node_names = graph_info[Helper.node_name_from_input(each_node_name)].outputs
                    if post_node_names:
                        for post_node_name in post_node_names:
                            post_node = graph_info[post_node_name].node
                            if each_node_name not in post_node.input:
                                continue
                            if (
                                post_node.op == "FusedBatchNormV3"
                                and "_print_identity"
                                not in graph_info[Helper.node_name_from_input(post_node.name)].node.input[0]
                            ):
                                identity_node = Helper.create_node(
                                    "Identity",
                                    post_node.name + "_print_identity",
                                    [graph_info[Helper.node_name_from_input(post_node.name)].node.input[0]],
                                )
                                identity_node.attr["T"].CopyFrom(src_dt)
                                cur_graph.add_node(
                                    identity_node,
                                    graph_info[Helper.node_name_from_input(post_node.name)].node.input[0],
                                    [post_node.name],
                                )
                                identity_node.input.append("^" + print_node.name)
                            else:
                                post_node.input.append("^" + print_node.name)

                        cur_graph.add_node(print_node, each_node_name, [])
                    else:
                        identity_node1 = Helper.create_node(
                            "Identity", print_node.name + "_identity", [print_node.name]
                        )
                        identity_node1.attr["T"].CopyFrom(src_dt)
                        cur_graph.add_node(print_node, each_node_name, [identity_node1.name])
                        cur_graph.add_node(identity_node1, print_node.name, [])
                        output_names.append(identity_node1.name)

        return cur_graph.dump_graph()

    def evaluate(self, model):
        """Evaluate function that inference the model to apply calibration.

        Args:
            model (tf.python.trackable.autotrackable): The model to be evaluated.
            The object is usually gotten by using tf.saved_model.load(model_dir) API.

        Returns:
            accuracy (float): The accuracy result.
        """
        input_tensor_names = model.input_tensor_names
        auto_trackable = model.model
        infer = auto_trackable.signatures["serving_default"]
        for idx, (inputs, _) in enumerate(self.dataloader):
            feed_dict = {}
            if len(input_tensor_names) == 1:
                feed_dict[input_tensor_names[0]] = inputs
            else:
                assert len(input_tensor_names) == len(inputs), "inputs len must equal with input_tensor"
                for i, input_tensor_name in enumerate(input_tensor_names):
                    feed_dict[input_tensor_name] = inputs[i]

            _ = infer(**feed_dict)

            if idx >= self.iterations:
                break

    def _inference(self, sampling_graph_def):
        logger.info("Start sampling on calibration dataset for Smooth Quantization.")
        # reconstruct graph_def that inserted print node to saved_model
        reconstruct_saved_model(sampling_graph_def, self.func, self.frozen_func, self._saved_model, self.temp_path)
        model = Model(self.temp_path, modelType="llm_saved_model")
        self.evaluate(model)

    def _inference_for_calibration(self, model):
        """Run the calibration on the input graph."""
        sampling_graph_def = self._insert_print_for_activation(model)
        tmp_dump_file = tempfile.mkstemp(suffix=".log")[1]
        with CaptureOutputToFile(tmp_dump_file):
            self._inference(sampling_graph_def)
        self._parse_calibration_logs(tmp_dump_file)
        del sampling_graph_def

    def _get_weight_tensors(self):
        model = load.load(self.model, [tag_constants.SERVING])
        for weight_tensor in model.variables:
            parsed_name = self.weight_name_mapping(weight_tensor.name)
            if parsed_name in self._sq_target_node_names:
                self._sq_weight_tensor_dict[parsed_name] = weight_tensor.numpy()

        assert len(self._sq_weight_tensor_dict) == len(
            self._sq_target_node_names
        ), "Failed to get weights for some nodes, please check variables"

    def _generate_calibration_data(self, input_node_names, output_node_names):
        """Generate the calibration data."""
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(
            self.graph_def,
            input_node_names,
            output_node_names,
        )

        for node in sorted_graph.node:
            if node.op not in self.op_types:
                continue
            # Fix retval already been set issue
            if "while" in node.input[0]:  # pragma: no cover
                continue
            self._sq_input_node_names.append(node.input[0])
            self.print_node_list.append([node.name])
            self._sq_target_node_names[node.input[1]] = node.name
        self._get_weight_tensors()
        sampling_graph_def = copy.deepcopy(self.graph_def)
        self._inference_for_calibration(sampling_graph_def)

    def __call__(self, input_node_names, output_node_names):
        """Generates calibration data and calculate the maximum values per channel.

        Args:
            input_node_names: (list): A list of names for input nodes.
            output_node_names: (list): A list of names for output nodes.

        Returns:
            max_vals_per_channel (dict): A dictionary containing the maximum values per channel.
            sq_target_node_names (dict): A dictionary mapping from weight names to target node names.
            sq_weight_tensor_dict (dict): A dictionary containing tensor of weights.
        """
        self.graph_def, self._saved_model, self.func, self.frozen_func, _, _ = parse_saved_model(self.model)
        self._generate_calibration_data(input_node_names, output_node_names)
        max_vals_per_channel = {}
        for activation_name, output_tensor in self._sq_output_tensor_dict.items():
            max_val_per_channel = self._get_maxval_per_channel(output_tensor, percentile=self.percentile)
            max_vals_per_channel[activation_name] = max_val_per_channel
        return max_vals_per_channel, self._sq_target_node_names, self._sq_weight_tensor_dict, self.graph_def
