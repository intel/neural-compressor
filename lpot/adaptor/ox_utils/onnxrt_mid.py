#!/usr/bin/env python
# coding: utf-8
#
# Copyright (c) 2021 Intel Corporation
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
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


import copy
import logging

import numpy as np
import onnx
import onnxruntime
from onnx import helper, TensorProto

logger = logging.getLogger()


class ONNXRTAugment:
    '''augment input model to dump tensor or for calibration'''

    def __init__(self, model, dataloader,
                 dump_op_types,
                 augmented_model_path,
                 black_nodes=[],
                 white_nodes=[],
                 iterations=[]):
        '''
        :param model: ONNX model to calibrate
        :param dataloader: user implemented object to read in and preprocess calibration dataset
        :param op_types: operator types to be calibrated and quantized, default = 'Conv,MatMul'
        :param black_nodes: operator names that should not be quantized, default = ''
        :param white_nodes: operator names that force to be quantized, default = ''
        :param augmented_model_path: save augmented_model to this path
        :param iterations: tensor of which iteration will be collected.
        '''
        self.model = model
        self.dataloader = dataloader
        self.dump_op_types = dump_op_types
        self.black_nodes = black_nodes
        self.white_nodes = white_nodes
        self.augmented_model = None
        self.augmented_model_path = augmented_model_path
        self.input_name_to_nodes = {}
        self.iterations = iterations
        self.augment_nodes = []

    def augment_graph(self):
        '''
        Adds nodes to all quantization_candidates op type nodes in
        model and ensures their outputs are stored as part of the graph output
        :return: augmented ONNX model
        '''

        model = copy.deepcopy(self.model)

        added_nodes = []
        added_outputs = []
        tensors_to_dump = set()

        for node in model.graph.node: # pylint: disable=no-member
            should_be_dump = ((node.op_type in self.dump_op_types) and
                                   (node.name not in self.black_nodes)) or \
                                   (node.name in self.white_nodes)
            if should_be_dump:
                if node.op_type == "Attention":
                    if len(node.input) >= 3:
                        logger.debug("indice input {} of attention node {} is in integer format"
                                 .format(node.input[3:], node.name))
                        tensors_to_dump.update(node.input[:2])
                    else:
                        tensors_to_dump.update(node.input)
                elif node.op_type == "Gather":
                    logger.debug("indice input {} of gather node {} is in integer format"
                                 .format(node.input[-1], node.name))
                    tensors_to_dump.update(node.input[:-1])
                else:
                    tensors_to_dump.update(node.input)
                tensors_to_dump.update(node.output)

            for tensor in tensors_to_dump:
                if tensor in model.graph.initializer: # pylint: disable=no-member
                    tensors_to_dump.remove(tensor)


        for tensor in tensors_to_dump:
            if self.augment_nodes:
                for augment_node_type in self.augment_nodes:
                    if augment_node_type not in ['ReduceMin', 'ReduceMax']:
                        raise ValueError("Unexpected augment_node {} only \
                               ReduceMin/ReduceMax are supported".format(augment_node_type))
                    augment_node_name = tensor + "_" + augment_node_type
                    augment_node = onnx.helper.make_node(augment_node_type, [tensor],
                                                         [augment_node_name],
                                                         augment_node_name,
                                                         keepdims=0)
                    added_nodes.append(augment_node)
                    added_outputs.append(helper.make_tensor_value_info(
                                           augment_node.output[0], # pylint: disable=no-member
                                           TensorProto.FLOAT, ())) # pylint: disable=no-member
            else:
                if tensor not in [t.name for t in model.graph.output]:
                    added_tensor = helper.ValueInfoProto()
                    added_tensor.name = tensor
                    added_outputs.append(added_tensor)

        if self.augment_nodes:
            model.graph.node.extend(added_nodes) # pylint: disable=no-member
        model.graph.output.extend(added_outputs) # pylint: disable=no-member

        self.augmented_model = model
        onnx.save(model, self.augmented_model_path)

    def get_intermediate_outputs(self):
        '''
            Gather intermediate model outputs after running inference
            :return: dictionary mapping: {node output tensor names: node output tensor }
        '''

        # conduct inference session and get intermediate outputs
        session = onnxruntime.InferenceSession(self.augmented_model.SerializeToString(), None)

        intermediate_outputs = []

        for idx, batch in enumerate(self.dataloader):
            ort_inputs = {}
            if self.iterations:
                if idx in self.iterations:
                    for i in range(len(session.get_inputs())):
                        ort_inputs.update({session.get_inputs()[i].name: batch[i]})
                    intermediate_outputs.append(session.run(None, ort_inputs))
            else:
                for i in range(len(session.get_inputs())):
                    ort_inputs.update({session.get_inputs()[i].name: batch[i]})
                intermediate_outputs.append(session.run(None, ort_inputs))
        node_output_names = [session.get_outputs()[i].name for i in
                             range(len(intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(node_output_names, intermediate_output)) for intermediate_output in
            intermediate_outputs
        ]

        return node_output_names, output_dicts_list

    def _map_calibration(self, node_output_names, output_dicts_list, calib_mode='naive'):
        model = self.model
        num_model_outputs = len(model.graph.output)
        merged_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)
        added_node_output_names = node_output_names[num_model_outputs:]
        node_names = [added_node_output_names[i].rpartition('_')[0]
                      for i in range(0, len(added_node_output_names), 2)]  # output names

        # Characterizing distribution of a node's values across test data sets
        clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict
                                  if i != list(merged_dict.keys())[0])
        if calib_mode == 'naive':
            pairs = [
                tuple([
                    float(min(clean_merged_dict[added_node_output_names[i]])),
                    float(max(clean_merged_dict[added_node_output_names[i + 1]]))
                ]) for i in range(0, len(added_node_output_names), 2)
            ]
        else:
            raise ValueError('Unknown value for calib_mode. \
                             Currently only naive mode is supported.')

        final_dict = dict(zip(node_names, pairs))

        return final_dict

    def dump_calibration(self, calib_mode='naive'):
        '''
            Gather calibration params for quantization
            parameter calib_mode: type 'naive' gives (ReduceMin, ReduceMax) pairs
                                for each augmented node across test data sets, where
                                the first element is a minimum of all ReduceMin values
                                and the second element is a maximum of all ReduceMax
                                values;
            :return: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }
        '''

        self.augment_nodes = ["ReduceMin", "ReduceMax"]
        self.augment_graph()
        node_output_names, output_dicts_list = self.get_intermediate_outputs()
        mapped_dict = self._map_calibration(node_output_names, output_dicts_list,
                                            calib_mode=calib_mode)

        return self.calculate_quantization_params(mapped_dict)

    def calculate_quantization_params(self, quantization_thresholds):
        '''
            Given quantization thresholds, calculate the quantization params.
        :param quantization_thresholds:
            Dictionary specifying the min and max values for outputs of conv and matmul nodes.
            The quantization_thresholds should be specified in the following format:
                {
                    "param_name": [min, max]
                }
            example:
                {
                    'Conv_3:0': [np.float32(0), np.float32(0.5)],
                    'Conv_4:0': [np.float32(1), np.float32(3.5)]
                }
        :return: Dictionary containing the zero point and
                  scale values for outputs of conv and matmul nodes.
            The dictionary format is
                {
                    "param_name": [zero_point, scale]
                }
        '''
        if quantization_thresholds is None:
            raise ValueError(
                'quantization thresholds is required to calculate quantization \
                    params (zero point and scale)')

        quantization_params = {}
        model = self.model

        self._get_input_name_to_nodes(model)
        self._get_output_name_to_nodes(model)

        for tensor_name in quantization_thresholds.keys():
            child = None
            if tensor_name in self.input_name_to_nodes:
                children = self.input_name_to_nodes[tensor_name]
                if len(children) == 1:
                    child = children[0]
            parent = None
            if tensor_name in self.output_name_to_nodes:
                parents = self.output_name_to_nodes[tensor_name]
                if len(parents) == 1:
                    parent = parents[0]
            node_thresholds = quantization_thresholds[tensor_name]
            node_params = calculate_scale_zeropoint(parent, child, node_thresholds[0],
                                                         node_thresholds[1])
            quantization_params[tensor_name] = node_params

        return quantization_params

    def _get_input_name_to_nodes(self, model):
        '''
            Helper function to get input_name_to_nodes dictionary
        '''

        for node in model.graph.node:
            for input_name in node.input:
                if input_name not in self.input_name_to_nodes:
                    self.input_name_to_nodes[input_name] = [node]
                else:
                    self.input_name_to_nodes[input_name].append(node)

    def _get_output_name_to_nodes(self, model):
        '''
            Helper function to get output_name_to_nodes dictionary
        '''
        self.output_name_to_nodes = {}
        for node in model.graph.node:
            for output_name in node.output:
                if output_name not in self.output_name_to_nodes:
                    self.output_name_to_nodes[output_name] = [node]
                else:
                    self.output_name_to_nodes[output_name].append(node)

    def dump_tensor(self):
        self.augment_graph()
        return self.get_intermediate_outputs()[1]

def calculate_scale_zeropoint(last_node, next_node, rmin, rmax):
    '''
       Given the source and destination node of tensor, \
             return calculated zero point and scales.

      :param last_node: the source of the tensor
      :param next_node: the destination of the tensor
      :param rmin: min threshold of the tensor
      :param rmax: max threshold of the tensor
      :return (List): zero_point and scale

    '''

    zp_and_scale = []
    # adjust rmin and rmax such that 0 is included in the range. This is required
    # to make sure zero can be uniquely represented.
    rmin = min(rmin, 0)
    rmax = max(rmax, 0)
    if next_node:
        if next_node.op_type == 'Relu':
            if rmin < 0:
                rmin = 0

    if last_node:
        if last_node.op_type in ['Conv', 'FusedConv']:
            attrs = [attr for attr in last_node.attribute]
            attrs_names = [attr.name for attr in last_node.attribute]
            if 'activation' in attrs_names:
                if attrs[attrs_names.index('activation')].s == b'Relu':
                    rmin = max(rmin, 0)
                if attrs[attrs_names.index('activation')].s == b'Clip':
                    assert 'activation_params' in attrs_names, "the model contains no \
                                                               params for clip node \
                                                               {}".format(last_node)
                    clip_params = attrs[attrs_names.index('activation_params')].floats
                    rmin = min(rmin, clip_params[0], clip_params[1])
                    rmax = max(rmax, clip_params[0], clip_params[1])

    scale = np.float32((rmax - rmin) / 255 if rmin != rmax else 1)
    initial_zero_point = (0 - rmin) / scale
    zero_point = np.uint8(round(max(0, min(255, initial_zero_point))))

    zp_and_scale.append(zero_point)
    zp_and_scale.append(scale)

    return zp_and_scale
