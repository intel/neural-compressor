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
import sys

import numpy as np
import onnx
import onnxruntime
import onnx.numpy_helper as numpy_helper
from onnx import helper, TensorProto, shape_inference
from packaging.version import Version
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.adaptor.ox_utils.util import make_dquant_node, is_B_transposed

logger = logging.getLogger()
ONNX18_VERSION = Version("1.8.0")
ORT112_VERSION = Version("1.12.0")

class ONNXRTAugment:
    '''augment input model to dump tensor or for calibration'''

    def __init__(self, model_wrapper,
                 dataloader,
                 dump_op_types,
                 black_nodes=[],
                 white_nodes=[],
                 iterations=[]):
        '''
        :param model: ONNX model to calibrate
        :param dataloader: user implemented object to read in and preprocess calibration dataset
        :param op_types: operator types to be calibrated and quantized, default = 'Conv,MatMul'
        :param black_nodes: operator names that should not be quantized, default = ''
        :param white_nodes: operator names that force to be quantized, default = ''
        :param iterations: tensor of which iteration will be collected.
        '''
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        ai_onnx_domain = [opset for opset in self.model.opset_import \
            if not opset.domain or opset.domain == "ai.onnx"]
        self.opset_version = ai_onnx_domain[0].version
        self.dataloader = dataloader
        self.dump_op_types = dump_op_types
        self.black_nodes = black_nodes
        self.white_nodes = white_nodes
        self.augmented_model = None
        self.iterations = iterations
        self.augment_nodes = []
        self.dequantized_output = {}
        self.already_quantized = 'DequantizeLinear' in \
                                 [node.op_type for node in self.model.graph.node]
        self.dynamically_quantized = False
        self.ort_version = Version(onnxruntime.__version__)

    def augment_graph(self, activation_only=False, weight_only=False):
        '''
        Adds nodes to all quantization_candidates op type nodes in
        model and ensures their outputs are stored as part of the graph output
        :param activation_only(bool): whether to dump activation tensor only
        :param weight_only(bool): whether to dump weight_only
        :return: augmented ONNX model
        '''
        self.dequantized_output.clear()
        onnx_version = Version(onnx.__version__)
        if onnx_version < ONNX18_VERSION:
            logger.warning("Static quantization for NLP model is supported " \
                           "at onnx 1.8.0 and newer.")  
        if self.already_quantized and any([i.dims in [1, 2] for i in \
            self.model_wrapper.initializer() if i.name.endswith('_scale')]):
            if self.opset_version < 13 and self.ort_version >= ORT112_VERSION:
                logger.warning("Please use onnxruntime < 1.12.0 or upgrade model opset " \
                    "version to 13 or higher to inspect per-channel quantized weight") 
 
        model = copy.deepcopy(self.model)
        model_nodes_names = [node.name for node in model.graph.node]

        added_nodes = []
        added_outputs = []
        tensors_to_dump = set()

        for augment_node_type in self.augment_nodes:
            if augment_node_type not in ['DequantizeLinear']: # pragma: no cover
                raise ValueError("Unexpected augment_node {} only DequantizeLinear is " \
                                 "supported".format(augment_node_type))

        if self.already_quantized:
            # mapping between fp32 node and int8 node
            new_white_nodes = []
            for white_node in self.white_nodes:
                new_white_node = white_node + "_quant"
                assert new_white_node in model_nodes_names, "no quantized {} in the " \
                                                            "graph".format(white_node)
                new_white_nodes.append(new_white_node)
            self.white_nodes = new_white_nodes

        initializers = {i.name: i.data_type for i in model.graph.initializer}
        node_outputs = []
        for node in model.graph.node: # pylint: disable=no-member
            node_outputs.extend(node.output)
            should_be_dump = ((node.op_type in self.dump_op_types) and
                                   (node.name not in self.black_nodes)) or \
                                   (node.name in self.white_nodes)
            if should_be_dump:
                if not weight_only and not activation_only:
                    tensors_to_dump.update(node.input)
                    tensors_to_dump.update(node.output)
                elif weight_only:
                    for input in node.input:
                        if self.already_quantized and \
                            input.replace('_dequantized', '_quantized') in initializers:
                            tensors_to_dump.add(input)
                        elif not self.already_quantized and input in initializers:
                            tensors_to_dump.add(input)
                elif activation_only:
                    tensors_to_dump.update(node.output)

        for tensor in tensors_to_dump:
            if self.augment_nodes:
                if tensor not in node_outputs and tensor not in initializers:
                    continue
                for augment_node_type in self.augment_nodes:
                    if augment_node_type in ['DequantizeLinear']:
                        # insert DequantizeLinear node as output
                        if tensor.endswith('_scale') or tensor.endswith('_zero_point') or \
                            tensor.endswith('_QuantizeLinear') or \
                            tensor.endswith('_QuantizeInput_quantized'):
                            continue 

                        if not self.dynamically_quantized:
                            tensor = tensor.replace('_QuantizeInput', '_quantized') if \
                                tensor.endswith('_QuantizeInput') else tensor
                        else:
                            tensor = tensor.replace('_output_quantized', '') if \
                                tensor.endswith('_output_quantized') else tensor

                        augment_node_name = tensor + "_new_" + augment_node_type
                        scale, zero_point = self.model_wrapper.get_scale_zero(tensor)
                        if scale:
                            # the tensor is in INT8 dtype
                            nodes, output = self._dequantize(tensor, scale, zero_point)
                            if output:
                                added_nodes.extend(nodes)
                                added_outputs.append(helper.make_tensor_value_info(
                                                   output, # pylint: disable=no-member
                                                   TensorProto.FLOAT, ())) # pylint: disable=no-member
                        else:
                            # the tensor is in FP32 dtype
                            if tensor not in [t.name for t in model.graph.output]:
                                added_tensor = helper.ValueInfoProto()
                                added_tensor.name = tensor
                                added_outputs.append(added_tensor)
            else:
                if tensor not in [t.name for t in model.graph.output]:
                    added_tensor = helper.ValueInfoProto()
                    added_tensor.name = tensor
                    added_outputs.append(added_tensor)

        if self.augment_nodes:
            model.graph.node.extend(added_nodes) # pylint: disable=no-member
        model.graph.output.extend(added_outputs) # pylint: disable=no-member

        self.augmented_model = model

    def get_intermediate_outputs(self, calib_mode=None):
        '''
            Gather intermediate model outputs after running inference
            :return: dictionary mapping: {node output tensor names: node output tensor }
        '''

        # conduct inference session and get intermediate outputs
        so = onnxruntime.SessionOptions()
        if sys.version_info < (3,10): # pragma: no cover
            from onnxruntime_extensions import get_library_path
            so.register_custom_ops_library(get_library_path())

        session = onnxruntime.InferenceSession(self.augmented_model.SerializeToString(), so)

        intermediate_outputs = []
        len_inputs = len(session.get_inputs())
        inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]
        output_dicts = {}
        
        node_output_names = [output.name if output.name not in self.dequantized_output \
                             else self.dequantized_output[output.name] \
                             for output in session.get_outputs()]

        for idx, (inputs, labels) in enumerate(self.dataloader):
            ort_inputs = {}
            if len_inputs == 1:
                ort_inputs.update(
                    inputs if isinstance(inputs, dict) else {inputs_names[0]: inputs}
                )
            else:
                assert len_inputs == len(inputs), \
                    'number of input tensors must align with graph inputs'
                if isinstance(inputs, dict):  # pragma: no cover
                    ort_inputs.update(inputs)
                else:
                    for i in range(len_inputs):
                        if not isinstance(inputs[i], np.ndarray): # pragma: no cover
                            ort_inputs.update({inputs_names[i]: np.array(inputs[i])})
                        else:
                            ort_inputs.update({inputs_names[i]: inputs[i]})
            if self.iterations != []:
                if idx > max(self.iterations):
                    break
                if idx in self.iterations:
                    for output_idx, output in enumerate(session.run(None, ort_inputs)): 
                        if calib_mode == 'naive' and output.size != 0:
                            output_dicts.setdefault(node_output_names[output_idx], \
                                []).append([output.min(), output.max()])
                        elif calib_mode == None:
                            output_dicts.setdefault(node_output_names[output_idx], \
                                []).append(output)
            else:
                for output_idx, output in enumerate(session.run(None, ort_inputs)): 
                    if calib_mode == 'naive' and output.size != 0:
                        output_dicts.setdefault(node_output_names[output_idx], \
                            []).append([output.min(), output.max()])
                    elif calib_mode == None:
                        output_dicts.setdefault(node_output_names[output_idx], \
                            []).append(output)

        return list(output_dicts.keys()), output_dicts

    def _dequantize(self, tensor, scale_tensor, zo_tensor):
        ''' helper function to dequantize tensor
        '''
        int_tensor = self.model_wrapper.get_initializer(tensor)
        if int_tensor: # weight tensor
            return self._dequantize_weight(tensor, scale_tensor, zo_tensor)
        else:
            return self._dequantize_activation(tensor, scale_tensor, zo_tensor)

    def _dequantize_activation(self, activation_tensor_name, scale_tensor, zo_tensor):
        ''' helper funtion to dequantize activation'''
        added_nodes, added_output = self._add_dequantize_node(activation_tensor_name, \
                                                              scale_tensor, zo_tensor)
        self.dequantized_output[added_output] = activation_tensor_name
        return added_nodes, added_output

    def _dequantize_weight(self, weight_tensor_name, scale_tensor, zo_tensor):
        ''' helper function to dequantize weight'''
        weight_tensor = self.model_wrapper.get_initializer(weight_tensor_name)
        if len(scale_tensor.dims) in [1, 2] and weight_tensor.dims[0] == max(scale_tensor.dims):
            logger.debug("weight {} is quantized with per channel granularity."
                         .format(weight_tensor_name))
            if self.opset_version < 13 and self.ort_version >= ORT112_VERSION:
                logger.warning("Skip dequantizing weight {}, please use onnxruntime < 1.12.0 " \
                    "or upgrade model opset version to 13 or higher".format(weight_tensor_name))
                return [], None
            node = self.model_wrapper.input_name_to_nodes[weight_tensor_name][0]
            if 'Conv' in node.op_type or \
                ('Gemm' in node.op_type and is_B_transposed(node)):
                added_nodes, added_output = self._add_dequantize_transpose_node(
                                                                 weight_tensor_name,
                                                                 scale_tensor, zo_tensor,
                                                                 len(weight_tensor.dims))
            else:
                added_nodes, added_output = self._add_dequantize_node(
                    weight_tensor_name, 
                    scale_tensor,
                    zo_tensor,
                    axis=1 if self.opset_version > 12 else None)
        else:
            added_nodes, added_output = self._add_dequantize_node(weight_tensor_name, 
                                                                 scale_tensor,\
                                                                 zo_tensor)
        self.dequantized_output[added_output] = weight_tensor_name
        return added_nodes, added_output

    def _add_dequantize_node(self, tensor_name, scale_tensor, zo_tensor, axis=None):
        '''helper function to generate dequantize node'''
        dequantize_node = make_dquant_node(tensor_name + '_DequantizeLinear',
                                           [tensor_name,
                                           scale_tensor.name,
                                           zo_tensor.name],
                                           [tensor_name + '_output'],
                                           axis)
        return [dequantize_node], tensor_name + '_output'

    def _add_dequantize_transpose_node(self, tensor_name, scale_tensor, zo_tensor, dim):
        pre_transpose_node = onnx.helper.make_node(
                                 'Transpose',
                                 inputs=[tensor_name],
                                 outputs=[tensor_name + '_transposed'],
                                 perm=(1,0,2,3) if dim == 4 else (1,0),
                                 name=tensor_name + '_pre_transpose')
        dequantize_node = make_dquant_node(
                                 tensor_name + '_DequantizeLinear',
                                 [tensor_name + '_transposed', 
                                  scale_tensor.name, 
                                  zo_tensor.name], 
                                 [tensor_name + '_DequantizeLinear'], 
                                 axis=1 if self.opset_version > 12 else None)
        post_transpose_node = onnx.helper.make_node(
                                 'Transpose',
                                 inputs=[tensor_name + '_DequantizeLinear'],
                                 outputs=[tensor_name + '_output'],
                                 perm=(1,0,2,3) if dim == 4 else (1,0),
                                 name=tensor_name + '_post_transpose')
        added_nodes = [pre_transpose_node, dequantize_node, post_transpose_node]
        return added_nodes, tensor_name + '_output'

    def _map_calibration(self, node_output_names, output_dicts, calib_mode='naive'):
        merged_dict = {}
        for name, minmaxs in output_dicts.items():
            for minmax in minmaxs:
                merged_dict.setdefault(name + '_Min', []).append(minmax[0])
                merged_dict.setdefault(name + '_Max', []).append(minmax[1])

        # Characterizing distribution of a node's values across test data sets
        clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict)
        if calib_mode == 'naive':
            pairs = [
                tuple([
                    float(min(clean_merged_dict[name + '_Min'])),
                    float(max(clean_merged_dict[name + '_Max']))
                ]) for name in node_output_names
            ]
        else:
            raise ValueError('Unknown value for calib_mode. \
                             Currently only naive mode is supported.')

        final_dict = dict(zip(node_output_names, pairs))

        return final_dict

    def dump_minmax(self, calib_mode='naive'):
        self.augment_graph()
        node_output_names, output_dicts = self.get_intermediate_outputs(calib_mode)
        return self._map_calibration(node_output_names, output_dicts,
                                     calib_mode=calib_mode)

    def dump_calibration(self, calib_mode='naive'):
        '''
            Gather calibration params for quantization
            parameter calib_mode: type 'naive' gives (Min, Max) pairs
                                for each intermediate model output across 
                                test data sets, where the first element is 
                                a minimum of all values and the 
                                second element is a maximum of all values;
            :return: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }
        '''
        return self.calculate_quantization_params(self.dump_minmax(calib_mode))

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

        input_name_to_nodes = self.model_wrapper.input_name_to_nodes
        output_name_to_nodes = self.model_wrapper.output_name_to_node

        for tensor_name in quantization_thresholds.keys():
            child = None
            if tensor_name in input_name_to_nodes:
                children = input_name_to_nodes[tensor_name]
                if len(children) == 1:
                    child = children[0]
            parent = None
            if tensor_name in output_name_to_nodes:
                parent = output_name_to_nodes[tensor_name]
            node_thresholds = quantization_thresholds[tensor_name]
            node_params = self.calculate_scale_zeropoint(parent, child, node_thresholds[0],
                                                         node_thresholds[1])
            quantization_params[tensor_name] = node_params

        return quantization_params

    def dump_tensor(self, activation=True, weight=False):
        if "QuantizeLinear" in [node.op_type for node in self.model.graph.node] or \
            "DynamicQuantizeLinear" in [node.op_type for node in self.model.graph.node]:
            self.augment_nodes = ["DequantizeLinear"]
            self.already_quantized = True
            self.dynamically_quantized = \
                "DynamicQuantizeLinear" in [node.op_type for node in self.model.graph.node]
        self.augment_graph(activation_only=not weight, weight_only=not activation)
        _, output_dicts = self.get_intermediate_outputs()
        iters = len(list(output_dicts.values())[-1])
        map_node_activation = [{} for _ in range(iters)]
        map_node_weight = {}
        self.white_nodes = [node.replace('_quant', '') for node in self.white_nodes]
        augmengted_wrapper = ONNXModel(self.augmented_model)
        map_output = augmengted_wrapper.output_name_to_node
        map_input = augmengted_wrapper.input_name_to_nodes
        model_output_names = [t.name for t in self.model.graph.output]
        model_input_names = [t.name for t in self.model.graph.input]
        model_initializer_names = [t.name for t in self.model.graph.initializer]
        for tensor_name, tensors in output_dicts.items():
            if tensor_name.replace('_dequantized', '_quantized') in model_initializer_names:
                nodes = [node for node in map_input[tensor_name] \
                    if node.name.replace('_quant', '') in self.white_nodes]
            elif tensor_name.replace('_quantized', '') in model_input_names:
                continue
            else:
                nodes = [map_output[tensor_name]]
            for node in nodes:
                node_name = node.name.replace('_quant', '')
                if tensor_name in model_output_names and node_name not in self.white_nodes:
                    continue
                while node_name not in self.white_nodes and self.already_quantized:
                    node = augmengted_wrapper.get_parents(node, output_name_to_node=map_output)[0]
                    node_name = node.name.replace('_quant', '')
                if node_name not in self.white_nodes:
                    continue
                if node_name not in map_node_weight:
                    map_node_weight[node_name] = {}
                if tensor_name not in model_initializer_names:
                    for i in range(iters):
                        map_node_activation[i][node_name] = \
                                               {tensor_name.replace('_quantized', ''): tensors[i]}
                else:
                    map_node_weight[node_name].update({tensor_name.replace('_quantized', ''): \
                                                                                      tensors[0]})
        dumped_tensors_map = {}
        if weight:
            dumped_tensors_map.update({"weight": map_node_weight})
        if activation:
            dumped_tensors_map.update({"activation": map_node_activation})
        return dumped_tensors_map

    def calculate_scale_zeropoint(self, last_node, next_node, rmin, rmax):
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
            elif next_node.op_type == 'Clip' and len(next_node.input) ==3:
                if rmin < numpy_helper.to_array(
                        self.model_wrapper.get_initializer(next_node.input[1])):
                    rmin = numpy_helper.to_array(
                        self.model_wrapper.get_initializer(next_node.input[1]))
                if rmax > numpy_helper.to_array(
                        self.model_wrapper.get_initializer(next_node.input[2])):
                    rmax = numpy_helper.to_array(
                        self.model_wrapper.get_initializer(next_node.input[2]))
    
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
