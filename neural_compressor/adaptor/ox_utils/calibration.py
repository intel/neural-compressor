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
"""Calibration for onnx models."""

import copy
import logging
import sys

import numpy as np
import onnx
import onnxruntime
import onnx.numpy_helper as numpy_helper
from onnx import helper, TensorProto, shape_inference
from packaging.version import Version
from importlib.util import find_spec
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.adaptor.ox_utils.util import make_dquant_node, is_B_transposed, \
    _get_qrange_for_qType, calculate_scale_zp
import itertools

logger = logging.getLogger("neural_compressor")
ONNX18_VERSION = Version("1.8.0")
ORT112_VERSION = Version("1.12.0")


class ONNXRTAugment:
    """augment input model to dump tensor or for calibration."""

    def __init__(self, model_wrapper,
                 dataloader,
                 dump_op_types,
                 black_nodes=[],
                 white_nodes=[],
                 iterations=[],
                 backend=['CPUExecutionProvider'],
                 reduce_range=False):
        """Initialization.

        Args:
            model_wrapper (Model): model to be augmented
            dataloader (object): user implemented object to read in and preprocess calibration dataset
            dump_op_types (list): operator types to be calibrated and quantized
            black_nodes (list, optional): operator names that should not be quantized. Defaults to [].
            white_nodes (list, optional): operator names that force to be quantized. Defaults to [].
            iterations (list, optional): tensor of which iteration will be collected. Defaults to [].
            backend (list, optional): execution provider for onnxruntime. Defaults to ['CPUExecutionProvider'].
            reduce_range (bool, optional): use 7 bit or not. Defaults to False.
        """
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
        self.backend = backend
        self.augment_nodes = []
        self.dequantized_output = {}
        self.already_quantized = 'DequantizeLinear' in \
                                 [node.op_type for node in self.model.graph.node]
        self.dynamically_quantized = False
        self.ort_version = Version(onnxruntime.__version__)
        self.reduce_range = reduce_range

    def augment_graph(self, activation_only=False, weight_only=False):
        """Augment_graph.
        
        Adds nodes to all quantization_candidates op type nodes in model and
        ensures their outputs are stored as part of the graph output.

        Args:
            activation_only (bool, optional): whether to dump activation tensor only. Defaults to False.
            weight_only (bool, optional): whether to dump weight_only. Defaults to False.
        """
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
            if augment_node_type not in ['DequantizeLinear']:  # pragma: no cover
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
        for node in model.graph.node:  # pylint: disable=no-member
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

        model_inputs = [i.name for i in model.graph.input]
        for tensor in tensors_to_dump:
            if tensor not in node_outputs and tensor not in initializers and \
                    tensor not in model_inputs:
                continue
            if self.augment_nodes:
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
                                    output,  # pylint: disable=no-member
                                    TensorProto.FLOAT, ()))  # pylint: disable=no-member
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
            model.graph.node.extend(added_nodes)  # pylint: disable=no-member
        model.graph.output.extend(added_outputs)  # pylint: disable=no-member

        self.augmented_model = model
        if self.model_wrapper.large_size:  # pragma: no cover
            onnx.save_model(model,
                            self.model_wrapper.model_path + '_augment.onnx',
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            location="weights.pb",
                            convert_attribute=False)

    def get_intermediate_outputs(self, calib_mode=None, model=None, model_path=None):
        """Gather intermediate model outputs after running inference."""
        # conduct inference session and get intermediate outputs
        so = onnxruntime.SessionOptions()
        if sys.version_info < (3, 10) and find_spec('onnxruntime_extensions'):  # pragma: no cover
            from onnxruntime_extensions import get_library_path
            so.register_custom_ops_library(get_library_path())
        if model == None:
            model = self.augmented_model
        if model_path == None:
            model_path = self.model_wrapper.model_path + '_augment.onnx'
        if self.model_wrapper.large_size:
            session = onnxruntime.InferenceSession(
                model_path,
                so,
                provider=self.backend)
        else:
            session = onnxruntime.InferenceSession(
                model.SerializeToString(),
                so,
                provider=self.backend)

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
                        if not isinstance(inputs[i], np.ndarray):  # pragma: no cover
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
        """Helper function to dequantize tensor."""
        int_tensor = self.model_wrapper.get_initializer(tensor)
        if int_tensor:  # weight tensor
            return self._dequantize_weight(tensor, scale_tensor, zo_tensor)
        else:
            return self._dequantize_activation(tensor, scale_tensor, zo_tensor)

    def _dequantize_activation(self, activation_tensor_name, scale_tensor, zo_tensor):
        """Helper funtion to dequantize activation."""
        added_nodes, added_output = self._add_dequantize_node(activation_tensor_name, \
                                                              scale_tensor, zo_tensor)
        self.dequantized_output[added_output] = activation_tensor_name
        return added_nodes, added_output

    def _dequantize_weight(self, weight_tensor_name, scale_tensor, zo_tensor):
        """Helper function to dequantize weight."""
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
                                                                  scale_tensor, \
                                                                  zo_tensor)
        self.dequantized_output[added_output] = weight_tensor_name
        return added_nodes, added_output

    def _add_dequantize_node(self, tensor_name, scale_tensor, zo_tensor, axis=None):
        """Helper function to generate dequantize node."""
        dequantize_node = make_dquant_node(tensor_name + '_DequantizeLinear',
                                           [tensor_name,
                                            scale_tensor.name,
                                            zo_tensor.name],
                                           [tensor_name + '_output'],
                                           axis)
        return [dequantize_node], tensor_name + '_output'

    def _add_dequantize_transpose_node(self, tensor_name, scale_tensor, zo_tensor, dim):
        """Insert Transpose-DequantizelLinear-Transpose pairs."""
        pre_transpose_node = onnx.helper.make_node(
            'Transpose',
            inputs=[tensor_name],
            outputs=[tensor_name + '_transposed'],
            perm=(1, 0, 2, 3) if dim == 4 else (1, 0),
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
            perm=(1, 0, 2, 3) if dim == 4 else (1, 0),
            name=tensor_name + '_post_transpose')
        added_nodes = [pre_transpose_node, dequantize_node, post_transpose_node]
        return added_nodes, tensor_name + '_output'

    def _map_calibration(self, node_output_names, output_dicts, calib_mode='naive'):
        """Map tensor names and min/max values."""
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
        """Get min/max values of tensors."""
        self.augment_graph()
        node_output_names, output_dicts = self.get_intermediate_outputs(calib_mode)
        return self._map_calibration(node_output_names, output_dicts,
                                     calib_mode=calib_mode)

    def dump_calibration(self, q_config, calib_mode='naive'):
        """Gather calibration params for quantization.

        Args:
            q_config (dict): op-wise quantization config
            calib_mode (str, optional): type 'naive' gives (Min, Max) pairs
                                        for each intermediate model output across
                                        test data sets, where the first element is
                                        a minimum of all values and the second element 
                                        is a maximum of all values. Defaults to 'naive'.
        """
        return self.calculate_quantization_params(q_config, self.dump_minmax(calib_mode))

    def calculate_quantization_params(self, q_config, quantization_thresholds):
        """Given quantization thresholds, calculate the quantization params.

        Args:
            q_config (dict): op-wise quantization config
            quantization_thresholds (dict): Dictionary specifying the min and max values
                                              or outputs of conv and matmul nodes, should be
                                              specified in the following format:
                                              {"param_name": [min, max]}
        """
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
            scheme = 'asym'
            qType = 2  # uint8
            if tensor_name in output_name_to_nodes:
                parent = output_name_to_nodes[tensor_name]
            if parent and parent.name in q_config and q_config[parent.name] not in ['fp32']:
                scheme = q_config[parent.name]['activation']['scheme']
                qType = q_config[parent.name]['activation']['dtype']
            elif self.backend in ['TensorrtExecutionProvider']:
                scheme = 'sym'
                qType = 3
            node_thresholds = quantization_thresholds[tensor_name]
            node_params = self.calculate_scale_zeropoint(parent, child, node_thresholds[0],
                                                         node_thresholds[1], scheme, qType,
                                                         _get_qrange_for_qType(qType, self.reduce_range))
            quantization_params[tensor_name] = node_params

        return quantization_params

    def dump_tensor(self, activation=True, weight=False):
        """Dump activation or weight or both from the model."""
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

    def calculate_scale_zeropoint(self, last_node, next_node, rmin, rmax, scheme, qType, quantize_range):
        """Given the source and destination node of tensor, return calculated zero point and scales."""
        zp_and_scale = []
        # adjust rmin and rmax such that 0 is included in the range. This is required
        # to make sure zero can be uniquely represented.
        rmin = min(rmin, 0)
        rmax = max(rmax, 0)
        if next_node:
            if next_node.op_type == 'Relu':
                if rmin < 0:
                    rmin = 0
            elif next_node.op_type == 'Clip' and len(next_node.input) == 3:
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

        scale, zp = calculate_scale_zp(rmin, rmax, quantize_range, qType, scheme)
        if qType == 2:
            zp_and_scale.append(np.uint8(zp))
        else:
            zp_and_scale.append(np.int8(zp))
        zp_and_scale.append(np.float32(scale))

        return zp_and_scale

    def _create_initializer_tensor(self,
                                   name: str,
                                   tensor_array: np.ndarray,
                                   data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
                                   ) -> onnx.TensorProto:
        """
        A help function to create initializer tensor in onnx graph
        Args:
            name: The name of tensor to be created
            tensor_array: The numpy array to initialize the tensor
            data_type: The data type

        Returns:
            An onnx tensor

        """

        # (TensorProto)
        initializer_tensor = onnx.helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=tensor_array.shape,
            vals=tensor_array.flatten().tolist())

        return initializer_tensor

    def _save_large_onnx_model(self, model, model_path, weights_name):
        f"""
        A help function to save onnx model > 2GB, for large onnx model,
        the graph and weights will be saved in different files
        Args:
            model: The onnx model
            model_path: the path to be saved
            weights_name: the file name of weights to be saved in the {model_path} 

        Returns:

        """
        onnx.save_model(model,
                        model_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=weights_name,
                        convert_attribute=False)

    def _check_is_group_conv(self, node, model):
        """
        Check the op is group wised or not(depthwise conv is excluded,return false)
        Args:
            node: The op node
            model: The onnx model

        Returns:
            Bool: group wised True, otherwise False, depthwise False

        """

        name_to_indices = {}
        for index, i in enumerate(model.graph.initializer):
            name_to_indices[i.name] = index

        if node.op_type == "Conv":
            group = 1
            for attr in node.attribute:
                if hasattr(attr, 'name'):
                    if attr.name == "group":
                        group = attr.i
                        break
            ##currently only normal conv and depthwise conv are supported
            if group > 1:  ##group conv, need to check depthwise or not\
                weight_name = node.input[1]
                weight_shape = numpy_helper.to_array(
                    model.graph.initializer[name_to_indices[weight_name]]).shape
                input_channel = weight_shape.shape[1]
                if input_channel != 1:  ##TODO need to double check
                    return True
        return False

    def _get_input_tensor_of_ops(self, op_types=['MatMul', 'Linear', 'Conv']):
        f"""
        Traverse the graph and get all the data tensors flowing into layers of {op_types}, group conv is excluded
        TODO the tensors could be set/filtered in configuration

        Args:
            op_types: The op types whose input tensor will be dumped

        Returns:
            A set of tensor names 

        """

        tensors_to_dump = set()
        ##model = copy.deepcopy(self.model)
        model = self.model
        initializers = {i.name: i for i in model.graph.initializer}

        for node in model.graph.node:
            if len(op_types) == 0 or node.op_type in op_types:
                if node.op_type == "Conv" and self._check_is_group_conv(node, model):
                    continue
                ##also need to check whether the layer has weight
                if len(node.input) >= 2 and node.input[1] in initializers.keys():
                    tensors_to_dump.add(node.input[0])
        return tensors_to_dump

    def _add_tensors_to_outputs(self, tensor_names, model):
        """
        Add the tensors to the model outputs to gets their values
        Args:
            tensor_names: The names of tensors to be dumped
            model: The onnx model

        Returns: None

        """
        ##model_inputs = [i.name for i in model.graph.input]
        added_outputs = []
        for tensor in tensor_names:
            if tensor not in [t.name for t in model.graph.output]:
                added_tensor = helper.ValueInfoProto()
                added_tensor.name = tensor
                added_outputs.append(added_tensor)
        model.graph.output.extend(added_outputs)  # pylint: disable=no-member

    def _is_large_model(self):
        """
        Check the onnx model is over 2GB
        Returns:
            Bool
        """
        return self.model_wrapper.large_size

    def _get_max_per_channel(self, datas: list, percentile):
        """
        Get the max values per input channel
        Args:
            datas: The tensors
            percentile: percentile of calibration to remove outliers

        Returns:
            The max values per input channel

        """
        permute_datas = []
        for data in datas:
            if len(data.shape) == 3:  ##TODO  mammul batchsize*seq*inchannel, conv:batchsize*inchannle*f*f
                tensor = np.abs(np.reshape(data, (-1, data.shape[-1])))
                permute_datas.append(tensor)
            elif len(data.shape) == 4:
                tensor = np.swapaxes(data, 1, -1)
                tensor = np.abs(np.reshape(tensor, (-1, tensor.shape[-1])))
                permute_datas.append(tensor)
            elif len(data.shape) == 2:
                permute_datas.append(np.abs(data))
            else:
                assert False, "not supported"
        permute_datas = np.stack(permute_datas, axis=0)
        permute_datas = permute_datas.reshape(-1, permute_datas.shape[-1])
        max_per_channels = np.percentile(permute_datas, percentile, axis=0)
        max_per_channels = max_per_channels.astype(np.single)
        return max_per_channels

    def _augment_smooth_calib_graph(self, op_types=['MatMul', 'Linear', 'Conv']):
        """
        add the input tensors of {op_types} to outputs of the model

        Args:
            op_types:The op types whose input tensor will be dumped

        Returns:
            tensors_to_dump: The names of tensor to be dumped
            model: the modified onnx model
            large_model_path: the saved path for large onnx model
        """
        model = copy.deepcopy(self.model)

        tensors_to_dump = self._get_input_tensor_of_ops(op_types)
        self._add_tensors_to_outputs(tensors_to_dump, model)

        self.smooth_calib_augmented_model = model
        large_model_path = None
        if self._is_large_model():  # pragma: no cover
            large_model_path = self.model_wrapper.model_path + '_smooth_calib_augment.onnx'
            self._save_large_onnx_model(model, large_model_path, "smooth_weights.pb")

        return tensors_to_dump, model, large_model_path

    def _adjust_weights(self, model, nodes, scales):
        """
        Adjust the wights per input scale
        Args:
            model: The onnx model
            nodes: The nodes whose weights needs to be adjustd
            scales: The input scales

        Returns:

        """
        name_to_indices = {}
        for index, i in enumerate(model.model.graph.initializer):
            name_to_indices[i.name] = index

        for key in nodes.keys():
            curr_nodes = nodes[key]
            for node in curr_nodes:
                input = node.input[1]  ##TODO
                if input in name_to_indices.keys():
                    weight = numpy_helper.to_array(model.model.graph.initializer[name_to_indices[input]])
                    if len(weight.shape) == 2:
                        scale = np.expand_dims(scales[key],
                                               axis=-1)  ##TODO, to support conv
                        new_weight = weight * scale
                    elif len(weight.shape) == 4:  ##TODO need to check conv
                        scale = np.reshape(scales[key], (1, -1, 1, 1))
                        new_weight = weight * scale
                    else:
                        assert False, "not support"
                    new_tensor = numpy_helper.from_array(new_weight, input)
                    model.model.graph.initializer[name_to_indices[input]].CopyFrom(new_tensor)

    def _calib_smooth(self, percentile, op_types):
        """
        smooth model calibration, mainly get the max info per channel of input tensors
        Args:
            percentile:Percentile of calibration to remove outliers
            op_types: The op types whose input tensor will be dumped

        Returns:
            max_vals_per_channel: max values per channel of input tensors
            shape_infos: The shape information of input tensors

        """
        tensors_to_dump, model, large_model_path = self._augment_smooth_calib_graph(op_types)
        _, output_dicts = self.get_intermediate_outputs(None, model, large_model_path)
        max_vals_per_channel = {}
        shape_infos = {}
        for key in tensors_to_dump:
            max_val_per_channel = self._get_max_per_channel(output_dicts[key], percentile=percentile)
            max_vals_per_channel[key] = max_val_per_channel
            shape_infos[key] = output_dicts[key][0].shape
        return max_vals_per_channel, shape_infos

    def _input_tensor_2_weights(self, model, tensor_names, op_types):
        """
        get the corresponding weights needs to be adjusted later
        Args:
            model: The onnx model
            tensor_names: The input tensor names
            op_types:The op types whose input tensor will be dumped

        Returns:
            input_tensors_2_weights: A dict, key is the input tensor name, value is the corresponding weights
            input_tensors_2_weights_nodes: A dict, key is the input tensor name , value is the corresponding nodes with weight

        """
        tensor_to_absorbed_nodes = {}
        for key in tensor_names:
            tensor_to_absorbed_nodes[key] = []
        for node in model.model.graph.node:
            for item in node.input:
                if item in tensor_names:
                    tensor_to_absorbed_nodes[item].append(node)

        input_tensors_2_weights = {}
        input_tensors_2_weights_nodes = {}
        weight_name_to_init_index = {}
        for index, i in enumerate(model.model.graph.initializer):
            weight_name_to_init_index[i.name] = index
        for key in tensor_to_absorbed_nodes.keys():
            curr_tensor_to_weight = []
            curr_tensor_to_weight_nodes = []
            nodes = tensor_to_absorbed_nodes[key]

            for node in nodes:
                if node.op_type not in op_types:
                    continue
                if len(node.input) >= 2:
                    input = node.input[1]  ##TODO always dump the index 1 to get the weight
                    if input in weight_name_to_init_index.keys():
                        weight = numpy_helper.to_array(model.model.graph.initializer[weight_name_to_init_index[input]])
                        curr_tensor_to_weight.append(weight)
                        curr_tensor_to_weight_nodes.append(node)

            input_tensors_2_weights[key] = curr_tensor_to_weight
            input_tensors_2_weights_nodes[key] = curr_tensor_to_weight_nodes
        return input_tensors_2_weights, input_tensors_2_weights_nodes

    def _get_smooth_scales(self, max_vals_per_channel, input_tensors_2_weights, alpha):
        """
        Get the smooth scales for weights
        TODO support individual scales for each layer
        Args:
            max_vals_per_channel: Max values per channel after calibration
            input_tensors_2_weights: A dict saved input tensor name and its corresponding weights
            alpha: smooth alpha in paper

        Returns:
            the smooth scales for weights, currently one input tensor only have one scale

        """
        scales = {}
        for key in input_tensors_2_weights.keys():
            weights = input_tensors_2_weights[key]
            weights_in_channel_max = []
            for weight in weights:  ##mamul ic*oc, conv oc*ic*k*k
                if len(weight.shape) == 4:  ##conv
                    if weight.shape[1] == 1:  ##depthwise conv
                        pass
                    else:
                        weight = np.moveaxis(weight, 0, 1)

                weight = weight.reshape(weight.shape[0], -1)
                cur_max = np.amax(weight, axis=-1)
                weights_in_channel_max.append(cur_max)

            weigths_stack = np.stack(weights_in_channel_max, axis=-1)
            weigths_stack = np.abs(weigths_stack.reshape(weigths_stack.shape[0], -1))
            weights_max = np.amax(weigths_stack, axis=-1)
            input_power = np.power(max_vals_per_channel[key], alpha)
            weight_power = np.power(weights_max, 1 - alpha)
            scale = np.clip(input_power / weight_power, a_min=1e-5, a_max=None)
            scales[key] = scale
        return scales

    def _insert_smooth_mul_op(self, scales, shape_infos, input_tensors_2_weights_nodes):
        """
        Insert the mul layer before each op
        Args:
            scales: The smooth scales
            shape_infos: the input tensor shape information
            input_tensors_2_weights_nodes:  A dict

        Returns:
            new_added_mul_nodes: added Mul layers
            new_init_tensors: added scales tensor

        """
        new_added_mul_nodes = []
        new_init_tensors = []  ##scales_tensor
        for key in scales.keys():
            scale_factor = 1.0 / scales[key]
            shape_info = shape_infos[key]
            if len(shape_info) == 3 or len(shape_info) == 2:  ##the last dim is input channel
                pass
            elif len(shape_info) == 4:
                scale_factor = np.reshape(scale_factor, (1, -1, 1, 1))
            else:
                assert False, "not support"
            name = key + "_" + "smooth_scale"
            scale_tensor = self._create_initializer_tensor(
                name, scale_factor
            )
            new_init_tensors.append(scale_tensor)
            mul_output_name = key + "_smooth_output"
            mul_node = onnx.helper.make_node(
                "Mul",
                inputs=[key, key + "_" + "smooth_scale"],
                outputs=[mul_output_name],
                name=key + "_smooth_mul"
            )
            new_added_mul_nodes.append(mul_node)

            for node in input_tensors_2_weights_nodes[key]:
                for index, input in enumerate(node.input):
                    if input == key:
                        node.input[index] = mul_output_name
        return new_added_mul_nodes, new_init_tensors

    def augment_smooth_graph(self, alpha=1.0, percentile=99.999, op_types=['MatMul', 'Linear', 'Conv']):
        """
        fake input channel quantization, for more details please refer to
        [1] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
        [2] SPIQ: Data-Free Per-Channel Static Input Quantization
        inert Mul op before each conv/matmul with adjusted weights

        Args:
            alpha: smooth alpha in SmoothQuant, 1.0 will fallback to SPIQ
            percentile:Percentile of calibration to remove outliers
            op_types: The op types whose input tensor will be dumped

        Returns:
            model: A modified onnx model

        """

        max_vals_per_channel, shape_infos = self._calib_smooth(percentile, op_types)

        model = copy.deepcopy(self.model_wrapper)
        tensor_names = [key for key in max_vals_per_channel.keys()]
        input_tensors_2_weights, input_tensors_2_weights_nodes = self._input_tensor_2_weights(model, tensor_names,
                                                                                              op_types)
        scales = self._get_smooth_scales(max_vals_per_channel, input_tensors_2_weights, alpha)
        new_added_mul_nodes, new_init_tensors = self._insert_smooth_mul_op(scales, shape_infos,
                                                                           input_tensors_2_weights_nodes)
        self._adjust_weights(model, input_tensors_2_weights_nodes, scales)

        model.model.graph.node.extend(new_added_mul_nodes)
        model.model.graph.initializer.extend(new_init_tensors)
        model.update()
        model.topological_sort()
        model.remove_unused_constant()
        return model
