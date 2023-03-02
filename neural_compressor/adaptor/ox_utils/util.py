#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2021 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Helper classes or functions for onnxrt adaptor."""

import os
import numpy as np
from neural_compressor.utils.utility import LazyImport
from enum import Enum
from pathlib import Path
import abc

helper = LazyImport('onnx.helper')
numpy_helper = LazyImport('onnx.numpy_helper')
onnx_proto = LazyImport('onnx.onnx_pb')

__producer__ = "onnx.quantize"
__version__ = "0.1.0"
onnx_domain = "ai.onnx"
ms_domain = "com.microsoft"      

support_pair = {
    'float32 bfloat16': True,
    '1 16': True,
    'bfloat16 float32': True,
    '16 1': True,
    'uint8 uint8': True,
    '2 2': True,
    'float16 float16': True,
    '10 10': True,
    'bfloat16 bfloat16': True,
    '16 16': True,
    'float32 float16': True,
    '1 10': True,
    'float16 float32': True,
    '10 1': True
}

dtype_mapping = {
    'fp32': 1,
    'uint8': 2,
    'int8': 3,
    'uint16': 4,
    'int16': 5,
    'int32': 6,
    'int64': 7,
    'string': 8,
    'bool': 9,
    'fp16': 10,
    'double': 11,
    'uint32': 12,
    'uint64': 13,
    'complex64': 14,
    'complex128': 15,
    'bf16': 16
}

PROVIDERS = {
    'default': 'CPUExecutionProvider',
    'onnxrt_trt_ep': 'TensorrtExecutionProvider',
    'onnxrt_cuda_ep': 'CUDAExecutionProvider',
}

ONNXRT_BACKENDS = {
    'CPUExecutionProvider': 'default',
    'TensorrtExecutionProvider': 'onnxrt_trt_ep',
    'CUDAExecutionProvider': 'onnxrt_cuda_ep'
}

def dtype_to_name(dtype_mapping, dtype):
    """Map data type and its string representation."""
    return list(dtype_mapping.keys())[list(dtype_mapping.values()).index(dtype)]

class QuantType(Enum): # pragma: no cover
    """Represent QuantType value."""

    QInt8 = 0
    QUInt8 = 1

def make_quant_node(name, inputs, outputs):
    """Make a QuantizeLinear node."""
    return helper.make_node("QuantizeLinear", inputs, outputs, name)

def make_dquant_node(name, inputs, outputs, axis=None):
    """Make a DequantizeLinear node."""
    if axis is not None:
        return helper.make_node("DequantizeLinear", inputs, outputs, name, axis=axis)
    else:
        return helper.make_node("DequantizeLinear", inputs, outputs, name)

def is_B_transposed(node):
    """Wheter inuput B is transposed."""
    transB = [attr for attr in node.attribute if attr.name == "transB"]
    if len(transB):
        return 0 < helper.get_attribute_value(transB[0])
    return False

def _get_qrange_for_qType(qType, reduce_range=False):
    """Helper function to get the quantization range for a type.

    Args:
        qType (int): data type
        reduce_range (bool, optional): use 7 bit or not. Defaults to False.
    """
    if qType == onnx_proto.TensorProto.UINT8:
        return 127 if reduce_range else 255
    elif qType == onnx_proto.TensorProto.INT8:
        # [-64, 64] for reduce_range, and [-127, 127] full_range.
        return 128 if reduce_range else 254
    else:
        raise ValueError('unsupported quantization data type')

def split_shared_bias(model):
    """Split shared tensor."""
    for input_name, node_list in model.input_name_to_nodes.items():
        if len(node_list) > 1 and input_name in [i.name for i in model.model.graph.initializer]:
            for node in node_list[1:]:
                if node.op_type not in ['Conv', 'FusedConv']:
                    continue
                if len(node.input) > 2 and node.input[2] == input_name:
                    new_input_name = node.input[2] + '_nc_split_' + node.name
                    new_input = helper.make_tensor(
                                    new_input_name,
                                    model.get_initializer(input_name).data_type,
                                    model.get_initializer(input_name).dims,
                                    model.get_initializer(input_name).raw_data,
                                    True)
                    model.add_initializer(new_input)
                    node.input[2] = new_input_name
    return model    

def float_to_float16(tensor):
    """Convert float to float16."""
    min_val = 5.96e-08
    max_val = 65504.0
    tensor[(tensor > max_val) & (tensor < float('inf'))] = max_val
    tensor[(tensor < min_val) & (tensor > 0)] = min_val
    tensor[(tensor > -min_val) & (tensor < 0)] = -min_val
    tensor[(tensor < -max_val) & (tensor > float('-inf'))] = -max_val
    return np.float16(tensor)

def float_to_bfloat16(tensor):
    """Convert float to bfloat16."""
    min_val = 9.2e-41
    max_val = 3.38953139e38
    tensor[(tensor > max_val) & (tensor < float('inf'))] = max_val
    tensor[(tensor < min_val) & (tensor > 0)] = min_val
    tensor[(tensor > -min_val) & (tensor < 0)] = -min_val
    tensor[(tensor < -max_val) & (tensor > float('-inf'))] = -max_val
    return tensor

def cast_tensor(tensor, dtype): # pragma: no cover
    """Convert tensor float to target dtype.

    Args:
        tensor (TensorProto): TensorProto object
        dtype (int): target data type
    """
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError('Expected input type is an ONNX TensorProto but got %s' % type(tensor))

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        val = numpy_helper.to_array(tensor).copy()
        if dtype == 'fp16':
            new_val = float_to_float16(val)
        elif dtype == 'bf16':
            new_val = float_to_bfloat16(val)
        else:
            raise ValueError('Expect fp16 or bf16 but get {}.'.format(dtype))
        try:
            new_tensor = helper.make_tensor(
                    name=tensor.name,
                    data_type=dtype_mapping[dtype],
                    dims=numpy_helper.to_array(tensor).shape if \
                        len(numpy_helper.to_array(tensor).shape) != 0 else [],
                    vals=new_val if \
                        len(numpy_helper.to_array(tensor)) != 0 else [numpy_helper.to_array(tensor)])
            tensor.CopyFrom(new_tensor)
        except:
            tensor.float_data[:] = []
            tensor.int32_data[:] = []
            tensor.raw_data = new_val.tostring()
            tensor.data_type = dtype_mapping[dtype]
        return True
    return False

def remove_init_from_model_input(model):
    """Remove initializer from model input."""
    inputs = model.model.graph.input
    name_to_input = {}
    for inp in inputs:
        name_to_input[inp.name] = inp
    for initializer in model.model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

def collate_preds(results):
    """Collect model outputs."""
    batch = results[0]
    if isinstance(batch, list):
        results = zip(*results)
        collate_results = []
        for output in results:
           collate_results.append(np.concatenate(output))
    elif isinstance(batch, np.ndarray):
        collate_results = np.concatenate(results)
    return collate_results

def quantize_data_with_scale_zero(data, qType, scheme, scale, zero_point):
    """Quantize data with scale and zero point.
    
    To pack weights, we compute a linear transformation
        - when data type == uint8 mode, from [rmin, rmax] -> [0, 2^{b-1}] and
        - when data type == int8, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
            m = max(abs(rmin), abs(rmax))

    Args:
        data (np.array): data to quantize
        qType (int): data type to quantize to. Supported types UINT8 and INT8
        scheme (string): sym or asym quantization.
        scale (float): computed scale of quantized data
        zero_point (uint8 or int8): computed zero point of quantized data
    """
    data = np.asarray(data)
    if qType == onnx_proto.TensorProto.INT8 and scheme == 'sym':
        # signed byte type
        quantized_data = (data.astype(np.float32) / scale).round().astype('b')
    elif qType == onnx_proto.TensorProto.UINT8 and scheme == 'asym':
        quantized_data = ((data.astype(np.float32) / scale).round() + zero_point).astype('B')
    else:
        raise ValueError("Unexpected combination of data type {} and scheme {}.".format(
                                                                        qType, scheme))
    return quantized_data

def calculate_scale_zp(rmin, rmax, quantize_range, qType, scheme):
    """Calculate scale and zero point."""
    if scheme == 'sym':
        max_range = max(abs(rmin), abs(rmax))
        scale = (float(max_range) * 2) / quantize_range if max_range > 0 else 1
    else:
        scale = (float(rmax) - float(rmin)) / quantize_range if rmin != rmax else 1

    if scale == 1 or (scheme == 'sym' and qType == onnx_proto.TensorProto.INT8):
        zero_point = 0
    elif qType == onnx_proto.TensorProto.UINT8:
        zero_point = round((0 - float(rmin)) / scale)
        zero_point = np.uint8(round(max(0, min(255, zero_point))))
    else:
        zero_point = round((-64 - float(rmin)) / scale) if quantize_range == 128 \
            else round((-127 - float(rmin)) / scale)
    return scale, zero_point

def quantize_data(data, quantize_range, qType, scheme):
    """Quantize data.

    To pack weights, we compute a linear transformation
        - when data type == uint8 mode, from [rmin, rmax] -> [0, 2^{b-1}] and
        - when data type == int8, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
            m = max(abs(rmin), abs(rmax))
    and add necessary intermediate nodes to trasnform quantized weight to full weight
    using the equation r = S(q-z), where
        r: real original value
        q: quantized value
        S: scale
        z: zero point

    Args:
        data (array): data to quantize
        quantize_range (list): list of data to weight pack.
        qType (int): data type to quantize to. Supported types UINT8 and INT8
        scheme (string): sym or asym quantization.
    """
    rmin = min(min(data), 0)
    rmax = max(max(data), 0)

    scale, zero_point = calculate_scale_zp(rmin, rmax, quantize_range, qType, scheme)
    quantized_data = quantize_data_with_scale_zero(data, qType, scheme, scale, zero_point)
    return rmin, rmax, zero_point, scale, quantized_data

def quantize_data_per_channel(tensor_value, qType, scheme, scale_value, zo_value):
    """Quantize tensor per-channel."""
    channel_count = tensor_value.shape[0] # TBD, default from axis 0
    new_per_channel_tensor_values = []
    for i in range(channel_count):
        per_channel_tensor_value = tensor_value.take(i, 0)
        per_channel_scale_value = scale_value.take(i)
        per_channel_zero_value = zo_value.take(i)
        new_per_channel_tensor_values.append(quantize_data_with_scale_zero(\
                                                       per_channel_tensor_value,
                                                       qType,
                                                       scheme,
                                                       per_channel_scale_value,
                                                       per_channel_zero_value))
    # combine per_channel_data into one
    reshape_dims = list(tensor_value.shape)  # deep copy
    reshape_dims[0] = 1  # only one per channel for reshape
    new_tensor_value = new_per_channel_tensor_values[0].reshape(reshape_dims)
    for i in range(1, channel_count):
        new_per_channel_tensor_value = new_per_channel_tensor_values[i].\
                                                       reshape(reshape_dims)
        new_tensor_value = np.concatenate((new_tensor_value, \
                                           new_per_channel_tensor_value), 0)
    return new_tensor_value

def dequantize_data_with_scale_zero(tensor_value, scale_value, zo_value): # pragma: no cover
    """Dequantize tensor with sacale and zero point."""
    return (tensor_value.astype(np.float32) - zo_value.astype(np.float32)) * scale_value

def dequantize_data(tensor_value, scale_value, zo_value, axis=0): # pragma: no cover
    """Dequantize tensor."""
    if scale_value.size == 1:
        return dequantize_data_with_scale_zero(tensor_value, scale_value, zo_value)
    else:
        channel_count = tensor_value.shape[axis] # TBD, default from axis 0
        new_per_channel_tensor_values = []
        for i in range(channel_count):
            per_channel_tensor_value = tensor_value.take(i, 0)
            per_channel_scale_value = scale_value.take(i)
            per_channel_zero_value = zo_value.take(i)
            new_per_channel_tensor_values.append(dequantize_data_with_scale_zero(\
                                                           per_channel_tensor_value,
                                                           per_channel_scale_value,
                                                           per_channel_zero_value))
        # combine per_channel_data into one
        reshape_dims = list(tensor_value.shape)  # deep copy
        reshape_dims[0] = 1  # only one per channel for reshape
        new_tensor_value = new_per_channel_tensor_values[0].reshape(reshape_dims)
        for i in range(1, channel_count):
            new_per_channel_tensor_value = new_per_channel_tensor_values[i].\
                                                           reshape(reshape_dims)
            new_tensor_value = np.concatenate((new_tensor_value, \
                                               new_per_channel_tensor_value), 0)
        return new_tensor_value

class ValueInfo: # pragma: no cover
    """Represents a casted tensor info."""

    def __init__(self,
                 tensor_name,
                 dtype,
                 new_dtype):
        """Initialization.

        Args:
            tensor_name (string): tensor name
            dtype (int): original data type
            new_dtype (int): target data type
        """
        self.tensor_name = tensor_name
        self.dtype = dtype
        self.new_dtype = new_dtype

class QuantizedValue:
    """Represents a linearly quantized value (input/output/intializer)."""

    def __init__(self,
                 name,
                 new_quantized_name,
                 scale_name,
                 zero_point_name,
                 quantized_value_type,
                 axis=None,
                 qType=QuantType.QUInt8):
        """Initialization.

        Args:
            name (string): tensor name
            new_quantized_name (string): quantized tensor name
            scale_name (string): scale name
            zero_point_name (string): zero point name
            quantized_value_type (QuantizedValueType): quantized value type
            axis (int, optional): quantized axis. Defaults to None.
            qType (int, optional): quantized data type. Defaults to QuantType.QUInt8.
        """
        self.name = name
        self.q_name = new_quantized_name
        self.scale_name = scale_name
        self.zp_name = zero_point_name
        self.value_type = quantized_value_type
        self.axis = axis
        self.qType = qType

class QuantizedInitializer:
    """Represents a linearly quantized weight input from ONNX operators."""

    def __init__(self,
                 name,
                 initializer,
                 rmins,
                 rmaxs,
                 zero_points,
                 scales,
                 data=[],
                 quantized_data=[],
                 axis=None,
                 qType=QuantType.QUInt8):
        """Initialization.

        Args:
            name (string): initializer name
            initializer (onnx.onnx_ml_pb2.TensorProto): initializer
            rmins (list): list of min value
            rmaxs (list): list of max value
            zero_points (list): list of zero point
            scales (list): list of scale
            data (list, optional): array version of the initializer. Defaults to [].
            quantized_data (list, optional): quantized data. Defaults to [].
            axis (int, optional): quantized axis. Defaults to None.
            qType (int, optional): quantized data type. Defaults to QuantType.QUInt8.
        """
        self.name = name
        self.initializer = initializer  # TensorProto initializer in ONNX graph
        self.rmins = rmins  # List of minimum range for each axis
        self.rmaxs = rmaxs  # List of maximum range for each axis
        # 1D tensor of zero points computed for each axis. scalar if axis is empty
        self.zero_points = zero_points
        self.scales = scales  # 1D tensor of scales computed for each axis. scalar if axis is empty
        self.data = data  # original data from initializer TensorProto
        self.quantized_data = quantized_data  # weight-packed data from data
        # Scalar to specify which dimension in the initializer to weight pack.
        self.axis = axis
        # If empty, single zero point and scales computed from a single rmin and rmax
        self.qType = qType


class QuantizationMode(Enum): # pragma: no cover
    """Represent QuantizationMode value."""
    IntegerOps = 0
    QLinearOps = 1

class QuantizedValueType(Enum): # pragma: no cover
    """Represent QuantizedValueType value."""
    Input = 0
    Initializer = 1

class QuantFormat(Enum): # pragma: no cover
    """Represent QuantFormat value."""
    QOperator = 0
    QDQ = 1

def quantize_nparray(qtype, arr, scale, zero_point, low=None, high=None):
    """Quantize numpy array."""
    dtype = np.uint8 if qtype == "uint8" else np.int8
    cliplow = max(0 if dtype == np.uint8 else -127, -127 if low is None else low)
    cliphigh = min(255 if dtype == np.uint8 else 127, 255 if high is None else high)
    arr_fp32 = np.asarray((arr.astype(np.float32) / scale).round() + zero_point)
    np.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)
    return arr_fp32.astype(dtype)

def attribute_to_kwarg(attribute):
    """Convert attribute to kwarg format for use with onnx.helper.make_node."""
    attribute_mapping = {
        1: attribute.f,
        2: attribute.i,
        3: attribute.s,
        4: attribute.t,
        5: attribute.g,
        6: attribute.floats,
        7: attribute.ints,
        8: attribute.strings,
        9: attribute.tensors,
        10: attribute.graphs
    }
    if attribute.type in attribute_mapping:
        value = attribute_mapping[attribute.type]
    else: # pragma: no cover
        raise ValueError(
            'attribute {} has no type specified '
            'or unsupported type {}.'.format(attribute.name, attribute.type))
    return {attribute.name: value}

def find_by_name(name, item_list):
    """Helper function to find item by name in a list."""
    items = []
    for item in item_list:
        assert hasattr(item, "name"), \
            "{} should have a 'name' atrribute defined".format(item) # pragma: no cover
        if item.name == name:
            items.append(item)
    if len(items) > 0:
        return items[0]
    else:
        return None

def get_smooth_scales_per_op(max_vals_per_channel, input_tensors_2_weights,
                              input_tensors_2_weights_nodes, alpha):
    """Get the smooth scales for weights.

    The ops with the same input will share one mul layer.
    TODO support individual scales for each layer.

    Args:
        max_vals_per_channel: Max values per channel after calibration
        input_tensors_2_weights: A dict saved input tensor name and its corresponding weights
        input_tensors_2_weights_nodes:A dict saved input tensor name and its corresponding weight nodes
        alpha: smooth alpha in paper

    Returns:
        the smooth scales for weights, currently one input tensor only have one scale
    """
    scales = {}
    for key in input_tensors_2_weights_nodes.keys():
        nodes = input_tensors_2_weights_nodes[key]
        for index, node in enumerate(nodes):
            name = node.name
            weight = input_tensors_2_weights[key][index]
            if len(weight.shape) == 4:  # conv
                if weight.shape[1] == 1:  # depthwise conv
                    pass
                else:
                    weight = np.moveaxis(weight, 0, 1)
            weight = weight.reshape(weight.shape[0], -1)
            weight_max_per_channel = np.amax(weight, axis=-1)
            input_power = np.power(max_vals_per_channel[key], alpha)
            weight_power = np.power(weight_max_per_channel, 1 - alpha)
            scale = np.clip(input_power / weight_power, a_min=1e-5, a_max=None)
            scales[name] = scale
    return scales

def get_smooth_scales_per_input(max_vals_per_channel, input_tensors_2_weights, alpha):
    """Get the smooth scales for weights.

    The ops with the same input will share one mul layer.
    TODO support individual scales for each layer.

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
        for weight in weights:  # mamul ic*oc, conv oc*ic*k*k
            if len(weight.shape) == 4:  # conv
                if weight.shape[1] == 1:  # depthwise conv
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

def insert_smooth_mul_op_per_input(scales, shape_infos, input_tensors_2_weights_nodes):
    """Insert the mul layer after inupt.

    The ops with the same input will share one mul layer.

    Args:
        scales: The smooth scales
        shape_infos: the input tensor shape information
        input_tensors_2_weights_nodes:  A dict

    Returns:
        new_added_mul_nodes: added Mul layers
        new_init_tensors: added scales tensor
    """
    new_added_mul_nodes = []
    new_init_tensors = []  # scales_tensor
    for key in scales.keys():
        scale_factor = 1.0 / scales[key]
        shape_info = shape_infos[key]
        if len(shape_info) == 3 or len(shape_info) == 2:  # the last dim is input channel
            pass
        elif len(shape_info) == 4:
            scale_factor = np.reshape(scale_factor, (1, -1, 1, 1))
        else:
            assert False, "not support"
        name = key + "_" + "smooth_scale"
        scale_tensor = helper.make_tensor(
            name=name,
            data_type=onnx_proto.TensorProto.FLOAT,
            dims=scale_factor.shape,
            vals=scale_factor.flatten().tolist())
        new_init_tensors.append(scale_tensor)
        mul_output_name = key + "_smooth_output"
        mul_node = helper.make_node(
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

def adjust_weights_per_op(model, nodes, scales):
    """Adjust the weights per input scale.

    Each op has one individual Mul layer.

    Args:
        model: The onnx model
        nodes: The nodes whose weights needs to be adjustd
        scales: The input scales
    """
    name_to_indices = {}
    for index, i in enumerate(model.model.graph.initializer):
        name_to_indices[i.name] = index
    for key in nodes.keys():
        node = nodes[key]
        input = node.input[1]
        if input in name_to_indices.keys():
            weight = numpy_helper.to_array(model.model.graph.initializer[name_to_indices[input]])
            if len(weight.shape) == 2:
                scale = np.expand_dims(scales[key],
                                       axis=-1)  # TODO, to support conv
                new_weight = weight * scale
            elif len(weight.shape) == 4:  # TODO need to check conv
                scale = np.reshape(scales[key], (1, -1, 1, 1))
                new_weight = weight * scale
            else:
                assert False, "not support"
            new_tensor = numpy_helper.from_array(new_weight, input)
            model.model.graph.initializer[name_to_indices[input]].CopyFrom(new_tensor)

def adjust_weights_per_input(model, nodes, scales):
    """Adjust the weights per input scale.

    The ops with the same input will share one mul layer

    Args:
        model: The onnx model
        nodes: The nodes whose weights needs to be adjustd
        scales: The input scales
    """
    name_to_indices = {}
    for index, i in enumerate(model.model.graph.initializer):
        name_to_indices[i.name] = index
    for key in nodes.keys():
        curr_nodes = nodes[key]
        for node in curr_nodes:
            input = node.input[1]  # TODO
            if input in name_to_indices.keys():
                weight = numpy_helper.to_array(model.model.graph.initializer[name_to_indices[input]])
                if len(weight.shape) == 2:
                    scale = np.expand_dims(scales[key],
                                           axis=-1)  # TODO, to support conv
                    new_weight = weight * scale
                elif len(weight.shape) == 4:  # TODO need to check conv
                    scale = np.reshape(scales[key], (1, -1, 1, 1))
                    new_weight = weight * scale
                else:
                    assert False, "not support"
                new_tensor = numpy_helper.from_array(new_weight, input)
                model.model.graph.initializer[name_to_indices[input]].CopyFrom(new_tensor)

def insert_smooth_mul_op_per_op(scales, shape_infos, input_tensors_2_weights_nodes):
    """Insert the mul layer before op.

    Each op has one individual Mul layer.

    Args:
        scales: The smooth scales
        shape_infos: the input tensor shape information
        input_tensors_2_weights_nodes:  A dict

    Returns:
        new_added_mul_nodes: added Mul layers
        new_init_tensors: added scales tensor
        name_2_nodes: a dict, key is the node name, value is the node
    """
    name_2_nodes = {}
    for key in input_tensors_2_weights_nodes.keys():
        nodes = input_tensors_2_weights_nodes[key]
        for node in nodes:
            name_2_nodes[node.name] = node
    new_added_mul_nodes = []
    new_init_tensors = []  # scales_tensor
    for input_key in input_tensors_2_weights_nodes.keys():
        shape_info = shape_infos[input_key]
        nodes = input_tensors_2_weights_nodes[input_key]
        for node in nodes:
            key = node.name
            scale_factor = 1.0 / scales[key]
            if len(shape_info) == 3 or len(shape_info) == 2:  # the last dim is input channel
                pass
            elif len(shape_info) == 4:
                scale_factor = np.reshape(scale_factor, (1, -1, 1, 1))
            else:
                assert False, "not support"
            name = key + "_" + "smooth_scale"
            scale_tensor = helper.make_tensor(
                name=name,
                data_type=onnx_proto.TensorProto.FLOAT,
                dims=scale_factor.shape,
                vals=scale_factor.flatten().tolist())
            new_init_tensors.append(scale_tensor)
            mul_output_name = key + "_smooth_output"
            mul_node = helper.make_node(
                "Mul",
                inputs=[input_key, name],
                outputs=[mul_output_name],
                name=key + "_smooth_mul"
            )
            new_added_mul_nodes.append(mul_node)
            node = name_2_nodes[key]
            for index, input in enumerate(node.input):
                if input == input_key:
                    node.input[index] = mul_output_name
    return new_added_mul_nodes, new_init_tensors, name_2_nodes

def trt_env_setup(model):
    """Set environment variable for Tensorrt Execution Provider."""
    is_int8 = False
    for node in model.graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            is_int8 = True
            break
    if is_int8:
        os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"
    else:
        os.environ["ORT_TENSORRT_INT8_ENABLE"] = "0"
        
