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
#

import os
import numpy as np
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto  
from enum import Enum
from pathlib import Path
import abc

__producer__ = "onnx.quantize"
__version__ = "0.1.0"
onnx_domain = "ai.onnx"
ms_domain = "com.microsoft"      

support_pair = {
    'uint8 uint8': True,
    '2 2': True,
    'float16 float16': True,
    '10 10': True,
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
}

def dtype_to_name(dtype_mapping, dtype):
    return list(dtype_mapping.keys())[list(dtype_mapping.values()).index(dtype)]

class QuantType(Enum): # pragma: no cover
    QInt8 = 0
    QUInt8 = 1

def make_quant_node(name, inputs, outputs):
    return helper.make_node("QuantizeLinear", inputs, outputs, name)

def make_dquant_node(name, inputs, outputs, axis=None):
    if axis is not None:
        return helper.make_node("DequantizeLinear", inputs, outputs, name, axis=axis)
    else:
        return helper.make_node("DequantizeLinear", inputs, outputs, name)

def is_B_transposed(node):
    transB = [attr for attr in node.attribute if attr.name == "transB"]
    if len(transB):
        return 0 < helper.get_attribute_value(transB[0])
    return False

def _get_qrange_for_qType(qType, reduce_range=False):
    '''
    Helper function to get the quantization range for a type.
        parameter qType: quantization type.
        return: quantization range.
    '''
    if qType == onnx_proto.TensorProto.UINT8:
        return 127 if reduce_range else 255
    elif qType == onnx_proto.TensorProto.INT8:
        # [-64, 64] for reduce_range, and [-127, 127] full_range.
        return 128 if reduce_range else 254
    else:
        raise ValueError('unsupported quantization data type')

def split_shared_bias(model):
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

def cast_tensor(tensor, dtype): # pragma: no cover
    '''
    Convert tensor float to target dtype.
        param tensor: TensorProto object
        return tensor_target_dtype: converted TensorProto object
    '''
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError('Expected input type is an ONNX TensorProto but got %s' % type(tensor))

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        new_tensor = helper.make_tensor(
            name=tensor.name,
            data_type=dtype_mapping[dtype],
            dims=numpy_helper.to_array(tensor).shape,
            vals=numpy_helper.to_array(tensor)
        )
        return new_tensor
    return None

def remove_init_from_model_input(model):
    inputs = model.model.graph.input
    name_to_input = {}
    for inp in inputs:
        name_to_input[inp.name] = inp
    for initializer in model.model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

def collate_preds(results):
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
    '''
        :parameter data: data to quantize
        :parameter qType: data type to quantize to. Supported types UINT8 and INT8
        :parameter scheme: sym or asym quantization.
        :parameter scale: computed scale of quantized data
        :parameter zero_point: computed zero point of quantized data
        :return: quantized weights
        To pack weights, we compute a linear transformation
            - when data type == uint8 mode, from [rmin, rmax] -> [0, 2^{b-1}] and
            - when data type == int8, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
                m = max(abs(rmin), abs(rmax))
    '''
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

def quantize_data(data, quantize_range, qType, scheme):
    '''
        :parameter data: data to quantize
        :parameter quantize_range: list of data to weight pack.
        :parameter qType: data type to quantize to. Supported types UINT8 and INT8
        :param scheme: sym or asym quantization.
        :return: minimum, maximum, zero point, scale, and quantized weights
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
    '''
    rmin = min(min(data), 0)
    rmax = max(max(data), 0)

    if scheme == 'sym' and qType == onnx_proto.TensorProto.INT8:
        max_range = max(abs(rmin), abs(rmax))
        scale = (float(max_range) * 2) / quantize_range if max_range > 0 else 1
        zero_point = 0
    elif scheme == 'asym' and qType == onnx_proto.TensorProto.UINT8:
        scale = (float(rmax) - rmin) / quantize_range if rmin != rmax else 1
        zero_point = round((0 - rmin) / scale)
    else:
        raise ValueError("Unexpected combination of data type {} and scheme {}.".format(
            qType, scheme))

    quantized_data = quantize_data_with_scale_zero(data, qType, scheme, scale, zero_point)
    return rmin, rmax, zero_point, scale, quantized_data

def quantize_data_per_channel(tensor_value, qType, scheme, scale_value, zo_value):
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
    return (tensor_value.astype(np.float32) - zo_value.astype(np.float32)) * scale_value

def dequantize_data(tensor_value, scale_value, zo_value, axis=0): # pragma: no cover
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
    def __init__(self,
                 tensor_name,
                 dtype,
                 new_dtype):
        self.tensor_name = tensor_name
        self.dtype = dtype
        self.new_dtype = new_dtype

class QuantizedValue:
    '''
    Represents a linearly quantized value (input\output\intializer)
    '''
    def __init__(self,
                 name,
                 new_quantized_name,
                 scale_name,
                 zero_point_name,
                 quantized_value_type,
                 axis=None,
                 qType=QuantType.QUInt8):
        self.name = name
        self.q_name = new_quantized_name
        self.scale_name = scale_name
        self.zp_name = zero_point_name
        self.value_type = quantized_value_type
        self.axis = axis
        self.qType = qType

class QuantizedInitializer:
    '''
        Represents a linearly quantized weight input from ONNX operators
    '''
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
    IntegerOps = 0
    QLinearOps = 1

class QuantizedValueType(Enum): # pragma: no cover
    Input = 0
    Initializer = 1

class QuantFormat(Enum): # pragma: no cover
    QOperator = 0
    QDQ = 1

def quantize_nparray(qtype, arr, scale, zero_point, low=None, high=None):
    dtype = np.uint8 if qtype == "uint8" else np.int8
    cliplow = max(0 if dtype == np.uint8 else -127, -127 if low is None else low)
    cliphigh = min(255 if dtype == np.uint8 else 127, 255 if high is None else high)
    arr_fp32 = np.asarray((arr.astype(np.float32) / scale).round() + zero_point)
    np.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)
    return arr_fp32.astype(dtype)

def attribute_to_kwarg(attribute):
    '''
    Convert attribute to kwarg format for use with onnx.helper.make_node.
    '''
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
    '''
    Helper function to find item by name in a list.
    '''
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
