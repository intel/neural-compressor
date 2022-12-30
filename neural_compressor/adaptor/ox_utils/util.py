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

def cast_tensor(tensor, dtype): # pragma: no cover
    """Convert tensor float to target dtype.

    Args:
        tensor (TensorProto): TensorProto object
        dtype (int): target data type
    """
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
