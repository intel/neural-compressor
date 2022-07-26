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
from onnx import helper
from onnx import onnx_pb as onnx_proto  
from onnxruntime.quantization.quant_utils import QuantType       

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
    'fp16': 10,
    'int8': 3,
    'uint8': 2,
}

def make_quant_node(name, inputs, outputs):
    return helper.make_node("QuantizeLinear", inputs, outputs, name)

def make_dquant_node(name, inputs, outputs):
    return helper.make_node("DequantizeLinear", inputs, outputs, name)

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
                if node.input[2] == input_name:
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

def convert_np_to_float16(np_array, min_positive_val=1e-7, max_finite_val=1e4): # pragma: no cover
    '''
    Convert float32 numpy array to float16 without changing sign or finiteness.
    Positive values less than min_positive_val are mapped to min_positive_val.
    Positive finite values greater than max_finite_val are mapped to max_finite_val.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.
    '''
    def between(a, b, c):
        return np.logical_and(a < b, b < c)
    np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
    np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(between(max_finite_val, np_array, float('inf')), max_finite_val, np_array)
    np_array = np.where(between(float('-inf'), np_array, -max_finite_val), -max_finite_val, np_array)
    return np.float16(np_array)

def _npfloat16_to_int(np_list):
    '''
    Convert numpy float16 to python int.
        param np_list: numpy float16 list
        return int_list: python int list
    '''
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in np_list]

def cast_tensor(tensor, dtype, min_positive_val=1e-7, max_finite_val=1e4): # pragma: no cover
    '''
    Convert tensor float to float16.
        param tensor: TensorProto object
        return tensor_float16: converted TensorProto object
    Example:
        from onnxmltools.utils.float16_converter import convert_tensor_float_to_float16
        new_tensor = convert_tensor_float_to_float16(tensor)
    '''
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError('Expected input type is an ONNX TensorProto but got %s' % type(tensor))

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = convert_np_to_float16(np.array(tensor.float_data), min_positive_val, max_finite_val)
            int_list = _npfloat16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.fromstring(tensor.raw_data, dtype='float32')
            # convert float to float16
            float16_list = convert_np_to_float16(float32_list, min_positive_val, max_finite_val)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tostring()
    return tensor

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
    if qType == onnx_proto.TensorProto.INT8 and scheme == 'sym':
        # signed byte type
        quantized_data = (np.asarray(data) / scale).round().astype('b')
    elif qType == onnx_proto.TensorProto.UINT8 and scheme == 'asym':
        quantized_data = ((np.asarray(data) / scale).round() + zero_point).astype('B')
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
