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

from .op import Operator, operator_registry
from .tensor import Tensor


# x + y element-wise, supports broadcasting
@operator_registry(operator_type='AddV2')
class AddV2(Operator):
    def __init__(self):
        super().__init__()


# x + y element-wise, supports broadcasting
@operator_registry(operator_type='Add')
class Add(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='BinaryAdd')
class BinaryAdd(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='ConstantOfShape')
class ConstantOfShape(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='DequantizeLinear')
class DequantizeLinear(Operator):
    def __init__(self):
        super().__init__()


# Returns x / y element-wise.
# Div supports broadcasting
@operator_registry(operator_type='Div')
class Div(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='Equal')
class Equal(Operator):
    def __init__(self):
        super().__init__()


# Computes the Gauss error function of x element-wise.
@operator_registry(operator_type='Erf')
class Erf(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='Expand')
class Expand(Operator):
    def __init__(self):
        super().__init__()


# This operation creates a tensor of shape dims and fills it with value.
# tf.fill(dims, value, name=None)
@operator_registry(operator_type='Fill')
class Fill(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='FlatMapDataset')
class FlatMapDataset(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='Identity')
class Identity(Operator):
    def __init__(self):
        super().__init__()


# Fused_op MatMul + BiasAdd
# The inputs are two-dimensional matrices and 1-D const bias
@operator_registry(operator_type='InnerProduct')
class InnerProduct(Operator):
    def __init__(self):
        super().__init__()


# store input_tensors for engine
@operator_registry(operator_type='Input')
class Input(Operator):
    def __init__(self):
        super().__init__()


# Fused_op Mean, AddV2, Mul, etc.
# This pattern has several ops combinations, so the input_tensors and output_tensors may various
@operator_registry(operator_type='LayerNorm')
class LayerNorm(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='LessEqual')
class LessEqual(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='MakeIterator')
class MakeIterator(Operator):
    def __init__(self):
        super().__init__()


# Fused_op MatMulWithBias + Add/AddV2
# The inputs are two-dimensional matrices, 1-D const bias and one tensor from Add op
@operator_registry(operator_type='MatMulWithBiasAdd')
class MatMulWithBiasAdd(Operator):
    def __init__(self):
        super().__init__()


# Fused_op MatMulWithBias + Gelu
# The inputs are two-dimensional matrices and 1-D const bias
@operator_registry(operator_type='MatMulWithBiasGelu')
class MatMulWithBiasGelu(Operator):
    def __init__(self):
        super().__init__()


# Fused_op MatMulWithBias + Tanh
# The inputs are two-dimensional matrices and 1-D const bias
@operator_registry(operator_type='MatMulWithBiasTanh')
class MatMulWithBiasTanh(Operator):
    def __init__(self):
        super().__init__()


# Fused_op MatMul + BiasAdd
# The inputs are two-dimensional matrices and 1-D const bias
@operator_registry(operator_type='MatMulWithBias')
class MatMulWithBias(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='Mul')
class Mul(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='NonZero')
class NonZero(Operator):
    def __init__(self):
        super().__init__()


# store the output_tensors for engine
@operator_registry(operator_type='Output')
class Output(Operator):
    def __init__(self):
        super().__init__()


# Fused_op Reshape, ExpandDims+Sub+Mul
# This pattern is used for dealing with input_mask originally in bert model
@operator_registry(operator_type='PaddingSequence')
class PaddingSequence(Operator):
    def __init__(self):
        super().__init__()


# Given a tensor x and a tensor y ,
# this operation computes x^y for corresponding elements in x and y
@operator_registry(operator_type='Pow')
class Pow(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='QLinearMatMul')
class QLinearMatMul(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='QLinearAdd')
class QLinearAdd(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='QLinearMul')
class QLinearMul(Operator):
    def __init__(self):
        super().__init__()


# Returns x / y element-wise for real types.
# If x and y are reals, this will return the floating-point division.
# RealDiv supports broadcasting
@operator_registry(operator_type='RealDiv')
class RealDiv(Operator):
    def __init__(self):
        super().__init__()


# tf.math.rsqrt(x, name=None)
# Computes reciprocal of square root of x element-wise.
@operator_registry(operator_type='Rsqrt')
class Rsqrt(Operator):
    def __init__(self):
        super().__init__()


# tf.shape(input, out_type=tf.dtypes.int32, name=None)
# Returns a tensor containing the shape of the input tensor.
@operator_registry(operator_type='Shape')
class Shape(Operator):
    def __init__(self):
        super().__init__()


# tf.slice(input_, begin, size, name=None)
# Extracts a slice from a tensor
@operator_registry(operator_type='Slice')
class Slice(Operator):
    def __init__(self):
        super().__init__()


# Computes element-wise square root of the input tensor.
@operator_registry(operator_type='Sqrt')
class Sqrt(Operator):
    def __init__(self):
        super().__init__()


# Computes square of x element-wise.
@operator_registry(operator_type='Square')
class Square(Operator):
    def __init__(self):
        super().__init__()


# tf.math.squared_difference(x, y, name=None)
# Returns conj(x - y)(x - y) element-wise, supports broadcasting
@operator_registry(operator_type='SquaredDifference')
class SquaredDifference(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='StopGradient')
class StopGradient(Operator):
    def __init__(self):
        super().__init__()


# x - y element-wise, supports broadcasting
@operator_registry(operator_type='Sub')
class Sub(Operator):
    def __init__(self):
        super().__init__()


# Given an input tensor, this function computes hyperbolic tangent of every element in the tensor.
@operator_registry(operator_type='Tanh')
class Tanh(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='TensorSliceDataset')
class TensorSliceDataset(Operator):
    def __init__(self):
        super().__init__()


# Fused_op Reshape, Transpose and BatchMatMul / BatchMatMulV2
# This pattern has several ops combinations, so the input_tensors and output_tensors may various
@operator_registry(operator_type='TransposeBatchMatMul')
class TransposeBatchMatMul(Operator):
    def __init__(self):
        super().__init__()


@operator_registry(operator_type='Where')
class Where(Operator):
    def __init__(self):
        super().__init__()

@operator_registry(operator_type='Range')
class Range(Operator):
    def __init__(self):
        super().__init__()

@operator_registry(operator_type='Relu')
class Relu(Operator):
    def __init__(self):
        super().__init__()

@operator_registry(operator_type='MatMulWithBiasRelu')
class MatMulWithBiasRelu(Operator):
    def __init__(self):
        super().__init__()