#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

import sys
import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.util.tf_export import keras_export

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers.core import Dense

class FakeQuant(Layer):
    def __init__(self, mode='per_tensor', **kwargs):
        super(FakeQuant, self).__init__(**kwargs)
        self.mode = mode
        self.axis = 1 if mode == 'per_channel' else 0
        self.min_value = tf.constant(np.finfo(np.float32).max, dtype=tf.float32)
        self.max_value = tf.constant(np.finfo(np.float32).min, dtype=tf.float32)

    def call(self, inputs):
        if self.mode == 'per_tensor':
            self.min_value = tf.math.reduce_min(inputs)
            self.max_value = tf.math.reduce_max(inputs)
        else:
            self.min_value = tf.math.reduce_min(inputs, axis=self.axis)
            self.max_value = tf.math.reduce_max(inputs, axis=self.axis)
        return inputs

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return {'mode': self.mode,
                'min_value': self.min_value.numpy(),
                'max_value': self.max_value.numpy(),
                'name': self.name}

class Quantize(Layer):
    def __init__(self, min_range, max_range, T=tf.qint8, mode='SCALED', 
                 round_mode='HALF_AWAY_FROM_ZERO', narrow_range=False, axis=None):
        super(Quantize, self).__init__()
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.T = T
        self.mode = mode
        self.round_mode = round_mode
        self.narrow_range = narrow_range
        self.axis = axis

    def call(self, inputs):
        outputs, _, _ = tf.quantization.quantize(inputs, self.min_range,
                                        self.max_range, self.T,
                                        mode=self.mode, round_mode=self.round_mode,
                                        narrow_range=self.narrow_range, axis=self.axis)
        return outputs

    def get_config(self):
        return {'min_range': self.min_range, 'max_range': self.max_range,
                'T': self.T, 'mode': self.mode, 'round_mode': self.round_mode,
                'narrow': self.narrow_range, 'axis': self.axis}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class QConv2D(Conv):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid',
                 data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
                 use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 min_value=-10000, max_value=10000, **kwargs):
        super(QConv2D, self).__init__(rank=2, filters=filters, kernel_size=kernel_size, 
                strides=strides, padding=padding, data_format=data_format, 
                dilation_rate=dilation_rate, groups=groups,
                activation=activations.get(activation),
                use_bias=use_bias, kernel_initializer=initializers.get(kernel_initializer),
                bias_initializer=initializers.get(bias_initializer),
                kernel_regularizer=regularizers.get(kernel_regularizer),
                bias_regularizer=regularizers.get(bias_regularizer), 
                activity_regularizer=regularizers.get(activity_regularizer),
                kernel_constraint=constraints.get(kernel_constraint),
                bias_constraint=constraints.get(bias_constraint), **kwargs)
        self.weight_quantizer = Quantize(float(min_value), float(max_value))
        self.weight_dequantizer = DeQuantize(float(min_value), float(max_value))

    def call(self, inputs):
      input_shape = inputs.shape

      if self._is_causal:  # Apply causal padding to inputs for Conv1D.
        inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

      # add the Q/DQ here
      kernel = self.weight_quantizer(self.kernel)
      kernel = self.weight_dequantizer(kernel)
      outputs = self._convolution_op(inputs, kernel)

      if self.use_bias:
        output_rank = outputs.shape.rank
        if self.rank == 1 and self._channels_first:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        else:
          # Handle multiple batch dimensions.
          if output_rank is not None and output_rank > 2 + self.rank:

            def _apply_fn(o):
              return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

            outputs = conv_utils.squeeze_batch_dims(
                outputs, _apply_fn, inner_rank=self.rank + 1)
          else:
            outputs = nn.bias_add(
                outputs, self.bias, data_format=self._tf_data_format)

      if not context.executing_eagerly():
        # Infer the static output shape:
        out_shape = self.compute_output_shape(input_shape)
        outputs.set_shape(out_shape)

      if self.activation is not None:
        return self.activation(outputs)
      return outputs

class QDense(Dense):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 min_value=-10000,
                 max_value=10000,
                 **kwargs):
      super(QDense, self).__init__(
                          units=units,
                          activation=activation,
                          use_bias=use_bias,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          activity_regularizer=activity_regularizer,
                          kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint,
                          **kwargs)
      self.weight_quantizer = Quantize(float(min_value), float(max_value))
      self.weight_dequantizer = DeQuantize(float(min_value), float(max_value))

    def call(self, inputs):
      if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
        inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

      # add the Q/DQ here
      # (TODO) we have not try sparse dense and may have issues
      kernel = self.weight_quantizer(self.kernel)
      kernel = self.weight_dequantizer(kernel)
      rank = inputs.shape.rank
      if rank == 2 or rank is None:
        # We use embedding_lookup_sparse as a more efficient matmul operation for
        # large sparse input tensors. The op will result in a sparse gradient, as
        # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
        # gradients. This can lead to sigfinicant speedups, see b/171762937.
        if isinstance(inputs, sparse_tensor.SparseTensor):
          # We need to fill empty rows, as the op assumes at least one id per row.
          inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
          # We need to do some munging of our input to use the embedding lookup as
          # a matrix multiply. We split our input matrix into separate ids and
          # weights tensors. The values of the ids tensor should be the column
          # indices of our input matrix and the values of the weights tensor
          # can continue to the actual matrix weights.
          # The column arrangement of ids and weights
          # will be summed over and does not matter. See the documentation for
          # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
          # of the inputs to both ops.
          ids = sparse_tensor.SparseTensor(
              indices=inputs.indices,
              values=inputs.indices[:, 1],
              dense_shape=inputs.dense_shape)
          weights = inputs
          outputs = embedding_ops.embedding_lookup_sparse_v2(
              kernel, ids, weights, combiner='sum')
        else:
          outputs = gen_math_ops.MatMul(a=inputs, b=kernel)
      # Broadcast kernel to inputs.
      else:
        outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
        # Reshape the output back to the original ndim of the input.
        if not context.executing_eagerly():
          shape = inputs.shape.as_list()
          output_shape = shape[:-1] + [kernel.shape[-1]]
          outputs.set_shape(output_shape)

      if self.use_bias:
        outputs = nn_ops.bias_add(outputs, self.bias)

      if self.activation is not None:
        outputs = self.activation(outputs)
      return outputs
                                      

class DeQuantize(Layer):
    def __init__(self, min_range, max_range, mode='SCALED',
                 narrow_range=False, axis=None):
        super(DeQuantize, self).__init__()
        self.min_range = min_range
        self.max_range = max_range
        self.mode = mode
        self.narrow_range = narrow_range
        self.axis = axis

    def call(self, inputs):
        return tf.quantization.dequantize(inputs, float(self.min_range),
                                          float(self.max_range), mode=self.mode,
                                          narrow_range=self.narrow_range, axis=self.axis)
    def get_config(self):
        return {'min_range': self.min_range, 'max_range': self.max_range,
                'mode': self.mode, 'narrow': self.narrow_range, 'axis': self.axis,
                'dtype': self.dtype}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
