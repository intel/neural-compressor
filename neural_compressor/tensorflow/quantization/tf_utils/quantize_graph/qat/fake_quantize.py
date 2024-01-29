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
"""QAT Fake Quantize Graph Class."""

import abc

import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class FakeQuantizeBase(object):
    """ABC interface class for applying fake quantization by insert qdq."""

    @abc.abstractmethod
    def __call__(self, inputs, range, training, **kwargs):
        """Apply quantization to the input tensor.

        This is the main logic of the 'FakeQuantize' which implements the core logic
        to quantize the tensor. It is invoked during the `call` stage of the layer,
        and allows modifying the tensors used in graph construction.

        Args:
            inputs (tf.Tensor): Input tensor to be quantized.
            range (dict): The min-max range of input tensor.
            training (bool): Whether the graph is currently training.
            **kwargs: Additional variables which may be passed to the FakeQuantize class.

        Returns:
            output (tf.Tensor): The tensor to be quantized.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_config(self):
        """Returns the config used to serialize the 'FakeQuantize'."""
        raise NotImplementedError("FakeQuantize should implement get_config().")

    @classmethod
    def from_config(cls, config):
        """Instantiates a 'FakeQuantize' from its config.

        Args:
            config (dict): A dict containing required information.

        Returns:
            output (FakeQuantize): A 'FakeQuantize' instance.
        """
        return cls(**config)


class FakeQuantize(FakeQuantizeBase):
    """The class that applies fake quantization."""

    def __init__(self, per_channel=False, num_bits=8, channel_axis=-1, symmetric=True, narrow_range=True):
        """Initialize a FakeQuantize class.

        Args:
            per_channel (bool): Whether to apply per_channel quantization. The last dimension is
                used as the channel.
            num_bits (int): Number of bits for quantization.
            channel_axis(int): Channel axis.
            symmetric (bool): If true, use symmetric quantization limits instead of training
                the minimum and maximum of each quantization range separately.
            narrow_range (bool): In case of 8 bits, narrow_range nudges the quantized range
                to be [-127, 127] instead of [-128, 127]. This ensures symmetric range
                has 0 as the centre.
        """
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.channel_axis = channel_axis
        self.name_prefix = "FakeQuantize"

    def __call__(self, inputs, ranges, training, **kwargs):
        """Applying fake quantization by insert qdq.

        The quantized tensor is calculated based on range of the last batch of values.

        Args:
            inputs (tf.Tensor): Input tensor to be quantized.
            range (dict): The min-max range of input tensor.
            training (bool): Whether the graph is currently training.
            **kwargs: Additional variables which may be passed to the FakeQuantize class.

        Returns:
            output (tf.Tensor): The tensor to be quantized.
        """
        with tf.name_scope(self.name_prefix):
            input_shape = inputs.get_shape()
            input_dim = len(input_shape)
            if self.channel_axis == -1:
                self.channel_axis += input_dim

            if not training:
                return self._insert_qdq(inputs, ranges["min_var"], ranges["max_var"])

            if self.per_channel:
                if input_dim == 2:
                    reduce_dims = [0]
                elif input_dim == 4:
                    reduce_dims = [i for i in range(input_dim) if i != self.channel_axis]

            if self.per_channel:
                if input_dim >= 2:
                    batch_min = tf.math.reduce_min(inputs, axis=reduce_dims, name="BatchMin")
                else:
                    batch_min = inputs
            else:
                batch_min = tf.math.reduce_min(inputs, name="BatchMin")

            if self.per_channel:
                if input_dim >= 2:
                    batch_max = tf.math.reduce_max(inputs, axis=reduce_dims, name="BatchMax")
                else:
                    batch_max = inputs
            else:
                batch_max = tf.math.reduce_max(inputs, name="BatchMax")

            if self.symmetric:
                if self.narrow_range:
                    min_max_ratio = -1
                else:
                    min_max_ratio = -((1 << self.num_bits) - 2) / (1 << self.num_bits)

                range_min = tf.math.minimum(batch_min, batch_max / min_max_ratio)
                range_max = tf.math.maximum(batch_max, batch_min * min_max_ratio)
            else:
                range_min = tf.math.minimum(batch_min, 0.0)
                range_max = tf.math.maximum(batch_max, 0.0)

            assign_min = ranges["min_var"].assign(range_min, name="AssignMinLast")
            assign_max = ranges["max_var"].assign(range_max, name="AssignMaxLast")

            return self._insert_qdq(inputs, assign_min, assign_max)

    def _insert_qdq(self, inputs, min_var, max_var):
        """Adds a fake quantization operation.

        Depending on value of self.per_channel, this operation may do global quantization
        or per channel quantization.  min_var and max_var should have corresponding
        shapes: [1] when per_channel == False and [d] when per_channel == True.

        Args:
            inputs (tf.Tensor): A tensor containing values to be quantized.
            min_var (tf.Variable): A variable containing quantization range lower end(s).
            max_var (tf.Variable): A variable containing quantization range upper end(s).

        Returns:
            outputs (tf.Tensor): A tensor containing quantized values.
        """
        if self.per_channel:
            return tf.quantization.quantize_and_dequantize_v2(
                inputs,
                min_var,
                max_var,
                num_bits=self.num_bits,
                narrow_range=self.narrow_range,
                axis=self.channel_axis,
                range_given=True,
            )
        else:
            assert min_var.get_shape() == []
            assert max_var.get_shape() == []

            return tf.quantization.quantize_and_dequantize_v2(
                inputs,
                min_var,
                max_var,
                num_bits=self.num_bits,
                narrow_range=self.narrow_range,
                range_given=True,
            )

    def get_config(self):
        """Returns the config used to serialize the 'FakeQuantize'.

        Returns:
            config (dict): A dict containing required information.
        """
        return {
            "num_bits": self.num_bits,
            "per_channel": self.per_channel,
            "symmetric": self.symmetric,
            "narrow_range": self.narrow_range,
        }

    def __eq__(self, other):
        """Check if this instance is equal to another instance.

        Args:
            other (FakeQuantize): Another instance to be checked.

        Returns:
            is_equal (bool): If the two instances are equal.
        """
        if not isinstance(other, FakeQuantize):
            return False

        return (
            self.num_bits == other.num_bits
            and self.per_channel == other.per_channel
            and self.symmetric == other.symmetric
            and self.narrow_range == other.narrow_range
        )

    def __ne__(self, other):
        """Check if this instance is not equal to another instance.

        Args:
            other (FakeQuantize): Another instance to be checked.

        Returns:
            not_equal (bool): If the two instances are not equal.
        """
        return not self.__eq__(other)
