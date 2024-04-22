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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class FakeQuant(Layer):
    def __init__(self, mode="per_tensor", T="s8", **kwargs):
        super(FakeQuant, self).__init__(**kwargs)
        self.mode = mode
        self.T = T
        self.axis = 1 if mode == "per_channel" else 0
        self.min_value = tf.constant(np.finfo(np.float32).max, dtype=tf.float32)
        self.max_value = tf.constant(np.finfo(np.float32).min, dtype=tf.float32)

    def call(self, inputs):
        if self.mode == "per_tensor":
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
        return {
            "mode": self.mode,
            "min_value": self.min_value.numpy(),
            "max_value": self.max_value.numpy(),
            "T": self.T,
            "name": self.name,
        }


class Quantize(Layer):
    def __init__(
        self,
        min_range,
        max_range,
        T="s8",
        mode="SCALED",
        round_mode="HALF_AWAY_FROM_ZERO",
        narrow_range=False,
        axis=None,
        **kwargs
    ):
        super(Quantize, self).__init__(**kwargs)
        T_map = {"s8": tf.qint8, "u8": tf.quint8}
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.T = T_map[T]
        self.mode = mode
        self.round_mode = round_mode
        self.narrow_range = narrow_range
        self.axis = axis

    def call(self, inputs):
        outputs, _, _ = tf.quantization.quantize(
            inputs,
            self.min_range,
            self.max_range,
            self.T,
            mode=self.mode,
            round_mode=self.round_mode,
            narrow_range=self.narrow_range,
            axis=self.axis,
        )
        return outputs

    def get_config(self):
        return {
            "min_range": self.min_range,
            "max_range": self.max_range,
            "T": self.T,
            "mode": self.mode,
            "round_mode": self.round_mode,
            "narrow": self.narrow_range,
            "axis": self.axis,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DeQuantize(Layer):
    def __init__(self, min_range, max_range, mode="SCALED", narrow_range=False, axis=None, **kwargs):
        super(DeQuantize, self).__init__(**kwargs)
        self.min_range = min_range
        self.max_range = max_range
        self.mode = mode
        self.narrow_range = narrow_range
        self.axis = axis

    def call(self, inputs):
        return tf.quantization.dequantize(
            inputs,
            float(self.min_range),
            float(self.max_range),
            mode=self.mode,
            narrow_range=self.narrow_range,
            axis=self.axis,
        )

    def get_config(self):
        return {
            "min_range": self.min_range,
            "max_range": self.max_range,
            "mode": self.mode,
            "narrow": self.narrow_range,
            "axis": self.axis,
            "dtype": self.dtype,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UniformQuantize(Layer):
    def __init__(
        self,
        scales,
        zero_points,
        T="s8",
        quantization_axis=-1,
        **kwargs
    ):
        super(UniformQuantize, self).__init__(**kwargs)
        T_map = {"s8": tf.qint8, "u8": tf.quint8}
        self.scales = float(scales)
        self.zero_points = int(zero_points)
        self.T = T_map[T]
        self.quantization_axis = quantization_axis
        self.quantization_min_val = -128 if T=="s8" else 0
        self.quantization_max_val = 127 if T=="s8" else 255

    def call(self, inputs):
        outputs = tf.raw_ops.UniformQuantize(
            input=inputs,
            scales=self.scales,
            zero_points=self.zero_points,
            Tout=self.T,
            quantization_min_val=self.quantization_min_val,
            quantization_max_val=self.quantization_max_val,
            quantization_axis=self.quantization_axis)

        return outputs

    def get_config(self):
        return {
            "scales": self.scales,
            "zero_points": self.zero_points,
            "T": self.T,
            "quantization_axis": self.quantization_axis,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UniformDeQuantize(Layer):
    def __init__(self, 
                 scales,
                 zero_points,
                 T="s8",
                 quantization_axis=-1,
                 **kwargs):
        super(UniformDeQuantize, self).__init__(**kwargs)
        T_map = {"s8": tf.qint8, "u8": tf.quint8}
        self.scales = float(scales)
        self.zero_points = int(zero_points)
        self.T = T_map[T]
        self.quantization_axis = quantization_axis
        self.quantization_min_val = -128 if T=="s8" else 0
        self.quantization_max_val = 127 if T=="s8" else 255

    def call(self, inputs):
        return tf.raw_ops.UniformDequantize(
            input=inputs,
            scales=self.scales,
            zero_points=self.zero_points,
            Tout=tf.float32,
            quantization_min_val=self.quantization_min_val,
            quantization_max_val=self.quantization_max_val,
            quantization_axis=self.quantization_axis)

    def get_config(self):
        return {
            "scales": self.scales,
            "zero_points": self.zero_points,
            "T": self.T,
            "quantization_axis": self.quantization_axis,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
