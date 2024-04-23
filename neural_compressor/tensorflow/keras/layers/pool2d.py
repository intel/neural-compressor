#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

import json

import tensorflow as tf
from tensorflow import quantization
from tensorflow.keras import activations, backend, constraints, initializers, regularizers
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D


class QAvgPool2D(AveragePooling2D):
    def __init__(
        self,
        name,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        act_min_value=None,
        act_max_value=None,
        weight_min_value=None,
        weight_max_value=None,
        granularity="per_tensor",
        quant_status="calib",
        quant_mode="SCALED",
        quant_T="s8",
        quant_round_mode="HALF_AWAY_FROM_ZERO",
        quant_narrow_range=False,
        quant_axis=None,
        **kwargs
    ):
        super(QAvgPool2D, self).__init__(
            name=name, pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, **kwargs
        )
        T_map = {"s8": tf.qint8, "u8": tf.quint8}
        self.reverse_T_map = {tf.qint8: "s8", tf.quint8: "u8"}
        self.weight_min_value = weight_min_value
        self.weight_max_value = weight_max_value
        self.act_min_value = act_min_value
        self.act_max_value = act_max_value
        self.granularity = granularity
        self.quant_status = quant_status
        self.quant_mode = quant_mode
        self.quant_T = T_map[quant_T]
        self.quant_round_mode = quant_round_mode
        self.quant_narrow_range = quant_narrow_range
        self.quant_axis = quant_axis

    def __call__(self, inputs):
        if self.quant_status == "calib" and not isinstance(inputs, tf.keras.KerasTensor):
            if self.granularity == "per_tensor":
                self.act_min_value = tf.math.reduce_min(inputs)
                self.act_max_value = tf.math.reduce_max(inputs)
            else:
                self.act_min_value = tf.math.reduce_min(inputs, axis=1)
                self.act_max_value = tf.math.reduce_max(inputs, axis=1)
        elif self.quant_status == "quantize" and not isinstance(inputs, tf.keras.KerasTensor):
            inputs, _, _ = tf.quantization.quantize(
                inputs,
                self.act_min_value,
                self.act_max_value,
                self.quant_T,
                mode=self.quant_mode,
                round_mode=self.quant_round_mode,
                narrow_range=self.quant_narrow_range,
                axis=self.quant_axis,
            )
            inputs = tf.quantization.dequantize(
                inputs,
                self.act_min_value,
                self.act_max_value,
                mode=self.quant_mode,
                narrow_range=self.quant_narrow_range,
                axis=self.quant_axis,
            )

        return super(QAvgPool2D, self).__call__(inputs)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = super(QAvgPool2D, self).get_config()
        config.update(
            {
                "act_min_value": self.act_min_value,
                "act_max_value": self.act_max_value,
                "weight_min_value": self.weight_min_value,
                "weight_max_value": self.weight_max_value,
                "granularity": self.granularity,
                "quant_status": self.quant_status,
                "quant_mode": self.quant_mode,
                "quant_T": self.reverse_T_map[self.quant_T],
                "quant_round_mode": self.quant_round_mode,
                "quant_narrow_range": self.quant_narrow_range,
                "quant_axis": self.quant_axis,
            }
        )

        return config


class QMaxPool2D(MaxPooling2D):
    def __init__(
        self,
        name,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        act_min_value=None,
        act_max_value=None,
        weight_min_value=None,
        weight_max_value=None,
        granularity="per_tensor",
        quant_status="calib",
        quant_mode="SCALED",
        quant_T="s8",
        quant_round_mode="HALF_AWAY_FROM_ZERO",
        quant_narrow_range=False,
        quant_axis=None,
        **kwargs
    ):
        super(QMaxPool2D, self).__init__(
            name=name, pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, **kwargs
        )
        T_map = {"s8": tf.qint8, "u8": tf.quint8}
        self.reverse_T_map = {tf.qint8: "s8", tf.quint8: "u8"}
        self.weight_min_value = weight_min_value
        self.weight_max_value = weight_max_value
        self.act_min_value = act_min_value
        self.act_max_value = act_max_value
        self.granularity = granularity
        self.quant_status = quant_status
        self.quant_mode = quant_mode
        self.quant_T = T_map[quant_T]
        self.quant_round_mode = quant_round_mode
        self.quant_narrow_range = quant_narrow_range
        self.quant_axis = quant_axis

    def __call__(self, inputs):
        if self.quant_status == "calib" and not isinstance(inputs, tf.keras.KerasTensor):
            if self.granularity == "per_tensor":
                self.act_min_value = tf.math.reduce_min(inputs)
                self.act_max_value = tf.math.reduce_max(inputs)
            else:
                self.act_min_value = tf.math.reduce_min(inputs, axis=1)
                self.act_max_value = tf.math.reduce_max(inputs, axis=1)
        elif self.quant_status == "quantize" and not isinstance(inputs, tf.keras.KerasTensor):
            assert self.act_min_value is not None, "Invalid activation min-max values, please check calibration process"
            inputs, _, _ = tf.quantization.quantize(
                inputs,
                self.act_min_value,
                self.act_max_value,
                self.quant_T,
                mode=self.quant_mode,
                round_mode=self.quant_round_mode,
                narrow_range=self.quant_narrow_range,
                axis=self.quant_axis,
            )
            inputs = tf.quantization.dequantize(
                inputs,
                self.act_min_value,
                self.act_max_value,
                mode=self.quant_mode,
                narrow_range=self.quant_narrow_range,
                axis=self.quant_axis,
            )

        return super(QMaxPool2D, self).__call__(inputs)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = super(QMaxPool2D, self).get_config()
        config.update(
            {
                "act_min_value": self.act_min_value,
                "act_max_value": self.act_max_value,
                "weight_min_value": self.weight_min_value,
                "weight_max_value": self.weight_max_value,
                "granularity": self.granularity,
                "quant_status": self.quant_status,
                "quant_mode": self.quant_mode,
                "quant_T": self.reverse_T_map[self.quant_T],
                "quant_round_mode": self.quant_round_mode,
                "quant_narrow_range": self.quant_narrow_range,
                "quant_axis": self.quant_axis,
            }
        )

        return config


def initialize_int8_avgpool(fp32_layer, q_config):
    kwargs = fp32_layer.get_config()

    if "name" in kwargs:
        del kwargs["name"]
    if "pool_size" in kwargs:
        del kwargs["pool_size"]
    if "strides" in kwargs:
        del kwargs["strides"]
    if "padding" in kwargs:
        del kwargs["padding"]
    if "data_format" in kwargs:
        del kwargs["data_format"]

    q_layer = QAvgPool2D(
        name=fp32_layer.name,
        pool_size=fp32_layer.pool_size,
        strides=fp32_layer.strides,
        padding=fp32_layer.padding,
        data_format=fp32_layer.data_format,
        quant_T=q_config["T"],
        granularity=q_config["granularity"],
        **kwargs
    )

    return q_layer


def initialize_int8_maxpool(fp32_layer, q_config):
    kwargs = fp32_layer.get_config()

    if "name" in kwargs:
        del kwargs["name"]
    if "pool_size" in kwargs:
        del kwargs["pool_size"]
    if "strides" in kwargs:
        del kwargs["strides"]
    if "padding" in kwargs:
        del kwargs["padding"]
    if "data_format" in kwargs:
        del kwargs["data_format"]

    q_layer = QMaxPool2D(
        name=fp32_layer.name,
        pool_size=fp32_layer.pool_size,
        strides=fp32_layer.strides,
        padding=fp32_layer.padding,
        data_format=fp32_layer.data_format,
        quant_T=q_config["T"],
        granularity=q_config["granularity"],
        **kwargs
    )

    return q_layer
