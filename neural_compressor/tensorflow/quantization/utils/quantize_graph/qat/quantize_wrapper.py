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
"""QAT Quantize Wrapper Class."""

from abc import abstractmethod

import tensorflow as tf
from tensorflow.python.util import tf_inspect

from .fake_quantize import FakeQuantize
from .quantize_config import global_config, layer_wise_config


class QuantizeWrapperBase(tf.keras.layers.Wrapper):
    """Base class for quantize wrapper."""

    def __init__(self, layer, **kwargs):
        """Create a quantize wrapper for a keras layer.

        This wrapper provides options to quantize inputs and weights of the layer.

        Args:
          layer (tf.keras.layers.Layer): The keras layer to be wrapped.
          **kwargs: Additional keyword arguments to be passed.
        """
        assert layer is not None, "'layer' should not be None."

        assert isinstance(layer, tf.keras.layers.Layer) or isinstance(layer, tf.keras.Model), (
            "'layer' can only be a 'tf.keras.layers.Layer' instance."
            " You passed an instance of type: {input}.".format(input=layer.__class__.__name__)
        )

        if "name" not in kwargs:
            kwargs["name"] = self._make_layer_name(layer)

        super(QuantizeWrapperBase, self).__init__(layer, **kwargs)

        self.index = None
        self._layer_class = layer.__class__.__name__
        self._track_trackable(layer, name="layer")

    @staticmethod
    def _make_layer_name(layer):
        """Modify the layer name to be quantized layer."""
        return "{}_{}".format("quant", layer.name)

    def build(self, input_shape):
        """Creates the variables of the layer.

        Args:
          input_shape (tf.TensorShape or list): shapes of input tensors
        """
        super(QuantizeWrapperBase, self).build(input_shape)

        self.optimizer_step = self.add_weight(
            "optimizer_step",
            initializer=tf.keras.initializers.Constant(-1),
            dtype=tf.dtypes.int32,
            trainable=False,
        )

    def _init_min_max_variables(self, name, shape):
        """Initialize the minimum and maximum values of variables to the wrapped layer.

        Args:
            name (string): Name prefix of the variables.
            shape (tf.TensorShape): shape of variables to be added.

        Returns:
            min_variable (tf.Variable) : The initialized minimum value of given variables.
            min_variable (tf.Variable) : The initialized maximum value of given variables.
        """
        min_variable = self.layer.add_weight(
            name + "_min",
            shape=(shape),
            trainable=False,
            initializer=tf.keras.initializers.Constant(-6.0),
        )
        max_variable = self.layer.add_weight(
            name + "_max",
            shape=(shape),
            trainable=False,
            initializer=tf.keras.initializers.Constant(6.0),
        )

        return min_variable, max_variable

    def query_input_index(self):
        """Query QuantizeConfig to check if there is any designated input index for this layer."""
        quantize_config = global_config["quantize_config"]
        custom_layer_config = quantize_config.query_layer(self.layer)
        if custom_layer_config and "index" in custom_layer_config:
            self.index = custom_layer_config["index"]

    @abstractmethod
    def call(self, inputs, training=None):
        """This is where the quantize wrapper's logic lives.

        Args:
          inputs (tf.Tensor or dict/list/tuple): Inputs of the wrapped layer.

        Returns:
          outputs (tf.Tensor or dict/list/tuple): Outputs of the wrapped layer.
        """
        raise NotImplementedError

    @property
    def trainable(self):
        """Get trainable attribute for the layer and its sublayers."""
        return self.layer.trainable

    @trainable.setter
    def trainable(self, value):
        """Set trainable attribute for the layer and its sublayers.

        Args:
          value (Boolean): The desired state for the layer's trainable attribute.
        """
        self.layer.trainable = value

    @property
    def trainable_weights(self):
        """List of all trainable weights tracked by this layer.

        Trainable weights are updated via gradient descent during training.

        Returns:
          trainable_weights (list): A list of trainable variables.
        """
        return self.layer.trainable_weights + self._trainable_weights

    @property
    def non_trainable_weights(self):
        """List of all non-trainable weights tracked by this layer.

        Non-trainable weights are *not* updated during training. They are
        expected to be updated manually in `call()`.

        Returns:
          non_trainable_weights (list): A list of non-trainable variables.
        """
        return self.layer.non_trainable_weights + self._non_trainable_weights

    @property
    def updates(self):
        """Update layer."""
        return self.layer.updates + self._updates

    @property
    def losses(self):
        """List of losses added using the `add_loss()` API.

        Variable regularization tensors are created when this property is
        accessed, so it is eager safe: accessing `losses` under a
        `tf.GradientTape` will propagate gradients back to the corresponding
        variables.

        Returns:
          losses (list): A list of tensors.
        """
        return self.layer.losses + self._losses


class QuantizeWrapper(QuantizeWrapperBase):
    """General QuantizeWrapper for quantizable layers.

    Weights and inputs will be quantized according to the layer type and quantize config.
    """

    def __init__(self, layer, **kwargs):
        """Create a quantize wrapper for a keras layer.

        This wrapper provides options to quantize inputs and weights of the layer.

        Args:
          layer (tf.keras.layers.Layer): The keras layer to be wrapped.
          **kwargs: Additional keyword arguments to be passed.
        """
        super().__init__(layer, **kwargs)

        self.kernel = "kernel"
        self.kernel_weights = None
        self.channel_axis = kwargs.get("axis", -1)
        if self._layer_class == "DepthwiseConv2D":
            self.kernel = "depthwise_kernel"
            self.channel_axis = 2
        if self._layer_class in layer_wise_config["multiple_inputs_layers"]:
            self.query_input_index()

    def build(self, input_shape):
        """Creates the variables of the layer.

        Args:
          input_shape (tf.TensorShape or list): shapes of input tensors
        """
        super().build(input_shape)

        if self._layer_class in layer_wise_config["weighted_layers"]:
            self.kernel_weights = getattr(self.layer, self.kernel)

            weight_min, weight_max = self._init_min_max_variables(
                name=self.kernel_weights.name.split(":")[0], shape=self.kernel_weights.shape[self.channel_axis]
            )

            self.weight_range = {"min_var": weight_min, "max_var": weight_max}
            self._trainable_weights.append(self.kernel_weights)

        num_input = 1
        if not isinstance(input_shape, tf.TensorShape):
            num_input = len(input_shape)
        self.query_input_index()
        if not self.index:
            self.index = [i for i in range(num_input)]

        if num_input == 1:
            inputs_min, inputs_max = self._init_min_max_variables(
                name=self.layer.name + "_input{}".format(0), shape=None
            )
            self.inputs_range = {"min_var": inputs_min, "max_var": inputs_max}
        else:
            self.inputs_range = []
            for i in range(num_input):
                self.inputs_range.append({})
                if i in self.index:
                    inputs_min, inputs_max = self._init_min_max_variables(
                        name=self.layer.name + "_input{}".format(i), shape=None
                    )
                    self.inputs_range[i] = {"min_var": inputs_min, "max_var": inputs_max}

    def call(self, inputs, training=None):
        """This is where the quantize wrapper's logic lives.

        Args:
          inputs (tf.Tensor or dict/list/tuple): Inputs of the wrapped layer.

        Returns:
          outputs (tf.Tensor or dict/list/tuple): Outputs of the wrapped layer.
        """
        if training is None:
            training = tf.keras.backend.learning_phase()

        # Quantize all weights, and replace them in the underlying layer.
        if self._layer_class in layer_wise_config["weighted_layers"]:
            weight_quantizer = FakeQuantize(
                per_channel=True,
                channel_axis=self.channel_axis,
            )
            quantized_weight = weight_quantizer(self.kernel_weights, self.weight_range, training)
            setattr(self.layer, self.kernel, quantized_weight)

        quantized_inputs = inputs
        inputs_quantizer = FakeQuantize(
            per_channel=False,
            channel_axis=self.channel_axis,
        )

        if not isinstance(quantized_inputs, tf.Tensor):
            for i in range(len(quantized_inputs)):
                if i in self.index:
                    quantized_inputs[i] = inputs_quantizer(inputs[i], self.inputs_range[i], training)
        else:
            quantized_inputs = inputs_quantizer(inputs, self.inputs_range, training)

        args = tf_inspect.getfullargspec(self.layer.call).args
        if "training" in args:
            outputs = self.layer.call(quantized_inputs, training=training)
        else:
            outputs = self.layer.call(quantized_inputs)

        return outputs
