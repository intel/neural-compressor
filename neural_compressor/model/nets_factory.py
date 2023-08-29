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
"""TF-Slim nets factory."""

from ..utils.utility import singleton


@singleton
class TFSlimNetsFactory(object):
    """TF-Slim nets factory."""

    def __init__(self):
        """Initialize a TFSlimNetsFactory."""
        # tf_slim only support specific models by default
        self.default_slim_models = [
            "alexnet_v2",
            "overfeat",
            "vgg_a",
            "vgg_16",
            "vgg_19",
            "inception_v1",
            "inception_v2",
            "inception_v3",
            "resnet_v1_50",
            "resnet_v1_101",
            "resnet_v1_152",
            "resnet_v1_200",
            "resnet_v2_50",
            "resnet_v2_101",
            "resnet_v2_152",
            "resnet_v2_200",
        ]

        from tf_slim.nets import alexnet, inception, overfeat, resnet_v1, resnet_v2, vgg

        self.networks_map = {
            "alexnet_v2": {
                "model": alexnet.alexnet_v2,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1001,
                "arg_scope": alexnet.alexnet_v2_arg_scope,
            },
            "overfeat": {
                "model": overfeat.overfeat,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1001,
                "arg_scope": overfeat.overfeat_arg_scope,
            },
            "vgg_a": {
                "model": vgg.vgg_a,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1000,
                "arg_scope": vgg.vgg_arg_scope,
            },
            "vgg_16": {
                "model": vgg.vgg_16,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1000,
                "arg_scope": vgg.vgg_arg_scope,
            },
            "vgg_19": {
                "model": vgg.vgg_19,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1000,
                "arg_scope": vgg.vgg_arg_scope,
            },
            "inception_v1": {
                "model": inception.inception_v1,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1001,
                "arg_scope": inception.inception_v1_arg_scope,
            },
            "inception_v2": {
                "model": inception.inception_v2,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1001,
                "arg_scope": inception.inception_v2_arg_scope,
            },
            "inception_v3": {
                "model": inception.inception_v3,
                "input_shape": [None, 299, 299, 3],
                "num_classes": 1001,
                "arg_scope": inception.inception_v3_arg_scope,
            },
            "resnet_v1_50": {
                "model": resnet_v1.resnet_v1_50,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1000,
                "arg_scope": resnet_v1.resnet_arg_scope,
            },
            "resnet_v1_101": {
                "model": resnet_v1.resnet_v1_101,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1000,
                "arg_scope": resnet_v1.resnet_arg_scope,
            },
            "resnet_v1_152": {
                "model": resnet_v1.resnet_v1_152,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1000,
                "arg_scope": resnet_v1.resnet_arg_scope,
            },
            "resnet_v1_200": {
                "model": resnet_v1.resnet_v1_200,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1000,
                "arg_scope": resnet_v1.resnet_arg_scope,
            },
            "resnet_v2_50": {
                "model": resnet_v2.resnet_v2_50,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1001,
                "arg_scope": resnet_v2.resnet_arg_scope,
            },
            "resnet_v2_101": {
                "model": resnet_v2.resnet_v2_101,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1001,
                "arg_scope": resnet_v2.resnet_arg_scope,
            },
            "resnet_v2_152": {
                "model": resnet_v2.resnet_v2_152,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1001,
                "arg_scope": resnet_v2.resnet_arg_scope,
            },
            "resnet_v2_200": {
                "model": resnet_v2.resnet_v2_200,
                "input_shape": [None, 224, 224, 3],
                "num_classes": 1001,
                "arg_scope": resnet_v2.resnet_arg_scope,
            },
        }

    def register(self, name, model_func, input_shape, arg_scope, **kwargs):
        """Register a model to TFSlimNetsFactory.

        Args:
            name (str): name of a model.
            model_func (_type_): model that built from slim.
            input_shape (_type_): input tensor shape.
            arg_scope (_type_): slim arg scope that needed.
        """
        net_info = {"model": model_func, "input_shape": input_shape, "arg_scope": arg_scope}
        net = {name: {**net_info, **kwargs}}
        self.networks_map.update(net)
        self.default_slim_models.append(name)
