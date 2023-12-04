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
"""Base model for multiple framework backends."""

from abc import abstractmethod


class BaseModel:
    """Base class of all neural_compressor.model, will play graph role."""

    def __init__(self, model, **kwargs):
        """Initialize a BaseModel.

        Args:
            model (object): raw model format. For Tensorflow model, could be path to frozen pb file,
                path to ckpt or savedmodel folder, loaded estimator/graph_def/graph/keras model object.
                For PyTorch model, it's torch.nn.model instance. For MXNet model, it's mxnet.symbol.Symbol
                or gluon.HybirdBlock instance. For ONNX model, it's path to onnx model or loaded ModelProto
                model object.
        """
        self.component = None

    @property
    def model(self):
        """Return model itself."""
        raise NotImplementedError

    @property
    def graph_info(self):
        """Return a dict with content 'Node: Node_type'."""
        raise NotImplementedError

    @abstractmethod
    def save(self, root, *args, **kwargs):
        """Abstract method of model saving."""
        raise NotImplementedError

    @abstractmethod
    def export(
        self,
        save_path: str,
        conf,
    ):
        """Abstract method of model conversion to ONNX."""
        raise NotImplementedError

    @abstractmethod
    def framework(self):
        """Abstract method of model framework."""
        raise NotImplementedError
