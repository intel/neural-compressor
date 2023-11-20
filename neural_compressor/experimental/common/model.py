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
"""Common Model just collects the information to construct a Model."""
from deprecated import deprecated

from neural_compressor.model.model import MODELS, get_model_fwk_name
from neural_compressor.model.tensorflow_model import get_model_type
from neural_compressor.utils import logger

BACKEND = "default"


@deprecated(version="2.0")
class Model(object):
    """A wrapper of the information needed to construct a Model."""

    def __new__(cls, root, **kwargs):
        """Create a new instance object of Model.

        Args:
            root (object): raw model format. For Tensorflow model, could be path to frozen pb file,
                path to ckpt or savedmodel folder, loaded estimator/graph_def/graph/keras model object.
                For PyTorch model, it's torch.nn.model instance. For MXNet model, it's mxnet.symbol.Symbol
                or gluon.HybirdBlock instance. For ONNX model, it's path to onnx model or loaded ModelProto
                model object.

        Returns:
            BaseModel: neural_compressor built-in model
        """
        framework = kwargs.get("framework", "NA")
        if framework == "NA":
            framework = get_model_fwk_name(root)

        if "tensorflow" in framework:
            if "modelType" in kwargs:
                model_type = kwargs["modelType"]
            else:
                model_type = get_model_type(root)
            if model_type == "AutoTrackable":  # pragma: no cover
                model = MODELS["tensorflow"]("keras", root, **kwargs)
            else:
                model = MODELS["tensorflow"](model_type, root, **kwargs)
        elif framework == "keras":
            model = MODELS["keras"](root, **kwargs)
        elif framework == "pytorch":
            if BACKEND != "default":
                framework = BACKEND
            model = MODELS[framework](root, **kwargs)
        else:
            model = MODELS[framework](root, **kwargs)
        return model


@deprecated(version="2.0")
def set_backend(backend: str):
    """Set backed from configure file."""
    global BACKEND
    BACKEND = backend
