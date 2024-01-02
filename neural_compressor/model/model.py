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
"""Model for multiple framework backends."""

import copy
import importlib
import os
import sys

from neural_compressor.config import options
from neural_compressor.model.base_model import BaseModel
from neural_compressor.model.keras_model import KerasModel
from neural_compressor.model.mxnet_model import MXNetModel
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.model.tensorflow_model import (
    TensorflowBaseModel,
    TensorflowLLMModel,
    TensorflowModel,
    TensorflowQATModel,
    get_model_type,
)
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

TORCH = False
if importlib.util.find_spec("torch"):
    TORCH = True
    from neural_compressor.model.torch_model import *

torch = LazyImport("torch")
tf = LazyImport("tensorflow")
mx = LazyImport("mxnet")
onnx = LazyImport("onnx")
ort = LazyImport("onnxruntime")
yaml = LazyImport("yaml")
json = LazyImport("json")
np = LazyImport("numpy")

MODELS = {
    "tensorflow": TensorflowModel,
    "tensorflow_itex": TensorflowModel,
    "tensorflow_qat": TensorflowQATModel,
    "keras": KerasModel,
    "mxnet": MXNetModel,
    "pytorch": PyTorchModel if TORCH else None,
    "pytorch_ipex": IPEXModel if TORCH else None,
    "pytorch_fx": PyTorchFXModel if TORCH else None,
    "onnxruntime": ONNXModel,
    "onnxrt_qlinearops": ONNXModel,
    "onnxrt_qdq": ONNXModel,
    "onnxrt_integerops": ONNXModel,
}


def get_model_fwk_name(model):
    """Detect the input model belongs to which framework.

    Args:
        model (string): framework name that supported by Neural Compressor, if there's no available fwk info,
                        then return 'NA'.
    """

    def _is_onnxruntime(model):
        from importlib.util import find_spec

        try:
            so = ort.SessionOptions()
            if sys.version_info < (3, 11) and find_spec("onnxruntime_extensions"):  # pragma: no cover
                from onnxruntime_extensions import get_library_path

                so.register_custom_ops_library(get_library_path())
            if isinstance(model, str):
                ort.InferenceSession(model, so, providers=ort.get_available_providers())
            else:
                ort.InferenceSession(model.SerializeToString(), so, providers=ort.get_available_providers())
        except Exception as e:  # pragma: no cover
            if "Message onnx.ModelProto exceeds maximum protobuf size of 2GB" in str(e):
                logger.warning("Please use model path instead of onnx model object to quantize")
            else:
                logger.warning(
                    "If you use an onnx model with custom_ops to do quantiztaion, "
                    "please ensure onnxruntime-extensions is installed"
                )
        else:
            return "onnxruntime"
        return "NA"

    def _is_pytorch(model):
        try:
            if (
                isinstance(model, torch.nn.Module)
                or isinstance(model, torch.fx.GraphModule)
                or isinstance(model, torch.jit._script.RecursiveScriptModule)
            ):
                return "pytorch"
            else:
                return "NA"
        except:
            return "NA"

    def _is_tensorflow(model):
        try:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            model_type = get_model_type(model)
        except:
            os.environ.pop("CUDA_DEVICE_ORDER")
            os.environ.pop("CUDA_VISIBLE_DEVICES")
            return "NA"
        else:
            return "tensorflow"

    def _is_mxnet(model):
        try:
            is_mxnet = isinstance(model, mx.gluon.HybridBlock) or (
                hasattr(model, "__len__") and len(model) > 1 and isinstance(model[0], mx.symbol.Symbol)
            )
        except:
            return "NA"
        else:
            return "mxnet" if is_mxnet else "NA"

    if isinstance(model, str):
        absmodel = os.path.abspath(os.path.expanduser(model))
        assert os.path.exists(absmodel) or os.path.exists(
            absmodel + ".pb"
        ), "invalid input path, the file does not exist!"

    # check if the input model is a neural_compressor model
    for name, nc_model in MODELS.items():
        if nc_model and isinstance(model, nc_model):
            return "pytorch" if name == "pytorch_ipex" or name == "pytorch_fx" else name
    if isinstance(model, TensorflowBaseModel):
        return "tensorflow"

    checker = [_is_tensorflow, _is_pytorch, _is_onnxruntime, _is_mxnet]
    for handler in checker:
        fwk_name = handler(model)
        if fwk_name != "NA":
            break
    assert (
        fwk_name != "NA"
    ), "Framework is not detected correctly from model format. This could be \
caused by unsupported model or inappropriate framework installation."

    return fwk_name


class Model(object):
    """A wrapper to construct a Neural Compressor Model."""

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
        conf = kwargs.pop("conf", "NA")
        if isinstance(root, BaseModel):
            if conf != "NA" and conf.framework is None:
                try:
                    conf.framework = list(MODELS.keys())[list(MODELS.values()).index(type(root))]
                except:
                    conf.framework = get_model_fwk_name(root._model)
                if hasattr(conf, "backend") and conf.backend == "ipex":
                    assert conf.framework == "pytorch_ipex", "Please wrap the model with correct Model class!"
                if hasattr(conf, "backend") and conf.backend == "itex":
                    if get_model_type(root.model) == "keras":
                        assert conf.framework == "keras", "Please wrap the model with KerasModel class!"
                    else:
                        assert conf.framework == "tensorflow", "Please wrap the model with TensorflowModel class!"
                        conf.framework = "tensorflow_itex"
                if getattr(conf, "approach", None) == "quant_aware_training":
                    assert conf.framework == "tensorflow_qat", "Please wrap the model with TensorflowQATModel class!"
            else:
                if "tensorflow" in conf.framework:
                    if getattr(root, "name", None) is None:
                        root.name = conf.model_name
                    if getattr(root, "output_tensor_names", None) is None:
                        root.output_tensor_names = conf.outputs
                    if getattr(root, "input_tensor_names", None) is None:
                        root.input_tensor_names = conf.inputs
                    if getattr(root, "workspace_path", None) is None:
                        root.workspace_path = options.workspace
            return root
        else:
            framework = get_model_fwk_name(root)
            if conf == "NA":
                if framework == "pytorch":
                    framework = "pytorch_fx"
                if "tensorflow" in framework:
                    if kwargs.get("approach", None) == "quant_aware_training":
                        return MODELS["tensorflow_qat"](root, **kwargs)
                    if "modelType" in kwargs:
                        model_type = kwargs["modelType"]
                    else:
                        model_type = get_model_type(root)
                    if model_type == "keras" and kwargs.get("backend", None) == "itex":
                        return MODELS["keras"](root, **kwargs)
                    elif model_type == "AutoTrackable":  # pragma: no cover
                        return MODELS[framework]("keras", root, **kwargs)
                    else:
                        return MODELS[framework](model_type, root, **kwargs)
                return MODELS[framework](root, **kwargs)
            else:
                conf.framework = framework
                if hasattr(conf, "backend") and conf.backend == "default":
                    if framework == "pytorch":
                        conf.framework = "pytorch_fx"
                elif hasattr(conf, "backend") and conf.backend == "ipex":
                    conf.framework = "pytorch_ipex"

                if "tensorflow" in conf.framework:
                    if getattr(conf, "approach", None) == "quant_aware_training":
                        model = MODELS["tensorflow_qat"](root, **kwargs)
                    else:
                        if "modelType" in kwargs:
                            model_type = kwargs["modelType"]
                        else:
                            model_type = get_model_type(root)
                        if model_type == "llm_saved_model":
                            return TensorflowLLMModel(root, **kwargs)
                        if hasattr(conf, "backend") and conf.backend == "itex":
                            if model_type == "keras":
                                conf.framework = "keras"
                                model = MODELS[conf.framework](root, **kwargs)
                            elif model_type == "AutoTrackable":  # pragma: no cover
                                # Workaround using HF model with ITEX
                                conf.framework = "tensorflow_itex"
                                model = MODELS[conf.framework]("keras", root, **kwargs)
                            else:
                                conf.framework = "tensorflow_itex"
                                model = MODELS[conf.framework](model_type, root, **kwargs)
                        else:
                            model = MODELS["tensorflow"](
                                "keras" if model_type == "AutoTrackable" else model_type, root, **kwargs
                            )
                else:
                    model = MODELS[conf.framework](root, **kwargs)
                if "tensorflow" in conf.framework and hasattr(conf, "model_name"):
                    model.name = conf.model_name
                    model.output_tensor_names = conf.outputs
                    model.input_tensor_names = conf.inputs
                    model.workspace_path = options.workspace
        return model
