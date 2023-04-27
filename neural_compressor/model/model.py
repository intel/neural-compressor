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
import os
import importlib
import sys
from neural_compressor.config import options
from neural_compressor.utils.utility import LazyImport
from neural_compressor.utils import logger
from neural_compressor.model.base_model import BaseModel
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.model.mxnet_model import MXNetModel
from neural_compressor.model.keras_model import KerasModel
from neural_compressor.model.tensorflow_model import (
     TensorflowBaseModel,
     TensorflowModel,
     TensorflowQATModel,
     get_model_type
     )

TORCH = False
if importlib.util.find_spec('torch'):
    TORCH = True
    from neural_compressor.model.torch_model import *

torch = LazyImport('torch')
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')
onnx = LazyImport('onnx')
ort = LazyImport("onnxruntime")
yaml = LazyImport('yaml')
json = LazyImport('json')
np = LazyImport('numpy')

MODELS = {'tensorflow': TensorflowModel,
          'tensorflow_itex': TensorflowModel,
          'keras': KerasModel,
          'tensorflow_qat': TensorflowQATModel,
          'mxnet': MXNetModel,
          'pytorch': PyTorchModel if TORCH else None,
          'pytorch_ipex': IPEXModel if TORCH else None,
          'pytorch_fx': PyTorchFXModel if TORCH else None,
          'onnxruntime': ONNXModel,
          'onnxrt_qlinearops': ONNXModel,
          'onnxrt_qdq': ONNXModel,
          'onnxrt_integerops': ONNXModel
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
            if sys.version_info < (3,10) and \
                find_spec('onnxruntime_extensions'): # pragma: no cover
                from onnxruntime_extensions import get_library_path
                so.register_custom_ops_library(get_library_path())
            if isinstance(model, str):
                ort.InferenceSession(model, so, providers=['CPUExecutionProvider'])
            else:
                ort.InferenceSession(model.SerializeToString(), so, providers=['CPUExecutionProvider'])
        except Exception as e:  # pragma: no cover
            if 'Message onnx.ModelProto exceeds maximum protobuf size of 2GB' in str(e):
                logger.warning('Please use model path instead of onnx model object to quantize')
            else:
                logger.warning("If you use an onnx model with custom_ops to do quantiztaion, "
                    "please ensure onnxruntime-extensions is installed")
        else:
            return 'onnxruntime'
        return 'NA'

    def _is_pytorch(model):
        try:
            if isinstance(model, torch.nn.Module) or isinstance(
                    model, torch.fx.GraphModule) or isinstance(
                        model, torch.jit._script.RecursiveScriptModule):
                return 'pytorch'
            else:
                return 'NA'
        except:
            return 'NA'

    def _is_tensorflow(model):
        try:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            model_type = get_model_type(model)
        except:
            os.environ.pop("CUDA_DEVICE_ORDER")
            os.environ.pop("CUDA_VISIBLE_DEVICES")
            return 'NA'
        else:
            return 'tensorflow'

    def _is_mxnet(model):
        try:
            is_mxnet = isinstance(model, mx.gluon.HybridBlock) or \
                (hasattr(model, '__len__') and len(model) > 1 and \
                isinstance(model[0], mx.symbol.Symbol))
        except:
            return 'NA'
        else:
            return 'mxnet' if is_mxnet else 'NA'

    if isinstance(model, str):
        absmodel = os.path.abspath(os.path.expanduser(model))
        assert os.path.exists(absmodel) or os.path.exists(absmodel+'.pb'), \
            'invalid input path, the file does not exist!'

    #check if the input model is a neural_compressor model
    for name, nc_model in MODELS.items():
        if nc_model and isinstance(model, nc_model):
            return 'pytorch' if name == 'pytorch_ipex' or name == 'pytorch_fx' else name
    if isinstance(model, TensorflowBaseModel):
        return 'tensorflow'

    checker = [_is_tensorflow, _is_pytorch, _is_onnxruntime, _is_mxnet]
    for handler in checker:
        fwk_name = handler(model)
        if fwk_name != 'NA':
            break
    assert fwk_name != 'NA', 'Framework is not detected correctly from model format. This could be \
caused by unsupported model or inappropriate framework installation.'

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
        backend = kwargs.get("backend", "NA")
        if backend == "NA" or backend == "default":
            backend_tmp = get_model_fwk_name(root)
            if backend_tmp == "pytorch":
                backend = "pytorch_fx"
            else:
                backend = backend_tmp
        elif backend == "ipex":
            backend = "pytorch_ipex"

        if 'tensorflow' in backend or backend == 'keras':
            if kwargs.get("approach", None) == "quant_aware_training" or backend == 'tensorflow_qat':
                return MODELS['tensorflow_qat'](root, **kwargs)

            if 'modelType' in kwargs:
                model_type = kwargs['modelType']
            else:
                model_type = get_model_type(root)
            if backend == 'keras' and model_type == 'keras':
                return MODELS['keras'](root, **kwargs)
            model = MODELS['tensorflow'](model_type, root, **kwargs)
        else:
            model = MODELS[backend](root, **kwargs)
        return model


def wrap_model_from(user_model, conf):
    """Wrap the user model and dispatch to framework specific internal model object.

    Args:
       user_model: user are supported to set model from original framework model format
                   (eg, tensorflow frozen_pb or path to a saved model), but not recommended.
                   Best practice is to set from a initialized neural_compressor.common.Model.
                   If tensorflow model is used, model's inputs/outputs will be auto inferred,
                   but sometimes auto inferred inputs/outputs will not meet your requests,
                   set them manually in config yaml file. Another corner case is slim model
                   of tensorflow, be careful of the name of model configured in yaml file,
                   make sure the name is in supported slim model list.
        conf: the instance of PostTrainingQuantConfig or QuantizationAwareTrainingConfig or MixedPrecisionConfig.
    """
    if conf.framework is None:
        if isinstance(user_model, BaseModel):  # pragma: no cover
            conf.framework = list(MODELS.keys())[list(MODELS.values()).index(type(user_model))]
            if conf.backend == "ipex":
                assert conf.framework == "pytorch_ipex",\
                      "Please wrap the model with correct Model class!"
            if conf.backend == "itex":
                if get_model_type(user_model.model) == 'keras':
                    assert conf.framework == "keras",\
                          "Please wrap the model with KerasModel class!"
                else:
                    assert conf.framework == "pytorch_itex", \
                        "Please wrap the model with TensorflowModel class!"
        else:
            framework = get_model_fwk_name(user_model)
            if framework == "tensorflow":
                if get_model_type(user_model) == 'keras' and conf.backend == 'itex':
                    framework = 'keras'
            if framework == "pytorch":
                if conf.backend == "default":
                    framework = "pytorch_fx"
                elif conf.backend == "ipex":
                    framework = "pytorch_ipex"
            conf.framework = framework

    if not isinstance(user_model, BaseModel):
        logger.warning("Force convert framework model to neural_compressor model.")
        if "tensorflow" in conf.framework or conf.framework == "keras":
            model = Model(user_model, backend=conf.framework, device=conf.device)
        else:
            model = Model(user_model, backend=conf.framework)
    else:  # pragma: no cover
        if conf.framework == "pytorch_ipex":
            from neural_compressor.model.torch_model import IPEXModel
            assert type(user_model) == IPEXModel, \
                        "The backend is ipex, please wrap the model with IPEXModel class!"
        elif conf.framework == "pytorch_fx":
            from neural_compressor.model.torch_model import PyTorchFXModel
            assert type(user_model) == PyTorchFXModel, \
                        "The backend is default, please wrap the model with PyTorchFXModel class!"

        model = user_model

    if 'tensorflow' in conf.framework:
        model.name = conf.model_name
        model.output_tensor_names = conf.outputs
        model.input_tensor_names = conf.inputs
        model.workspace_path = options.workspace

    return model
