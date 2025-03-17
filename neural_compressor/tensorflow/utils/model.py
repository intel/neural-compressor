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
"""The Model API for wrapping all TF model formats."""

import copy

from neural_compressor.common.utils import DEFAULT_WORKSPACE
from neural_compressor.tensorflow.utils.constants import TENSORFLOW_DEFAULT_CONFIG
from neural_compressor.tensorflow.utils.model_wrappers import BaseModel, KerasModel, TensorflowModel, get_tf_model_type
from neural_compressor.tensorflow.utils.utility import singleton


@singleton
class TensorflowGlobalConfig:
    """A global config class for setting framework specific information."""

    global_config = {
        "device": "cpu",
        "backend": "default",
        "approach": "post_training_static_quant",
        "random_seed": 1978,
        "workspace_path": DEFAULT_WORKSPACE,
        "format": "default",
        "use_bf16": True,
    }

    def reset_global_config(self):
        """Reset global config with default values."""
        self.global_config = copy.deepcopy(TENSORFLOW_DEFAULT_CONFIG)
        self.global_config["workspace_path"] = DEFAULT_WORKSPACE


TFConfig = TensorflowGlobalConfig()


class Model(object):  # pragma: no cover
    """A wrapper to construct a Neural Compressor TF Model."""

    def __new__(cls, root, **kwargs):
        """Create a new instance object of Model.

        Args:
            root (object): raw model format. For Tensorflow model, could be path to frozen pb file,
                path to ckpt or savedmodel folder, loaded estimator/graph_def/graph/keras model object.

        Returns:
            BaseModel: neural_compressor built-in model
        """
        from neural_compressor.tensorflow.utils import itex_installed

        if isinstance(root, BaseModel):
            framework_specific_info["backend"] = "itex"
            return root

        if kwargs.get("approach", None) == "quant_aware_training":
            model_type = "keras_qat"
        elif "modelType" in kwargs:
            model_type = kwargs["modelType"]
        else:
            model_type = get_tf_model_type(root)

        if model_type == "keras" and not itex_installed():
            model_type = "saved_model"

        # model = TensorflowSubclassedKerasModel(root)
        # framework_specific_info["backend"] = "itex"
        model = TensorflowModel(model_type, root, **kwargs)
        conf = kwargs.pop("conf", "NA")
        cls.set_framework_info(conf, model)

        return model

    @staticmethod
    def set_tf_config(conf, model):
        """Set tf global config with model information."""
        config = TFConfig.global_config
        framework = "keras" if isinstance(model, KerasModel) else "tensorflow"

        if conf.device:
            framework_specific_info["device"] = conf.device
        if conf.approach:
            framework_specific_info["approach"] = conf.approach
        if conf.random_seed:
            framework_specific_info["random_seed"] = conf.random_seed
        if conf.inputs:
            framework_specific_info["inputs"] = conf.inputs
        if conf.outputs:
            framework_specific_info["outputs"] = conf.outputs

        if framework == "keras":
            framework_specific_info["backend"] = "itex"
            return

        if conf.performance_only:
            framework_specific_info["performance_only"] = conf.performance_only

        if conf.workspace_path:
            framework_specific_info["workspace_path"] = conf.workspace_path
        if conf.recipes:
            framework_specific_info["recipes"] = conf.recipes

        framework_specific_info["use_bf16"] = conf.use_bf16 if conf.use_bf16 else False

        for item in ["scale_propagation_max_pooling", "scale_propagation_concat"]:
            if framework_specific_info["recipes"] and item not in framework_specific_info["recipes"]:
                framework_specific_info["recipes"].update({item: True})
