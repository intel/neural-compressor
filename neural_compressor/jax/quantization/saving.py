# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import jax
import keras
import keras.src.utils.dtype_utils as dtype_utils
import ml_dtypes
import numpy as np
from jax import numpy as jnp
from keras_hub.models import Gemma3CausalLM, Gemma3Tokenizer, ViTImageClassifier
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.task import Task
from keras_hub.src.utils.preset_utils import get_preset_saver

from ..quantization.config import BaseConfig


class SaveableLayerMixin:
    def save_own_variables(self, store):
        weight_dtype = getattr(self, "weight_dtype", None)
        for var in self._trainable_variables + self._non_trainable_variables:
            is_one_byte_format = dtype_utils.dtype_size(var.dtype) == 8
            if is_one_byte_format and var.dtype == weight_dtype:
                # Weights in 8 bit format will be stored as their int8 bit representation
                value_to_save = jnp.asarray(var.value).view(jnp.int8)
            else:
                value_to_save = jnp.asarray(var.value)
            store[var.name] = value_to_save

    def load_own_variables(self, store):
        weight_dtype = getattr(self, "weight_dtype", None)
        for var in self._trainable_variables + self._non_trainable_variables:
            value_to_load = store[var.name]
            if (value_to_load.dtype == jnp.int8) and (var.dtype == weight_dtype):
                # Quantized weights are saved in int8 format, need to convert back to original dtype
                value_to_load = value_to_load.view(var.dtype)
            var.assign(value_to_load)


@keras.saving.register_keras_serializable(package="INC", name=None)
class KerasQuantizedModelBackboneWrapper(Backbone):
    def __init__(self, model, quant_config: Optional[BaseConfig] = None):
        object.__setattr__(self, "_wrapped_model", model)
        object.__setattr__(
            self,
            "fields",
            {
                "_wrapped_model",
                "__class__",
                "__getattribute__",
                "__setattr__",
                "get_config",
                "save_to_preset",
                "_quant_config",
            },
        )
        # self.__class__ = model.__class__
        if quant_config is None:
            raise ValueError("quant_config must be provided for KerasQuantizedModelWrapper.")
        object.__setattr__(self, "_quant_config", quant_config)

    def __getattribute__(self, name):
        if name in object.__getattribute__(self, "fields"):
            return object.__getattribute__(self, name)
        return object.__getattribute__(self, "_wrapped_model").__getattribute__(name)

    def __setattr__(self, name, value):
        if name in object.__getattribute__(self, "fields"):
            return object.__setattr__(self, name, value)
        return object.__getattribute__(self, "_wrapped_model").__setattr__(name, value)

    def get_config(self):
        config = super().get_config()
        config["_wrapped_model"] = keras.saving.serialize_keras_object(self._wrapped_model)
        config["_quant_config"] = self._quant_config.to_dict()
        return config

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    @classmethod
    def from_config(cls, config):
        from neural_compressor.jax.algorithms.static import prepare_deserialized_quantized_model
        from neural_compressor.jax.quantization.config import StaticQuantConfig

        model = keras.saving.deserialize_keras_object(config["_wrapped_model"])
        quant_config = config.get("_quant_config", None)
        quant_config = StaticQuantConfig.from_dict(quant_config)
        qmodel = prepare_deserialized_quantized_model(model, quant_config)
        return qmodel

    def save_to_preset(self, preset_dir, max_shard_size=10):
        """Save backbone to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
            max_shard_size: `int` or `float`. Maximum size in GB for each
                sharded file. If `None`, no sharding will be done. Defaults to
                `10`.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_backbone(self, max_shard_size=max_shard_size)


@keras.saving.register_keras_serializable(package="INC", name=None)
class KerasQuantizedModelWrapper(Task):

    backbone_cls = KerasQuantizedModelBackboneWrapper

    def __init__(self, model, quant_config: Optional[BaseConfig] = None):
        object.__setattr__(self, "_wrapped_model", model)
        object.__setattr__(
            self,
            "fields",
            {
                "_wrapped_model",
                "__class__",
                "__getattribute__",
                "__setattr__",
                "get_config",
                "_quant_config",
                "save_to_preset",
            },
        )
        if quant_config is None:
            raise ValueError("quant_config must be provided for KerasQuantizedModelWrapper.")
        object.__setattr__(self, "_quant_config", quant_config)

    def __getattribute__(self, name):
        if name in object.__getattribute__(self, "fields"):
            return object.__getattribute__(self, name)
        return object.__getattribute__(self, "_wrapped_model").__getattribute__(name)

    def __setattr__(self, name, value):
        if name in object.__getattribute__(self, "fields"):
            return object.__setattr__(self, name, value)
        return object.__getattribute__(self, "_wrapped_model").__setattr__(name, value)

    def get_config(self):
        config = super().get_config()
        # Save backbone without wrapper for load/save_model <-> preset api compatibility
        backbone_wrapper = None
        if hasattr(self, "backbone"):
            if self.backbone.__class__ == KerasQuantizedModelBackboneWrapper:
                backbone_wrapper = self.backbone
                self.backbone = self.backbone._wrapped_model
        config["_wrapped_model"] = keras.saving.serialize_keras_object(self._wrapped_model)
        if backbone_wrapper is not None:
            self.backbone = backbone_wrapper
        config["_quant_config"] = self._quant_config.to_dict()
        return config

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    @classmethod
    def from_config(cls, config):
        from neural_compressor.jax.algorithms.static import prepare_deserialized_quantized_model
        from neural_compressor.jax.quantization.config import StaticQuantConfig

        model = keras.saving.deserialize_keras_object(config["_wrapped_model"])
        quant_config = config.get("_quant_config", None)
        quant_config = StaticQuantConfig.from_dict(quant_config)
        qmodel = prepare_deserialized_quantized_model(model, quant_config)

        return qmodel

    def save_to_preset(self, preset_dir, max_shard_size=10):
        """Save task to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
            max_shard_size: `int` or `float`. Maximum size in GB for each
                sharded file. If `None`, no sharding will be done. Defaults to
                `10`.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_task(self, max_shard_size=max_shard_size)


@keras.saving.register_keras_serializable(package="INC", name=None)
class KerasQuantizedGemmaWrapper(KerasQuantizedModelWrapper, Gemma3CausalLM):
    backbone_cls = KerasQuantizedModelBackboneWrapper


@keras.saving.register_keras_serializable(package="INC", name=None)
class KerasQuantizedViTWrapper(KerasQuantizedModelWrapper, ViTImageClassifier):
    backbone_cls = KerasQuantizedModelBackboneWrapper


@keras.saving.register_keras_serializable(package="INC", name=None)
class KerasQuantizedTokenizerWrapper(KerasQuantizedModelWrapper, Gemma3Tokenizer):
    backbone_cls = KerasQuantizedModelBackboneWrapper


WRAPPER_MAPPING = {
    Gemma3CausalLM: KerasQuantizedGemmaWrapper,
    ViTImageClassifier: KerasQuantizedViTWrapper,
    Gemma3Tokenizer: KerasQuantizedTokenizerWrapper,
}
