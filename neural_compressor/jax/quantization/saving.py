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

import keras
import keras.src.utils.dtype_utils as dtype_utils
from jax import numpy as jnp
from keras_hub.models import Gemma3CausalLM, Gemma3Tokenizer, ViTImageClassifier
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.task import Task
from keras_hub.src.utils.preset_utils import get_preset_saver

from neural_compressor.common.base_config import config_registry
from neural_compressor.jax.quantization.config import FRAMEWORK_NAME, BaseConfig, DynamicQuantConfig, StaticQuantConfig
from neural_compressor.jax.utils.utility import dtype_mapping, iterate_over_layers


def quant_config_to_json_object(quant_config: BaseConfig) -> dict:
    """Serialize a quant config to a JSON-compatible dict with class name.

    Args:
        quant_config: The quantization config object to serialize.

    Returns:
        A dict with 'quantization_type' and 'config' keys.
    """
    return {
        "quantization_type": quant_config.name,
        "config": quant_config.to_dict(),
    }


def quant_config_from_json_object(json_obj: dict) -> BaseConfig:
    """Deserialize a quant config from a JSON-compatible dict with class name.

    Args:
        json_obj: A dict with 'quantization_type' and 'config' keys.

    Returns:
        The instantiated quantization config object.

    Raises:
        ValueError: If the class name is unknown.
    """
    quant_type = json_obj.get("quantization_type")
    config_dict = json_obj.get("config", {})

    configs = config_registry.get_cls_configs()[FRAMEWORK_NAME]
    if quant_type not in configs:
        raise ValueError(f"Unknown config class: {quant_type}. Must be one of: {' or '.join(configs.keys())}.")

    config_class = configs[quant_type]
    return config_class.from_dict(config_dict)


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
        config["_quant_config"] = quant_config_to_json_object(self._quant_config)
        return config

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    @classmethod
    def from_config(cls, config):
        model = keras.saving.deserialize_keras_object(config["_wrapped_model"])
        quant_config_json = config.get("_quant_config")
        quant_config = quant_config_from_json_object(quant_config_json)
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
        config["_quant_config"] = quant_config_to_json_object(self._quant_config)
        return config

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    @classmethod
    def from_config(cls, config):
        model = keras.saving.deserialize_keras_object(config["_wrapped_model"])
        quant_config_json = config.get("_quant_config")
        quant_config = quant_config_from_json_object(quant_config_json)
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


def prepare_deserialized_quantized_model(
    model: keras.Model,
    quant_config: BaseConfig,
) -> KerasQuantizedModelWrapper:
    """Transform a loaded quantized model.

    It prepares the model for inference by preparing the quantized layers.
    Args:
        model: loaded base keras model
        quant_config: quantization configuration
    Returns:
        KerasQuantizedModelWrapper: the transformed quantized model
    """
    model_info = quant_config.get_model_info(model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)

    for _, value in configs_mapping.items():
        config = value
        break

    weight_dtype = dtype_mapping[config.weight_dtype]
    activation_dtype = dtype_mapping[config.activation_dtype]

    # Import here to avoid circular import with layers.py
    from neural_compressor.jax.quantization.layers_dynamic import dynamic_quant_mapping
    from neural_compressor.jax.quantization.layers_static import static_quant_mapping

    if isinstance(quant_config, StaticQuantConfig):
        layers_mapping = static_quant_mapping
    elif isinstance(quant_config, DynamicQuantConfig):
        layers_mapping = dynamic_quant_mapping
    else:
        raise ValueError(
            f"Unsupported quant_config type {type(quant_config).__name__}. "
            "Supported types are StaticQuantConfig and DynamicQuantConfig."
        )

    qmodel = model
    operations = [
        lambda layer: layers_mapping[layer.__class__].prepare(layer, weight_dtype, activation_dtype),
        lambda layer: layer.add_variables(),
        lambda layer: layer.post_quantization_cleanup(),
    ]

    iterate_over_layers(qmodel, operations, filter_function=lambda c: c in layers_mapping)
    if isinstance(qmodel, Backbone):
        qmodel = KerasQuantizedModelBackboneWrapper(qmodel, quant_config)
    else:
        wrapper_cls = WRAPPER_MAPPING.get(qmodel.__class__, KerasQuantizedModelWrapper)
        qmodel = wrapper_cls(qmodel, quant_config)
        if hasattr(qmodel, "backbone"):
            qmodel._tracker.unlock()
            qmodel.backbone = KerasQuantizedModelBackboneWrapper(qmodel.backbone, quant_config)
            qmodel._tracker.lock()

    return qmodel
