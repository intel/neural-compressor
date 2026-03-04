"""Serialization helpers for JAX quantized Keras models."""

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

from importlib import metadata as importlib_metadata
from typing import Optional

import keras
import keras.src.utils.dtype_utils as dtype_utils
from jax import numpy as jnp
from keras_hub.models import Gemma3CausalLM, Gemma3Tokenizer, ViTImageClassifier
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.task import Task
from keras_hub.src.utils.preset_utils import get_preset_saver

from neural_compressor.common import logger
from neural_compressor.common.base_config import config_registry
from neural_compressor.jax.quantization.config import FRAMEWORK_NAME, BaseConfig, DynamicQuantConfig, StaticQuantConfig
from neural_compressor.jax.utils.utility import dtype_mapping, iterate_over_layers


def quant_config_to_json_object(quant_config: BaseConfig) -> dict:
    """Serialize a quant config to a JSON-compatible dict with class name.

    Args:
        quant_config (BaseConfig): The quantization config object to serialize.

    Returns:
        dict: A dict with 'quantization_type' and 'config' keys.
    """
    return {
        "quantization_type": quant_config.name,
        "config": quant_config.to_dict(),
    }


def quant_config_from_json_object(json_obj: dict) -> BaseConfig:
    """Deserialize a quant config from a JSON-compatible dict with class name.

    Args:
        json_obj (dict): A dict with 'quantization_type' and 'config' keys.

    Returns:
        BaseConfig: The instantiated quantization config object.

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


class VersionManager:
    """Handle version metadata for serialized quantized models."""

    _MODULES = ["neural_compressor_jax", "keras", "keras_hub"]

    @classmethod
    def add_versions(cls, config):
        """Insert package versions into the serialized config.

        Args:
            config (dict): Configuration dictionary to update in-place.

        Returns:
            None: Updates the config dictionary in-place.
        """
        config["_versions"] = {}
        for package in cls._MODULES:
            config["_versions"][package] = importlib_metadata.version(package)

    @classmethod
    def check_versions_mismatch(cls, config):
        """Check for version mismatches between saved and current packages.

        Args:
            config (dict): Configuration dictionary that may include version metadata.

        Returns:
            None: Logs warnings if mismatches are found.
        """
        versions = config.get("_versions")
        if versions is None:
            logger.error(
                "No version information found in the saved model. Please save model with newer version of neural_compressor."
            )
            return
        for package, version_in_config in versions.items():
            current_version = importlib_metadata.version(package)
            if version_in_config != current_version:
                logger.warning(
                    f"{package}: version mismatch. Saved model: {version_in_config}, current version: {current_version}. "
                    f"This could cause unexpected behavior."
                )


class SaveableLayerMixin:
    """Mixin for saving and loading quantized layer variables."""

    def save_own_variables(self, store):
        """Save layer variables into the provided store.

        Args:
            store (dict): Mutable mapping to receive serialized variables.

        Returns:
            None: Updates the store mapping with serialized variables.
        """
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
        """Load layer variables from the provided store.

        Args:
            store (dict): Mapping containing serialized variables.

        Returns:
            None: Loads variables into the layer.
        """
        weight_dtype = getattr(self, "weight_dtype", None)
        for var in self._trainable_variables + self._non_trainable_variables:
            value_to_load = store[var.name]
            if (value_to_load.dtype == jnp.int8) and (var.dtype == weight_dtype):
                # Quantized weights are saved in int8 format, need to convert back to original dtype
                value_to_load = value_to_load.view(var.dtype)
            var.assign(value_to_load)


@keras.saving.register_keras_serializable(package="INC", name=None)
class KerasQuantizedModelBackboneWrapper(Backbone):
    """Wrapper that preserves quantization config when saving Keras backbones."""

    def __init__(self, model, quant_config: Optional[BaseConfig] = None):
        """Initialize the wrapper around a backbone model.

        Args:
            model (keras.Model): Backbone model to wrap.
            quant_config (Optional[BaseConfig]): Quantization configuration.

        Returns:
            None: Initializes the wrapper.
        """
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
        """Delegate attribute access to the wrapped model.

        Args:
            name (str): Attribute name to access.

        Returns:
            Any: Attribute value from the wrapper or wrapped model.
        """
        if name in object.__getattribute__(self, "fields"):
            return object.__getattribute__(self, name)
        return object.__getattribute__(self, "_wrapped_model").__getattribute__(name)

    def __setattr__(self, name, value):
        """Delegate attribute updates to the wrapped model.

        Args:
            name (str): Attribute name to update.
            value (Any): Value to assign.

        Returns:
            None: Updates the attribute on the wrapper or wrapped model.
        """
        if name in object.__getattribute__(self, "fields"):
            return object.__setattr__(self, name, value)
        return object.__getattribute__(self, "_wrapped_model").__setattr__(name, value)

    def get_config(self):
        """Serialize the wrapper configuration for Keras saving.

        Returns:
            dict: Serialized configuration for the wrapper.
        """
        config = super().get_config()
        config["_quant_config"] = quant_config_to_json_object(self._quant_config)
        config["_wrapped_model"] = keras.saving.serialize_keras_object(self._wrapped_model)
        return config

    def __new__(cls, *args, **kwargs):
        """Bypass BaseModel __new__ to allow manual initialization.

        Args:
            *args: Positional arguments for object creation.
            **kwargs: Keyword arguments for object creation.

        Returns:
            KerasQuantizedModelBackboneWrapper: New wrapper instance.
        """
        return object.__new__(cls)

    @classmethod
    def from_config(cls, config):
        """Recreate a wrapper from a serialized config dictionary.

        Args:
            config (dict): Serialized configuration dictionary.

        Returns:
            KerasQuantizedModelWrapper: Reconstructed quantized model wrapper.
        """
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

        Returns:
            None: Writes the preset files to disk.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_backbone(self, max_shard_size=max_shard_size)


@keras.saving.register_keras_serializable(package="INC", name=None)
class KerasQuantizedModelWrapper(Task):
    """Wrapper that preserves quantization config for Keras tasks."""

    backbone_cls = KerasQuantizedModelBackboneWrapper

    def __init__(self, model, quant_config: Optional[BaseConfig] = None):
        """Initialize the wrapper around a task model.

        Args:
            model (keras.Model): Task model to wrap.
            quant_config (Optional[BaseConfig]): Quantization configuration.

        Returns:
            None: Initializes the wrapper.
        """
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
        """Delegate attribute access to the wrapped model.

        Args:
            name (str): Attribute name to access.

        Returns:
            Any: Attribute value from the wrapper or wrapped model.
        """
        if name in object.__getattribute__(self, "fields"):
            return object.__getattribute__(self, name)
        return object.__getattribute__(self, "_wrapped_model").__getattribute__(name)

    def __setattr__(self, name, value):
        """Delegate attribute updates to the wrapped model.

        Args:
            name (str): Attribute name to update.
            value (Any): Value to assign.

        Returns:
            None: Updates the attribute on the wrapper or wrapped model.
        """
        if name in object.__getattribute__(self, "fields"):
            return object.__setattr__(self, name, value)
        return object.__getattribute__(self, "_wrapped_model").__setattr__(name, value)

    def get_config(self):
        """Serialize the wrapper configuration for Keras saving.

        Returns:
            dict: Serialized configuration for the wrapper.
        """
        config = super().get_config()
        VersionManager.add_versions(config)
        config["_quant_config"] = quant_config_to_json_object(self._quant_config)
        # Save backbone without wrapper for load/save_model <-> preset api compatibility
        backbone_wrapper = None
        if hasattr(self, "backbone"):
            if self.backbone.__class__ == KerasQuantizedModelBackboneWrapper:
                backbone_wrapper = self.backbone
                self.backbone = self.backbone._wrapped_model
        config["_wrapped_model"] = keras.saving.serialize_keras_object(self._wrapped_model)
        if backbone_wrapper is not None:
            self.backbone = backbone_wrapper
        return config

    def __new__(cls, *args, **kwargs):
        """Bypass BaseModel __new__ to allow manual initialization.

        Args:
            *args: Positional arguments for object creation.
            **kwargs: Keyword arguments for object creation.

        Returns:
            KerasQuantizedModelWrapper: New wrapper instance.
        """
        return object.__new__(cls)

    @classmethod
    def from_config(cls, config):
        """Recreate a wrapper from a serialized config dictionary.

        Args:
            config (dict): Serialized configuration dictionary.

        Returns:
            KerasQuantizedModelWrapper: Reconstructed quantized model wrapper.
        """
        VersionManager.check_versions_mismatch(config)
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

        Returns:
            None: Writes the preset files to disk.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_task(self, max_shard_size=max_shard_size)


@keras.saving.register_keras_serializable(package="INC", name=None)
class KerasQuantizedGemmaWrapper(KerasQuantizedModelWrapper, Gemma3CausalLM):
    """Quantized wrapper for Gemma3CausalLM models."""

    backbone_cls = KerasQuantizedModelBackboneWrapper


@keras.saving.register_keras_serializable(package="INC", name=None)
class KerasQuantizedViTWrapper(KerasQuantizedModelWrapper, ViTImageClassifier):
    """Quantized wrapper for ViTImageClassifier models."""

    backbone_cls = KerasQuantizedModelBackboneWrapper


@keras.saving.register_keras_serializable(package="INC", name=None)
class KerasQuantizedTokenizerWrapper(KerasQuantizedModelWrapper, Gemma3Tokenizer):
    """Quantized wrapper for Gemma3Tokenizer models."""

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
        model (keras.Model): Loaded base keras model.
        quant_config (BaseConfig): Quantization configuration.
    Returns:
        KerasQuantizedModelWrapper: The transformed quantized model wrapper.
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
