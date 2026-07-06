#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024-2026 Intel Corporation
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
"""The configs of algorithms for JAX."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import keras

from neural_compressor.common import logger
from neural_compressor.common.base_config import (
    DEFAULT_WHITE_LIST,
    OP_NAME_OR_MODULE_TYPE,
    BaseConfig,
    ComposableConfig,
    config_registry,
    register_config,
    register_supported_configs_for_fwk,
)
from neural_compressor.common.utils import DYNAMIC_QUANT, STATIC_QUANT
from neural_compressor.jax.utils.utility import dtype_mapping

FRAMEWORK_NAME = "jax"


def _layer_matches_filter(
    layer_id: str,
    class_name: str,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
) -> bool:
    """Check if a layer passes include/exclude filters.

    Each filter entry is matched against both the layer class name (exact match)
    and the layer path (regex search).

    Args:
        layer_id: Layer path or name identifier.
        class_name: Layer class name (e.g. "Dense").
        include: If set, layer must match at least one entry.
        exclude: If set, layer must NOT match any entry.

    Returns:
        True if the layer should be included in quantization.
    """

    def _matches(pattern: str) -> bool:
        if pattern == class_name:
            return True
        try:
            return re.search(pattern, layer_id) is not None
        except re.error as e:
            raise ValueError(f"Invalid regex pattern {pattern!r} in include/exclude filter.") from e

    if include is not None:
        if not any(_matches(p) for p in include):
            return False
    if exclude is not None:
        if any(_matches(p) for p in exclude):
            return False
    return True


def _resolve_include_patterns(white_list):
    """Translate a ``white_list`` into include patterns for ``get_model_info``.

    - ``DEFAULT_WHITE_LIST`` (``"*"``) -> ``None`` (target every supported layer).
    - ``EMPTY_WHITE_LIST`` (``None``) -> ``[]`` (target no layer).
    - a list -> class objects become their ``__name__``; strings pass through and
      are matched against both the layer class name and the layer path (regex).
    """
    if white_list == DEFAULT_WHITE_LIST:
        return None
    if white_list is None:
        return []
    return [entry if isinstance(entry, str) else entry.__name__ for entry in white_list]


def _serialize_white_list(white_list):
    """Convert a ``white_list`` into a JSON-serializable form (class -> name)."""
    if white_list == DEFAULT_WHITE_LIST or white_list is None:
        return white_list
    return [entry if isinstance(entry, str) else entry.__name__ for entry in white_list]


class OperatorConfig(NamedTuple):
    """Configuration pairing a quantization config with supported operators."""

    config: BaseConfig
    operators: List[str]


@register_config(framework_name=FRAMEWORK_NAME, algo_name=DYNAMIC_QUANT)
class DynamicQuantConfig(BaseConfig):
    """Config class for JAX Dynamic quantization.

    Dynamic quantization applies quantization to both weights and activations during runtime.
    This configuration supports various data types for flexible quantization strategies.

    Supported dtypes:
        - "fp8": 8-bit floating-point quantization (uses ml_dtypes.float8_e4m3 by default)
        - "int8": 8-bit integer quantization

    FP8 formats available:
        - "fp8_e4m3": 4 exponent bits, 3 mantissa bits (default for "fp8")
        - "fp8_e5m2": 5 exponent bits, 2 mantissa bits
    """

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "weight_dtype",
        "activation_dtype",
        "const_scale",
        "const_weight",
    ]

    name = DYNAMIC_QUANT

    def __add__(self, other: BaseConfig) -> BaseConfig:
        """Combine with another config, producing a ``ComposableQuantConfig``.

        Same-type configs merge their local configs (base behavior); different
        types compose so each sub-config keeps its own ``white_list`` / ``exclude``
        selection and its overridden ``to_config_mapping``.
        """
        if isinstance(other, type(self)):
            for op_name, config in other.local_config.items():
                self.set_local(op_name, config)
            return self
        return ComposableQuantConfig(configs=[self, other])

    def __init__(
        self,
        weight_dtype: str = "fp8_e4m3",
        activation_dtype: str = "fp8_e4m3",
        const_scale: bool = False,
        const_weight: bool = False,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        exclude: Optional[List[str]] = None,
    ):
        """Init Dynamic quantization config.

        Args:
            weight_dtype (str): Data type for weights, default is "fp8_e4m3".
            activation_dtype (str): Data type for activations, default is "fp8_e4m3".
            const_scale (bool): Whether to use a constant scale factor for quantization.
            const_weight (bool): Whether to use constant quantized weights.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): Layers to quantize. Each entry
                is a layer class (e.g. ``keras.layers.Dense``), a class name, or a path regex.
                Only matching supported layers are quantized. Defaults to ``"*"`` (all supported
                layers).
            exclude (Optional[List[str]]): List of layer class names or path patterns to exclude.
                Matching layers are skipped. Supports regular expression patterns.

        Returns:
            None: Initializes the configuration instance.
        """
        super().__init__(white_list=white_list)
        if not isinstance(weight_dtype, list):
            jnp_weight_dtype = dtype_mapping[weight_dtype]
            jnp_activation_dtype = dtype_mapping[activation_dtype]
            if (
                jnp.issubdtype(jnp_weight_dtype, jnp.floating) and jnp.issubdtype(jnp_activation_dtype, jnp.integer)
            ) or (jnp.issubdtype(jnp_weight_dtype, jnp.integer) and jnp.issubdtype(jnp_activation_dtype, jnp.floating)):
                raise ValueError("Mixed quantization with floating-point and integer dtypes is not supported.")
        self.weight_dtype = weight_dtype
        self.activation_dtype = activation_dtype
        self.const_scale = const_scale
        self.const_weight = const_weight
        self._exclude = exclude
        self._post_init()

    @property
    def exclude(self):
        """Get the exclude filter list."""
        return self._exclude

    @exclude.setter
    def exclude(self, value):
        """Set the exclude filter list."""
        self._exclude = value

    def to_config_mapping(self, config_list=None, model_info=None):
        """Map model ops to configs using the base local/global machinery.

        This reuses ``global_config`` and the op-type / op-name local configs the
        base ``_post_init`` builds from ``white_list``. Only the matching step is
        tweaked for Keras:

        * op-name patterns are matched against the hierarchical layer path with
          ``re.search`` instead of the base's anchored ``re.match``;
        * a string entry also matches a layer by its class name, so ``"Dense"``
          targets every Dense layer without needing the class object.
        """
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            global_config = config.global_config
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            for op_name, op_type in model_info:
                if global_config is not None:
                    config_mapping[(op_name, op_type)] = global_config
                if op_type in op_type_config_dict:
                    config_mapping[(op_name, op_type)] = op_type_config_dict[op_type]
                for op_name_pattern, pattern_config in op_name_config_dict.items():
                    if op_name_pattern == op_type or re.search(op_name_pattern, op_name):
                        config_mapping[(op_name, op_type)] = pattern_config
        return config_mapping

    def to_dict(self):
        """Serialize params plus the white_list / exclude selection filters."""
        result = self.get_params_dict()
        white_list = _serialize_white_list(self._white_list)
        if white_list != DEFAULT_WHITE_LIST and white_list is not None:
            result["white_list"] = white_list
        if self._exclude is not None:
            result["exclude"] = self._exclude
        return result

    def get_params_dict(self):
        """Get parameters dict, excluding internal and filter attributes."""
        result = dict()
        excluded = {"_global_config", "_local_config", "_white_list", "_exclude", "_is_initialized"}
        for param, value in self.__dict__.items():
            if param not in excluded:
                result[param] = value
        return result

    @classmethod
    def register_supported_configs(cls) -> None:
        """Register supported configs for dynamic quantization.

        Returns:
            None: Updates the class-level supported configuration list.
        """
        supported_configs = []
        dynamic_config = DynamicQuantConfig(
            weight_dtype=["fp8", "fp8_e4m3", "fp8_e5m2", "int8"],
            activation_dtype=["fp8", "fp8_e4m3", "fp8_e5m2", "int8"],
        )
        # Basic JAX operators for quantization
        operators = [keras.layers.Dense]
        supported_configs.append(OperatorConfig(config=dynamic_config, operators=operators))
        cls.supported_configs = supported_configs

    def get_model_info(self, model) -> List[Tuple[str, str]]:
        """Return all supported ops, minus any matched by ``exclude``.

        White-list selection is applied later by ``to_config_mapping`` through the
        base local/global configs. ``exclude`` is a JAX-side deny filter applied
        here so excluded layers never reach the mapping.

        Args:
            model (keras.Model): Keras model to inspect.

        Returns:
            List[Tuple[str, str]]: List of (layer path, layer class name) pairs.
        """

        from neural_compressor.jax.quantization.layers_dynamic import dynamic_quant_mapping

        supported = {layer_class.__name__ for layer_class in dynamic_quant_mapping.keys()}

        filter_result = []
        for layer in model._flatten_layers(recursive=True):
            class_name = layer.__class__.__name__
            if class_name not in supported:
                continue
            layer_id = layer.path if layer.path else layer.name
            if not _layer_matches_filter(layer_id, class_name, None, self.exclude):
                continue
            pair = (layer_id, class_name)
            if pair not in filter_result:
                filter_result.append(pair)

        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "DynamicQuantConfig", List["DynamicQuantConfig"]]:
        """Get a default config set for tuning.

        Returns:
            DynamicQuantConfig: Configuration to use for tuning.
        """
        return DynamicQuantConfig(
            weight_dtype=["fp8", "fp8_e4m3", "fp8_e5m2", "int8"],
            activation_dtype=["fp8", "fp8_e4m3", "fp8_e5m2", "int8"],
        )

    @classmethod
    def from_json_string(cls, json_string: str) -> "DynamicQuantConfig":
        """Create a DynamicQuantConfig from a JSON string.

        Args:
            json_string (str): JSON string describing the config.

        Returns:
            DynamicQuantConfig: Parsed configuration instance.
        """
        cfg = json.loads(json_string)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "DynamicQuantConfig":
        """Create a DynamicQuantConfig from a dictionary.

        Args:
            config_dict (Dict): Configuration fields.

        Returns:
            DynamicQuantConfig: Parsed configuration instance.
        """
        weight_dtype = config_dict.get("weight_dtype", "fp8_e4m3")
        activation_dtype = config_dict.get("activation_dtype", "fp8_e4m3")
        const_scale = config_dict.get("const_scale", False)
        const_weight = config_dict.get("const_weight", False)
        white_list = config_dict.get("white_list", DEFAULT_WHITE_LIST)
        exclude = config_dict.get("exclude", None)
        return cls(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            const_scale=const_scale,
            const_weight=const_weight,
            white_list=white_list,
            exclude=exclude,
        )


@register_config(framework_name=FRAMEWORK_NAME, algo_name=STATIC_QUANT)
class StaticQuantConfig(BaseConfig):
    """Config class for JAX Static quantization.

    Static quantization applies quantization to weights offline and activations during runtime
    using pre-computed calibration data. This configuration supports various data types for
    flexible quantization strategies.

    Supported dtypes:
        - "fp8": 8-bit floating-point quantization (uses ml_dtypes.float8_e4m3 by default)
        - "int8": 8-bit integer quantization

    FP8 formats available:
        - "fp8_e4m3": 4 exponent bits, 3 mantissa bits (default for "fp8")
        - "fp8_e5m2": 5 exponent bits, 2 mantissa bits
    """

    supported_configs: List[OperatorConfig] = []
    params_list = [
        "weight_dtype",
        "activation_dtype",
        "const_scale",
        "const_weight",
    ]

    name = STATIC_QUANT

    def __add__(self, other: BaseConfig) -> BaseConfig:
        """Combine with another config, producing a ``ComposableQuantConfig``.

        Same-type configs merge their local configs (base behavior); different
        types compose so each sub-config keeps its own ``white_list`` / ``exclude``
        selection and its overridden ``to_config_mapping``.
        """
        if isinstance(other, type(self)):
            for op_name, config in other.local_config.items():
                self.set_local(op_name, config)
            return self
        return ComposableQuantConfig(configs=[self, other])

    def __init__(
        self,
        weight_dtype: str = "fp8_e4m3",
        activation_dtype: str = "fp8_e4m3",
        const_scale: bool = False,
        const_weight: bool = False,
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
        exclude: Optional[List[str]] = None,
    ):
        """Init Static quantization config.

        Args:
            weight_dtype (str): Data type for weights, default is "fp8_e4m3".
            activation_dtype (str): Data type for activations, default is "fp8_e4m3".
            const_scale (bool): Whether to use a constant scale factor for quantization.
            const_weight (bool): Whether to use constant quantized weights.
            white_list (Optional[List[OP_NAME_OR_MODULE_TYPE]]): Layers to quantize. Each entry
                is a layer class (e.g. ``keras.layers.Dense``), a class name, or a path regex.
                Only matching supported layers are quantized. Defaults to ``"*"`` (all supported
                layers).
            exclude (Optional[List[str]]): List of layer class names or path patterns to exclude.
                Matching layers are skipped. Supports regular expression patterns.

        Returns:
            None: Initializes the configuration instance.
        """
        super().__init__(white_list=white_list)
        if not isinstance(weight_dtype, list):
            jnp_weight_dtype = dtype_mapping[weight_dtype]
            jnp_activation_dtype = dtype_mapping[activation_dtype]
            if (
                jnp.issubdtype(jnp_weight_dtype, jnp.floating) and jnp.issubdtype(jnp_activation_dtype, jnp.integer)
            ) or (jnp.issubdtype(jnp_weight_dtype, jnp.integer) and jnp.issubdtype(jnp_activation_dtype, jnp.floating)):
                raise ValueError("Mixed quantization with floating-point and integer dtypes is not supported.")

        self.weight_dtype = weight_dtype
        self.activation_dtype = activation_dtype
        self.const_scale = const_scale
        self.const_weight = const_weight
        self._exclude = exclude
        self._post_init()

    @property
    def exclude(self):
        """Get the exclude filter list."""
        return self._exclude

    @exclude.setter
    def exclude(self, value):
        """Set the exclude filter list."""
        self._exclude = value

    def to_config_mapping(self, config_list=None, model_info=None):
        """Map model ops to configs using the base local/global machinery.

        This reuses ``global_config`` and the op-type / op-name local configs the
        base ``_post_init`` builds from ``white_list``. Only the matching step is
        tweaked for Keras:

        * op-name patterns are matched against the hierarchical layer path with
          ``re.search`` instead of the base's anchored ``re.match``;
        * a string entry also matches a layer by its class name, so ``"Dense"``
          targets every Dense layer without needing the class object.
        """
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            global_config = config.global_config
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            for op_name, op_type in model_info:
                if global_config is not None:
                    config_mapping[(op_name, op_type)] = global_config
                if op_type in op_type_config_dict:
                    config_mapping[(op_name, op_type)] = op_type_config_dict[op_type]
                for op_name_pattern, pattern_config in op_name_config_dict.items():
                    if op_name_pattern == op_type or re.search(op_name_pattern, op_name):
                        config_mapping[(op_name, op_type)] = pattern_config
        return config_mapping

    def to_dict(self):
        """Serialize params plus the white_list / exclude selection filters."""
        result = self.get_params_dict()
        white_list = _serialize_white_list(self._white_list)
        if white_list != DEFAULT_WHITE_LIST and white_list is not None:
            result["white_list"] = white_list
        if self._exclude is not None:
            result["exclude"] = self._exclude
        return result

    def get_params_dict(self):
        """Get parameters dict, excluding internal and filter attributes."""
        result = dict()
        excluded = {"_global_config", "_local_config", "_white_list", "_exclude", "_is_initialized"}
        for param, value in self.__dict__.items():
            if param not in excluded:
                result[param] = value
        return result

    @classmethod
    def register_supported_configs(cls) -> None:
        """Register supported configs for static quantization.

        Returns:
            None: Updates the class-level supported configuration list.
        """
        supported_configs = []
        static_config = StaticQuantConfig(
            weight_dtype=["fp8", "fp8_e4m3", "fp8_e5m2", "int8"],
            activation_dtype=["fp8", "fp8_e4m3", "fp8_e5m2", "int8"],
        )
        # Basic JAX operators for quantization
        operators = [keras.layers.Dense]
        supported_configs.append(OperatorConfig(config=static_config, operators=operators))
        cls.supported_configs = supported_configs

    def get_model_info(self, model) -> List[Tuple[str, str]]:
        """Return all supported ops, minus any matched by ``exclude``.

        White-list selection is applied later by ``to_config_mapping`` through the
        base local/global configs. ``exclude`` is a JAX-side deny filter applied
        here so excluded layers never reach the mapping.

        Args:
            model (keras.Model): Keras model to inspect.

        Returns:
            List[Tuple[str, str]]: List of (layer path, layer class name) pairs.
        """
        from neural_compressor.jax.quantization.layers_static import static_quant_mapping

        supported = {layer_class.__name__ for layer_class in static_quant_mapping.keys()}

        filter_result = []
        for layer in model._flatten_layers(recursive=True):
            class_name = layer.__class__.__name__
            if class_name not in supported:
                continue
            layer_id = layer.path if layer.path else layer.name
            if not _layer_matches_filter(layer_id, class_name, None, self.exclude):
                continue
            pair = (layer_id, class_name)
            if pair not in filter_result:
                filter_result.append(pair)

        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "StaticQuantConfig", List["StaticQuantConfig"]]:
        """Get a default config set for tuning.

        Returns:
            StaticQuantConfig: Configuration to use for tuning.
        """
        return StaticQuantConfig(
            weight_dtype=["fp8_e4m3", "fp8_e5m2", "int8"],
            activation_dtype=["fp8_e4m3", "fp8_e5m2", "int8"],
        )

    @classmethod
    def from_json_string(cls, json_string: str) -> "StaticQuantConfig":
        """Create a StaticQuantConfig from a JSON string.

        Args:
            json_string (str): JSON string describing the config.

        Returns:
            StaticQuantConfig: Parsed configuration instance.
        """
        cfg = json.loads(json_string)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "StaticQuantConfig":
        """Create a StaticQuantConfig from a dictionary.

        Args:
            config_dict (Dict): Configuration fields.

        Returns:
            StaticQuantConfig: Parsed configuration instance.
        """
        weight_dtype = config_dict.get("weight_dtype", "fp8_e5m2")
        activation_dtype = config_dict.get("activation_dtype", "fp8_e5m2")
        const_scale = config_dict.get("const_scale", False)
        const_weight = config_dict.get("const_weight", False)
        white_list = config_dict.get("white_list", DEFAULT_WHITE_LIST)
        exclude = config_dict.get("exclude", None)
        return cls(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            const_scale=const_scale,
            const_weight=const_weight,
            white_list=white_list,
            exclude=exclude,
        )


class ComposableQuantConfig(ComposableConfig):
    """JAX composable config that reuses each sub-config's ``to_config_mapping``.

    The base :class:`ComposableConfig` keys ``model_info`` by config *name*, which
    collides when the same config type appears more than once (e.g. static +
    dynamic + static) and it ignores the per-config ``global_config``. This
    subclass keeps each sub-config's ``model_info`` positionally aligned and
    delegates op selection to the sub-config's own (JAX-tweaked)
    ``to_config_mapping``, then merges results with last-config-wins on overlap.
    """

    def get_model_info(self, model, *args, **kwargs) -> List[List[Tuple[str, str]]]:
        """Return each sub-config's model info, aligned with ``config_list``.

        A list (not a name-keyed dict) is used so repeated config types keep their
        own candidate ops instead of overwriting each other.
        """
        return [sub_config.get_model_info(model, *args, **kwargs) for sub_config in self.config_list]

    def to_config_mapping(self, config_list=None, model_info=None) -> OrderedDict:
        """Merge each sub-config's mapping, with later configs overriding earlier ones."""
        config_mapping = OrderedDict()
        for sub_config, sub_model_info in zip(self.config_list, model_info):
            sub_mapping = sub_config.to_config_mapping(model_info=sub_model_info)
            for key, cfg in sub_mapping.items():
                if key in config_mapping:
                    logger.debug(f"Layer {key} quant config override from {config_mapping[key]} to {cfg}")
                config_mapping[key] = cfg
        return config_mapping


register_supported_configs_for_fwk(fwk_name=FRAMEWORK_NAME)


def get_all_registered_configs() -> Dict[str, BaseConfig]:
    """Get all registered configs for JAX framework.

    Returns:
        Dict[str, BaseConfig]: Mapping of config names to config classes.
    """
    registered_configs = config_registry.get_cls_configs()
    return registered_configs.get(FRAMEWORK_NAME, {})


def get_default_dynamic_config() -> DynamicQuantConfig:
    """Generate the default Dynamic quantization config.

    Returns:
        DynamicQuantConfig: The default JAX Dynamic quantization config.
    """
    return DynamicQuantConfig()


def get_default_static_config() -> StaticQuantConfig:
    """Generate the default Static quantization config.

    Returns:
        StaticQuantConfig: The default JAX Static quantization config.
    """
    return StaticQuantConfig()
