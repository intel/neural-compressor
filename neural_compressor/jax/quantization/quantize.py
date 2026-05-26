# Copyright (c) 2024-2026 Intel Corporation
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
"""Intel Neural Compressor JAX quantization base API."""

from typing import Callable, Dict, Tuple

import jax.numpy as jnp
import keras

from neural_compressor.common import logger
from neural_compressor.common.base_config import BaseConfig
from neural_compressor.common.utils import Mode, log_process
from neural_compressor.jax.utils import algos_mapping, check_backend


class QuickClone:
    """Class used for cloning keras models as fast as possible.

    Internally uses from_config() with overridden initializers for add_weight()
    that will create copies of original weights for a new model, instead of
    creating randomly generated values that will get replaced later.
    Then manually copies protos of original model since they are not stored in config.
    """

    def clone_model(self, model):
        """Returns clone of a model.

        Aims to be faster than copy.deepcopy() and keras.models.clone_model()
        """

        self.orig = model
        self.original_var_map = self._build_var_map()
        model = self._build_new_model()
        self._restore_protos(self.orig, model)

        # Models built by subclassing keras.Model do not store their weights in config.
        # Additional loop over original model is required to build each layer and assign proper values to cloned model.
        if len(model.variables) != len(self.orig.variables):
            self._build_all_layers(model)
        return model

    def _build_new_model(self):
        """Replaces original add_weight() (used internally by from_config())
        with faster version, which creates copies of original weights.

        Then calls from_config(), and restores original add_weight()
        """

        self._original_add_weight = keras.layers.Layer.add_weight
        keras.layers.Layer.add_weight = self._make_fast_add_weight()
        try:
            model = self.orig.__class__.from_config(self.orig.get_config())
        finally:
            keras.layers.Layer.add_weight = self._original_add_weight
        return model

    def _build_var_map(self):
        """Builds a path->value map for name-based copy in add_weight.

        Keys use the path with the top-level model name stripped, because from_config
        rebuilds layers under a different parent.
        (e.g. Sequential: original path is "sequential/dense/kernel", but during
        from_config self.path is just "dense", giving a lookup key "dense/kernel").
        """

        model_name_prefix = self.orig.name + "/"
        original_var_map = {}
        for v in self.orig.variables:
            original_var_map[v.path.removeprefix(model_name_prefix)] = v.value
        return original_var_map

    def _make_fast_add_weight(self):
        """Returns a replacement for add_weight() that uses WeightCopyInitializer."""

        original_add_weight = self._original_add_weight
        original_var_map = self.original_var_map
        Initializer = self._WeightCopyInitializer

        def _fast_add_weight(layer_self, name=None, shape=None, initializer=None, **kwargs):
            var_path = f"{layer_self.path}/{name}" if name else None
            orig_value = original_var_map.get(var_path) if var_path else None
            if orig_value is not None:
                return original_add_weight(
                    layer_self,
                    name=name,
                    shape=shape,
                    initializer=Initializer(orig_value),
                    **kwargs,
                )
            return original_add_weight(layer_self, name=name, shape=shape, initializer=initializer, **kwargs)

        return _fast_add_weight

    def _restore_protos(self, orig_obj, new_obj, visited=None):
        """Restores non-serializable state that get_config() omits.

        Walk both model trees in parallel (via attribute names) to find objects
        like SentencePieceTokenizer whose proto is always saved as None in
        get_config() but must be present for inference.
        """

        if visited is None:
            visited = set()
        key = (id(orig_obj), id(new_obj))
        if key in visited:
            return
        visited.add(key)
        if hasattr(orig_obj, "proto") and hasattr(new_obj, "set_proto"):
            new_obj.set_proto(orig_obj.proto)
        for attr_name in vars(orig_obj):
            orig_attr = getattr(orig_obj, attr_name, None)
            new_attr = getattr(new_obj, attr_name, None)
            if isinstance(orig_attr, keras.layers.Layer):
                self._restore_protos(orig_attr, new_attr, visited)

    def _build_all_layers(self, model):
        """Reconstructs layers' weights by building each layer from its saved input shape
        and then assigning the original variable values."""

        original_build_shapes = [getattr(layer, "_build_shapes_dict", None) for layer in self.orig._flatten_layers()]

        # Build new model's layers using shapes from original model
        for layer, build_shapes in zip(model._flatten_layers(), original_build_shapes):
            if build_shapes and not layer.built:
                input_shape = build_shapes.get("input_shape")
                if input_shape is not None and type(layer).build is not keras.layers.Layer.build:
                    layer.build(input_shape)

        # Assign values to variables in newly built layers
        for target_var, source_var in zip(model.variables, self.orig.variables):
            target_var.assign(jnp.array(source_var.value))

    class _WeightCopyInitializer:
        """Returns a copy of existing JAX array instead of a new randomly initialized."""

        def __init__(self, value):
            self._value = jnp.array(value)

        def __call__(self, shape, dtype=None):
            value = self._value
            if dtype is not None:
                value = jnp.array(value, dtype=dtype)
            if shape is not None and tuple(value.shape) != tuple(shape):
                value = jnp.reshape(value, shape)
            return value

        def get_config(self):
            return {}


def need_apply(configs_mapping: Dict[Tuple[str, callable], BaseConfig], algo_name):
    """Determine whether a quantization algorithm should be applied.

    Args:
        configs_mapping (Dict[Tuple[str, callable], BaseConfig]): Mapping of layer identifiers to configs.
        algo_name (str): Algorithm name to check.

    Returns:
        bool: True if any config matches the algorithm name.
    """
    return any(config.name == algo_name for config in configs_mapping.values())


# fmt: off
@log_process(mode=Mode.QUANTIZE)
def quantize_model(
    model: keras.Model,
    quant_config: BaseConfig,
    calib_function: Callable = None,
    inplace: bool = True
):
    """Return a quantized Keras model according to the given configuration.

    Args:
        model (keras.Model): FP32 Keras model to be quantized.
        quant_config (BaseConfig): Quantization configuration.
        calib_function (Callable, optional): Function used for model calibration, required for static quantization.
        inplace (bool): When True, the original model is modified in-place and should not be used afterward. False creates a copy of original model

    Returns:
        keras.Model: The quantized model.
    """
# fmt: on
    check_backend()
    if not inplace:
        model = QuickClone().clone_model(model)

    model_info = quant_config.get_model_info(model)
    configs_mapping = quant_config.to_config_mapping(model_info=model_info)
    for algo_name, algo_func in algos_mapping.items():
        if need_apply(configs_mapping, algo_name):
            logger.info(f"Start to apply {algo_name} on the model.")
            model = algo_func(model, configs_mapping, quant_config, calib_function)
    return model
