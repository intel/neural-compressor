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

import jax.numpy as jnp
import keras


class WeightCopyInitializer:
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


def clone_model(model):
    """Returns clone of a model.

    Aims to be faster than copy.deepcopy() and keras.models.clone_model()
    Internally uses from_config() with overridden initializers for add_weight()
    that will create copies of original weights for a new model, instead of
    creating randomly generated values that will get replaced later.
    Then manually copies protos of original model since they are not stored in config.
    """
    original_model = model
    new_model = _build_new_model(original_model)
    _restore_protobufs(original_model, new_model)

    # Models built by subclassing keras.Model do not store their weights in config.
    # Additional loop over original model is required to build each layer and assign proper values to cloned model.
    if len(new_model.variables) != len(original_model.variables):
        _build_all_layers(original_model, new_model)
    return new_model


def _build_new_model(original_model):
    """Replaces original add_weight() (used internally by from_config())
    with faster version, which creates copies of original weights.

    Then calls from_config(), and restores original add_weight()
    """

    def _fast_add_weight(self, name=None, shape=None, initializer=None, **kwargs):
        """Replacement for add_weight() that uses WeightCopyInitializer."""
        var_path = f"{self.path}/{name}" if name else None
        orig_value = original_var_map.get(var_path) if var_path else None
        if orig_value is not None:
            return original_add_weight(
                self,
                name=name,
                shape=shape,
                initializer=WeightCopyInitializer(orig_value),
                **kwargs,
            )
        return original_add_weight(self, name=name, shape=shape, initializer=initializer, **kwargs)

    original_var_map = _build_var_map(original_model)
    original_add_weight = keras.layers.Layer.add_weight

    keras.layers.Layer.add_weight = _fast_add_weight
    try:
        model = original_model.__class__.from_config(original_model.get_config())
    finally:
        keras.layers.Layer.add_weight = original_add_weight
    return model


def _build_all_layers(original_model, model):
    """Reconstructs layers' weights by building each layer from its saved input shape
    and then assigning the original variable values."""

    original_build_shapes = [getattr(layer, "_build_shapes_dict", None) for layer in original_model._flatten_layers()]

    # Build new model's layers using shapes from original model
    for layer, build_shapes in zip(model._flatten_layers(), original_build_shapes):
        if build_shapes and not layer.built:
            input_shape = build_shapes.get("input_shape")
            if input_shape is not None and type(layer).build is not keras.layers.Layer.build:
                layer.build(input_shape)

    # Assign values to variables in newly built layers
    for target_var, source_var in zip(model.variables, original_model.variables):
        target_var.assign(jnp.array(source_var.value))


def _build_var_map(model):
    """Builds a path->value map for name-based copy in add_weight.

    Keys use the path with the top-level model name stripped, because from_config
    rebuilds layers under a different parent.
    (e.g. Sequential: original path is "sequential/dense/kernel", but during
    from_config self.path is just "dense", giving a lookup key "dense/kernel").
    """

    model_name_prefix = model.name + "/"
    original_var_map = {}
    for v in model.variables:
        original_var_map[v.path.removeprefix(model_name_prefix)] = v.value
    return original_var_map


def _restore_protobufs(orig_obj, new_obj, visited=None):
    """Restores non-serializable state that get_config() omits.

    Walk both model trees in parallel (via attribute names) to find objects
    like SentencePieceTokenizer whose proto is always saved as None in
    get_config() but must be present for inference.
    """

    # Return early if layer was visited before
    if visited is None:
        visited = set()
    key = (id(orig_obj), id(new_obj))
    if key in visited:
        return
    visited.add(key)

    # Replace new layer's protocol buffer with value from original layer
    if hasattr(orig_obj, "proto") and hasattr(new_obj, "set_proto"):
        new_obj.set_proto(orig_obj.proto)

    # Traverse the tree
    for attr_name in vars(orig_obj):
        orig_attr = getattr(orig_obj, attr_name, None)
        new_attr = getattr(new_obj, attr_name, None)
        if isinstance(orig_attr, keras.layers.Layer):
            _restore_protobufs(orig_attr, new_attr, visited)
