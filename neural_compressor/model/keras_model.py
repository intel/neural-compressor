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
"""Class for Keras model."""

import os
from abc import abstractmethod

from neural_compressor.model.base_model import BaseModel
from neural_compressor.utils.utility import LazyImport, compute_sparsity

tf = LazyImport("tensorflow")


class KerasModel(BaseModel):
    """Build Keras model."""

    def __init__(self, model, **kwargs):
        """Initialize a Keras model.

        Args:
            model (string or keras model object): model path or model object.
        """
        self.component = None
        self._model = model
        if not isinstance(model, tf.keras.Model):
            self._model_object = tf.keras.models.load_model(self._model)
        else:
            self._model_object = self._model
        self._q_config = None

    @property
    def q_config(self):
        """Return q_config."""
        return self._q_config

    @q_config.setter
    def q_config(self, q_config):
        """Set q_config."""
        self._q_config = q_config

    @property
    def model(self):
        """Return model itself."""
        return self._model_object

    @property
    def graph_info(self):
        """Return graph info."""
        # (TODO) get the graph info
        return None

    @abstractmethod
    def save(self, root, *args, **kwargs):
        """Save Keras model."""
        self._model_object.save(root)

    @abstractmethod
    def _export(
        self,
        save_path: str,
        conf,
    ):
        pass

    @abstractmethod
    def framework(self):
        """Return framework."""
        return "keras"

    def get_all_weight_names(self):
        """Get weight names of model.

        Returns:
            list: weight names list.
        """
        names = []
        for index, layer in enumerate(self.model.layers):
            if len(layer.weights):
                names.append(index)
        return names

    def report_sparsity(self):
        """Get sparsity of the model.

        Returns:
            df (DataFrame): DataFrame of sparsity of each weight.
            total_sparsity (float): total sparsity of model.
        """
        import numpy as np
        import pandas as pd
        import tensorflow as tf

        df = pd.DataFrame(columns=["Name", "Shape", "NNZ (dense)", "NNZ (sparse)", "Sparsity(%)"])
        pd.set_option("display.precision", 2)
        param_dims = [2, 4]
        params_size = 0
        sparse_params_size = 0
        for index, layer in enumerate(self.model.layers):
            if not len(layer.weights):
                continue
            # Extract just the actual parameter's name, which in this context we treat
            # as its "type"
            weights = layer.get_weights()[0]
            if weights.ndim in param_dims:
                param_size, sparse_param_size, dense_param_size = compute_sparsity(weights)
                density = dense_param_size / param_size
                params_size += param_size
                sparse_params_size += sparse_param_size
                df.loc[len(df.index)] = [
                    index,
                    list(weights.shape),
                    dense_param_size,
                    sparse_param_size,
                    (1 - density) * 100,
                ]

        total_sparsity = sparse_params_size / params_size * 100

        df.loc[len(df.index)] = [
            "Total sparsity:",
            "-",
            params_size,
            sparse_params_size,
            total_sparsity,
        ]

        return df, total_sparsity

    @property
    def input_node_names(self):
        """Return input node names."""
        return self.model.input_names

    @property
    def output_node_names(self):
        """Return output node names."""
        return self.model.output_names
