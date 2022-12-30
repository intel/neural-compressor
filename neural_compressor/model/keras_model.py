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
from neural_compressor.utils.utility import LazyImport
tf = LazyImport('tensorflow')

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
        #(TODO) get the graph info
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
        return 'keras'
