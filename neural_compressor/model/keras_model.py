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

import os
from abc import abstractmethod
from neural_compressor.model.base_model import BaseModel
from neural_compressor.utils.utility import LazyImport
tf = LazyImport('tensorflow')

class KerasModel(BaseModel):
    """Build KerasModel object

    Args:
        model (string or keras model object): model path or model object
        kwargs (dict): other required parameters

    """

    def __init__(self, model, **kwargs):
        self.component = None
        self._model = model
        if not isinstance(model, tf.keras.Model):
            self._model_object = tf.keras.models.load_model(self._model)
        else:
            self._model_object = self._model
        self._q_config = None

    @property
    def q_config(self):
        return self._q_config

    @q_config.setter
    def q_config(self, q_config):
        self._q_config = q_config

    @property
    def model(self):
        return self._model_object

    @property
    def graph_info(self):
        ''' return {Node: Node_type} like {'conv0': 'conv2d'} '''
        #(TODO) get the graph info
        return None

    @abstractmethod
    def save(self, root, *args, **kwargs):
        self._model_object.save(root)

    @abstractmethod
    def export(
        self,
        save_path: str,
        conf,
    ):
        pass

    @abstractmethod
    def framework(self):
        return 'keras'
