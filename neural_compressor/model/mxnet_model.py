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
from neural_compressor.conf import config as cfg
from neural_compressor.utils.utility import LazyImport
from neural_compressor.utils import logger
from .base_model import BaseModel
mx = LazyImport('mxnet')

class MXNetModel(BaseModel):
    """Build MXNetModel object

    Args:
        model (mxnet model): model path
    """

    def __init__(self, model, **kwargs):
        #(TODO) MXNet does not support recover model from tuning history currently
        self.q_config = None
        self._model = model
        self.calib_cache = {}

    def framework(self):
        return 'mxnet'

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def save(self, root=None):
        if root is None:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(os.path.dirname(root), exist_ok=True)

        if isinstance(self._model, mx.gluon.HybridBlock):
            self._model.export(root, remove_amp_cast=False)
            logger.info("Save quantized hybrid block model to {}.".format(root))
        else:
            symnet, args, auxs = self._model
            symnet = symnet.as_nd_ndarray()
            args = {k:v.as_nd_ndarray() for k, v in args.items()}
            auxs = {k:v.as_nd_ndarray() for k, v in auxs.items()}
            mx.model.save_checkpoint(root, 0, symnet, args, auxs, remove_amp_cast=False)
            logger.info("Save quantized symbol model to {}.".format(root))
