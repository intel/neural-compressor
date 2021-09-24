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

from neural_compressor.model.model import MODELS, get_model_fwk_name, get_model_type
from neural_compressor.utils.utility import LazyImport
onnx = LazyImport('onnx')
 
class Loader(object):
    def __call__(self, model):
        framework = get_model_fwk_name(model)
        if framework =='tensorflow':
            model_type = get_model_type(model)
            model =MODELS[framework](model_type, model)
        if framework =='onnxruntime':
            model = onnx.load(model)
            model =MODELS[framework](model)
        return model
