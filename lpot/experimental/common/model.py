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

class Model(object):
    """common Model just collect the infos to construct a Model
    """
    def __init__(self, root, **kwargs):
        """Wrap raw framework model format or path with specific infos

        Args:
            root:   raw model format. For Tensorflow model, could be path to frozen pb file, 
                    path to ckpt or savedmodel folder, loaded estimator/graph_def/graph/keras 
                    model object. For PyTorch model, it's torch.nn.model instance.
                    For MXNet model, it's mxnet.symbol.Symbol or gluon.HybirdBlock instance.
            kwargs: specific model format will rely on extra infomation to build the model 
                    a. estimator: need input_fn to initialize the Model, it will look like this 
                                  when initialize an estimator model:
                                  model = Model(estimator_object, input_fn=estimator_input_fn)
                   
        """
        self.root = root
        self.kwargs = kwargs
