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

from abc import abstractmethod

class BaseModel:
    ''' base class of all neural_compressor.model, will play graph role'''

    def __init__(self, model, **kwargs): 
        pass

    @property
    def model(self):
        ''' return model itself '''
        raise NotImplementedError

    @property
    def graph_info(self):
        ''' return {Node: Node_type} like {'conv0': 'conv2d'} '''
        raise NotImplementedError
 
    @abstractmethod
    def save(self, root, *args, **kwargs):
        ''' abstract method of model saving'''
        raise NotImplementedError

    @abstractmethod
    def framework(self):
        ''' abstract method of model framework'''
        raise NotImplementedError
    
