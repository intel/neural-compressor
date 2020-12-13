#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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

from functools import reduce

def deep_get(dictionary, keys, default=None):

    """get the dot key's item in nested dict
       eg person = {'person':{'name':{'first':'John'}}}
       deep_get(person, "person.name.first") will output 'John'
       
       Args:
           dictionary (dict): The dict object to get keys
           keys (dict): The deep keys
           default (object): The return item if key not exists 
       Returns:
           item: the item of the deep dot keys
    """
    return reduce(lambda d, key: d.get(key, default) \
        if isinstance(d, dict) else default, keys.split("."), dictionary)

def deep_set(dictionary, keys, value):

    """set the dot key's item in nested dict
       eg person = {'person':{'name':{'first':'John'}}}
       deep_set(person, "person.sex", 'male') will output
       {'person': {'name': {'first': 'John'}, 'sex': 'male'}} 
       
       Args:
           dictionary (dict): The dict object to get keys
           keys (dict): The deep keys
           value (object): The value of the setting key
    """
    keys = keys.split('.')
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = value

class DotDict(dict):
    """access yaml using attributes instead of using the dictionary notation.

    Args:
        value (dict): The dict object to access.

    """

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __getitem__(self, key):
        value = self.get(key, None)
        return value

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        if isinstance(value, list) and len(value) == 1 and isinstance(
                value[0], dict):
            value = DotDict(value[0])
        if isinstance(value, list) and len(value) > 1 and all(isinstance(
                v, dict) for v in value):
            value = DotDict({k: v for d in value for k, v in d.items()})
        super(DotDict, self).__setitem__(key, value)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    __setattr__, __getattr__ = __setitem__, __getitem__

