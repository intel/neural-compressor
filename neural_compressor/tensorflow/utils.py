# Copyright (c) 2023 Intel Corporation
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

import logging
import time
from functools import reduce
from typing import Callable, Dict

import numpy as np
import tensorflow as tf
from pkg_resources import parse_version

# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}


def version1_gte_version2(version1, version2):
    """Check if version1 is greater than or equal to version2."""
    return parse_version(version1) > parse_version(version2) or parse_version(version1) == parse_version(version2)


def register_algo(name):
    """Decorator function to register algorithms in the algos_mapping dictionary.

    Usage example:
        @register_algo(name=example_algo)
        def example_algo(model: torch.nn.Module, quant_config: RTNWeightQuantConfig) -> torch.nn.Module:
            ...
    Args:
        name (str): The name under which the algorithm function will be registered.
    Returns:
        decorator: The decorator function to be used with algorithm functions.
    """

    def decorator(algo_func):
        algos_mapping[name] = algo_func
        return algo_func

    return decorator


def deep_get(dictionary, keys, default=None):
    """Get the dot key's item in nested dict
       eg person = {'person':{'name':{'first':'John'}}}
       deep_get(person, "person.name.first") will output 'John'.

    Args:
        dictionary (dict): The dict object to get keys
        keys (dict): The deep keys
        default (object): The return item if key not exists
    Returns:
        item: the item of the deep dot keys
    """
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)


def dump_elapsed_time(customized_msg=""):
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): The parameter passed to decorator. Defaults to None.
    """

    def f(func):
        def fi(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            logging.getLogger("neural_compressor").info(
                "%s elapsed time: %s ms"
                % (customized_msg if customized_msg else func.__qualname__, round((end - start) * 1000, 2))
            )
            return res

        return fi

    return f
