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

"""quantization auto-tuning config system.

This file specifies default config options for quantization auto-tuning tool.
User should not change values in this file. Instead, user should write a config
file (in yaml) and use cfg_from_file(yaml_file) to load it and override the default
options.
"""

import os
import os.path as osp
import inspect
import time
import sys
import numpy as np
import cpuinfo
from contextlib import contextmanager
from tempfile import NamedTemporaryFile


def print_info():
    print(inspect.stack()[1][1], ":", inspect.stack()[1][2], ":", inspect.stack()[1][3])


def caller_obj(obj_name):
    """Get the object of local variable, function or class in caller.

       Args:
           obj_name (string): The name of object in caller
    """
    for f in inspect.stack()[1:]:
        if obj_name in f[0].f_locals:
            return f[0].f_locals[obj_name]


def singleton(cls):
    instances = {}

    def _singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _singleton

class LazyImport(object):
    """Lazy import python module till use

       Args:
           module_name (string): The name of module imported later
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = __import__(self.module_name)

        return getattr(self.module, name)


class AverageMeter(object):
    """Computes the average value

       Args:
           skip (optional, integar): the skipped iterations in calculation
    """

    def __init__(self, skip=0):
        self.skip = skip
        self.reset()

    def reset(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, val):
        self.cnt += 1
        if self.cnt <= self.skip:
            self.avg = 0
            return
        self.sum += val
        self.avg = self.sum / (self.cnt - self.skip)


class Timeout(object):
    """Timeout class to check if spending time is beyond target.

       Args:
           seconds (optional, integar): the timeout value, 0 means early stop.
    """

    def __init__(self, seconds=0):
        self.seconds = seconds

    def __enter__(self):
        self.die_after = time.time() + self.seconds
        return self

    def __exit__(self, type, value, traceback):
        pass

    @property
    def timed_out(self):
        return time.time() > self.die_after


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif isinstance(obj, list):
        size += sum([get_size(item, seen) for item in obj])
    else:
        size += sum([get_size(v, seen) for v in dir(obj)])

    return size

def compute_sparsity(tensor, eps = 1e-10):
    mask = np.ones_like(tensor)
    tensor_size = tensor.size 
    dense_mask = tensor != 0
    dense_size = dense_mask.sum()
    return tensor_size, tensor_size - dense_size, dense_size

@contextmanager
def fault_tolerant_file(name):
    dirpath, filename = osp.split(name)
    with NamedTemporaryFile(dir=os.path.abspath(os.path.expanduser(dirpath)), \
                            delete=False, suffix='.tmp') as f:
        yield f
        f.flush()
        os.fsync(f)
        f.close()
        os.replace(f.name, name)

def equal_dicts(d1, d2, compare_keys=None, ignore_keys=None):
    """Check whether two dicts are same except for those ignored keys.
    """
    assert not (compare_keys and ignore_keys)
    if compare_keys == None and ignore_keys == None:
        return d1 == d2
    elif compare_keys == None and ignore_keys != None:
        return {k: v for k,v in d1.items() if k not in ignore_keys} == \
               {k: v for k,v in d2.items() if k not in ignore_keys}
    elif compare_keys != None and ignore_keys == None:
        return {k: v for k,v in d1.items() if k in compare_keys} == \
            {k: v for k,v in d2.items() if k in compare_keys}
    else:
        assert False


@singleton
class CpuInfo(object):
    def __init__(self):
        self._bf16 = False
        self._vnni = False
        cpuid = cpuinfo.CPUID()
        max_extension_support = cpuid.get_max_extension_support()
        if max_extension_support >= 7:
            ecx = cpuid._run_asm(
                b"\x31\xC9",             # xor ecx, ecx
                b"\xB8\x07\x00\x00\x00"  # mov eax, 7
                b"\x0f\xa2"              # cpuid
                b"\x89\xC8"              # mov ax, cx
                b"\xC3"                  # ret
            )
            self._vnni = bool(ecx & (1 << 11))
            eax = cpuid._run_asm(
                b"\xB9\x01\x00\x00\x00", # mov ecx, 1
                b"\xB8\x07\x00\x00\x00"  # mov eax, 7
                b"\x0f\xa2"              # cpuid
                b"\xC3"                  # ret
            )
            self._bf16 = bool(eax & (1 << 5))

    @property
    def bf16(self):
        return self._bf16

    @property
    def vnni(self):
        return self._vnni
