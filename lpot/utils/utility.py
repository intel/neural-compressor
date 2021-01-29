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

"""quantization auto-tuning config system.

This file specifies default config options for quantization auto-tuning tool.
User should not change values in this file. Instead, user should write a config
file (in yaml) and use cfg_from_file(yaml_file) to load it and override the default
options.
"""
import re
import ast
import os
import time
import sys
import logging
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
import os.path as osp
import cpuinfo
import numpy as np

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
    tf = LazyImport("tensorflow")
    from tensorflow.python.framework import tensor_util

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    # for Tensorflow case
    if isinstance(obj, tf.Graph):
        _graph_def = obj.as_graph_def()
        _graph_node = _graph_def.node if isinstance(_graph_def, tf.compat.v1.GraphDef) \
                      else _graph_def.graph_def.node
        for node in _graph_node:
            if node.op == "Const":
                input_tensor = node.attr["value"].tensor
                tensor_value = tensor_util.MakeNdarray(input_tensor)
                size += tensor_value.size
            else:
                size += get_size(node)
        return size
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif isinstance(obj, list):
        size += sum([get_size(item, seen) for item in obj])
    else:
        size += sum([get_size(v, seen) for v in dir(obj)])

    return size


def compute_sparsity(tensor, eps=1e-10):
    mask = np.ones_like(tensor)
    tensor_size = tensor.size
    dense_mask = tensor != 0
    dense_size = dense_mask.sum()
    return tensor_size, tensor_size - dense_size, dense_size


@contextmanager
def fault_tolerant_file(name):
    dirpath, filename = osp.split(name)
    with NamedTemporaryFile(dir=os.path.abspath(os.path.expanduser(dirpath)),
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
        return {k: v for k, v in d1.items() if k not in ignore_keys} == \
               {k: v for k, v in d2.items() if k not in ignore_keys}
    elif compare_keys != None and ignore_keys == None:
        return {k: v for k, v in d1.items() if k in compare_keys} == \
            {k: v for k, v in d2.items() if k in compare_keys}
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
                b"\xB9\x01\x00\x00\x00",  # mov ecx, 1
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


def dump_elapsed_time(customized_msg=""):
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): the parameter passed to decorator. Defaults to None.
    """
    def f(func):
        def fi(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            logging.getLogger().info('%s elapsed time: %s ms' %
                                     (customized_msg if customized_msg else func.__qualname__,
                                      round((end - start) * 1000, 2)))
            return res
        return fi
    return f


def combine_histogram(old_hist, arr):
    """ Collect layer histogram for arr and combine it with old histogram.
    """
    new_max = np.max(arr)
    new_min = np.min(arr)
    new_th = max(abs(new_min), abs(new_max))
    (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist
    if new_th <= old_th:
        hist, _ = np.histogram(arr,
                               bins=len(old_hist),
                               range=(-old_th, old_th))
        return (old_hist + hist, old_hist_edges, min(old_min, new_min),
                max(old_max, new_max), old_th)
    else:
        old_num_bins = len(old_hist)
        old_step = 2 * old_th / old_num_bins
        half_increased_bins = int((new_th - old_th) // old_step + 1)
        new_num_bins = half_increased_bins * 2 + old_num_bins
        new_th = half_increased_bins * old_step + old_th
        hist, hist_edges = np.histogram(arr,
                                        bins=new_num_bins,
                                        range=(-new_th, new_th))
        hist[half_increased_bins:new_num_bins - half_increased_bins] += old_hist
        return (hist, hist_edges, min(old_min, new_min), max(old_max,
                                                             new_max), new_th)


def get_tensor_histogram(tensor_data, bins=2048):
    max_val = np.max(tensor_data)
    min_val = np.min(tensor_data)
    th = max(abs(min_val), abs(max_val))

    hist, hist_edges = np.histogram(tensor_data, bins=2048, range=(-th, th))

    return (hist, hist_edges, min_val, max_val, th)


def get_all_fp32_data(data):
    return [float(i) for i in data.replace('[', ' ').replace(']', ' ').split(' ') if i.strip()]


def str2array(s):
    s = re.sub(r'\[ +', '[', s.strip())
    s = re.sub(r'[,\s]+', ', ', s)
    s = re.sub(r'\]\[', '], [', s)

    return np.array(ast.literal_eval(s))

def Dequantize(data, scale_info):
    original_shape = data.shape
    size = data.size
    new_data = data.reshape(size, )
    max_value = 255 if scale_info[0].find("Relu") != -1 else 127
    return np.array([float(i / max_value) for i in new_data]).reshape(original_shape)
class CaptureOutputToFile(object):
    def __init__(self, tmp_file_path, stream=sys.stderr):
        self.orig_stream_fileno = stream.fileno()
        self.tmp_file = open(tmp_file_path, 'w')

    def __enter__(self):
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        os.dup2(self.tmp_file.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.tmp_file.close()
