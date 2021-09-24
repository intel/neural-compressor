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
import pickle
import logging
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
import os.path as osp
import threading, _thread
import cpuinfo
import numpy as np
from neural_compressor.utils import logger
import prettytable as pt

def singleton(cls):
    instances = {}

    def _singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _singleton

def set_backend(backend):
    global __BACKEND
    __BACKEND = backend

def get_backend():
    global __BACKEND
    return __BACKEND

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
            # __import__ returns top level module
            top_level_module = __import__(self.module_name)
            if len(self.module_name.split('.')) == 1:
                self.module = top_level_module
            else:
                # for cases that input name is foo.bar.module
                module_list = self.module_name.split('.')
                temp_module = top_level_module
                for temp_name in module_list[1:]:
                    temp_module = getattr(temp_module, temp_name)
                self.module = temp_module

        return getattr(self.module, name)

@contextmanager
def time_limit(seconds):
    if seconds == 0:
        seconds = threading.TIMEOUT_MAX
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()

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


def get_tuning_history(tuning_history_path):
    """
    :params tuning_history_path: need user to assign
    """
    with open(tuning_history_path, 'rb') as f:
        strategy_object = pickle.load(f)
    tuning_history = strategy_object.tuning_history
    return tuning_history


def recover(fp32_model, tuning_history_path, num, **kwargs):
    """offline recover tuned model.

    :params fp32_model: input model path
    :params tuning_history_path: need user to assign
    :params num: tune index
    """
    tuning_history = get_tuning_history(tuning_history_path)
    target_history = tuning_history[0]['history']
    q_config = target_history[num]['q_config']
    framework = tuning_history[0]['cfg']['model']['framework']

    if 'pytorch' in framework:
        from neural_compressor.utils.pytorch import load
        tune_index_qmodel = load(model=fp32_model, history_cfg=q_config, **kwargs)
        return tune_index_qmodel

    from neural_compressor.adaptor import FRAMEWORKS
    adaptor = FRAMEWORKS[framework](q_config['framework_specific_info'])
    if 'onnxrt' in framework:
        from neural_compressor.experimental import common
        ox_fp32_model = common.Model(fp32_model)
        tune_index_qmodel = adaptor.recover(ox_fp32_model, q_config)
        return tune_index_qmodel
    elif 'tensorflow' in framework:
        from neural_compressor.experimental import common
        tf_fp32_model = common.Model(fp32_model)
        tune_index_qmodel = adaptor.recover_tuned_model(tf_fp32_model, q_config)
        return tune_index_qmodel
    elif 'mxnet' in framework:
        from neural_compressor.experimental import common
        mx_fp32_model = common.Model(fp32_model)
        tune_index_qmodel = adaptor.recover_tuned_model(mx_fp32_model, q_config)
        return tune_index_qmodel


def str2array(s):
    s = re.sub(r'\[ +', '[', s.strip())
    s = re.sub(r'[,\s]+', ', ', s)
    s = re.sub(r'\]\[', '], [', s)

    return np.array(ast.literal_eval(s))


def DequantizeWeight(weight_tensor, min_filter_tensor, max_filter_tensor):
    weight_channel = weight_tensor.shape[-1]

    for i in range(weight_channel):
        data = weight_tensor[:,:,:,i]
        new_data = data.reshape(data.size,)
        if len(min_filter_tensor) == weight_channel:
            # per_channel mode
            for j in range((data.size)):
                new_data[j] = \
                        float(new_data[j] *(max_filter_tensor[i] - min_filter_tensor[i])/ 127.0)
        else:
            # per_tensor mode
            for j in range((data.size)):
                new_data[j] = \
                        float(new_data[j] *(max_filter_tensor[0] - min_filter_tensor[0])/ 127.0)


def Dequantize(data, scale_info):
    original_shape = data.shape
    size = data.size
    new_data = data.reshape(size, )
    max_value = 255 if scale_info[0].find("Relu") != -1 else 127
    return np.array([float(i *(scale_info[2] - scale_info[1])/ max_value) for i in new_data]\
            ).reshape(original_shape)


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


class OpPrecisionStatistics():
    def __init__(self, data, output_handle=logger.info):
        self.field_names = ["Op Type", "Total", "INT8", "BF16", "FP32"]
        self.data = data
        self.output_handle = output_handle
        self.tb = pt.PrettyTable(min_table_width=40)

    def print_stat(self):
        valid_field_names = []
        for index, value in enumerate(self.field_names):
            if index < 2:
                valid_field_names.append(value)
                continue

            if any(i[index] for i in self.data):
                valid_field_names.append(value)
        self.tb.field_names = valid_field_names
        for i in self.data:
            tmp_data = []
            for index, value in enumerate(i):
                if self.field_names[index] in valid_field_names:
                    tmp_data.append(value)
            if any(tmp_data[1:]):
                self.tb.add_row(tmp_data)
        lines = self.tb.get_string().split('\n')
        self.output_handle('|' + 'Mixed Precision Statistics'.center(len(lines[0]) - 2, "*") + '|')
        for i in lines:
            self.output_handle(i)

