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
"""Quantization auto-tuning config system.

This file specifies default config options for quantization auto-tuning tool.
User should not change values in this file. Instead, user should write a config
file (in yaml) and use cfg_from_file(yaml_file) to load it and override the default
options.
"""
import _thread
import ast
import importlib
import logging
import os
import os.path as osp
import pickle
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

import cpuinfo
import numpy as np
import prettytable as pt
import psutil
from pkg_resources import parse_version

from neural_compressor.utils import logger

required_libs = {
    "tensorflow": ["tensorflow"],
    "pytorch": ["torch"],
    "pytorch_fx": ["torch"],
    "pytorch_ipex": ["torch", "intel_extension_for_pytorch"],
    "onnxrt_qlinearops": ["onnx", "onnxruntime"],
    "onnxrt_integerops": ["onnx", "onnxruntime"],
    "onnxruntime": ["onnx", "onnxruntime"],
    "mxnet": ["mxnet"],
}


def version1_lt_version2(version1, version2):
    """Check whether version1 is less than version2."""
    return parse_version(version1) < parse_version(version2)


def version1_gt_version2(version1, version2):
    """Check whether version1 is greater than version2."""
    return parse_version(version1) > parse_version(version2)


def version1_eq_version2(version1, version2):
    """Check whether version1 is equal to version2."""
    return parse_version(version1) == parse_version(version2)


def version1_gte_version2(version1, version2):
    """Check whether version1 is greater than version2 or is equal to it."""
    return parse_version(version1) > parse_version(version2) or parse_version(version1) == parse_version(version2)


def version1_lte_version2(version1, version2):
    """Check whether version1 is less than version2 or is equal to it."""
    return parse_version(version1) < parse_version(version2) or parse_version(version1) == parse_version(version2)


class LazyImport(object):
    """Lazy import python module till use."""

    def __init__(self, module_name):
        """Init LazyImport object.

        Args:
           module_name (string): The name of module imported later
        """
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        """Get the attributes of the module by name."""
        try:
            self.module = importlib.import_module(self.module_name)
            mod = getattr(self.module, name)
        except:
            spec = importlib.util.find_spec(str(self.module_name + "." + name))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        return mod

    def __call__(self, *args, **kwargs):
        """Call the function in that module."""
        function_name = self.module_name.split(".")[-1]
        module_name = self.module_name.split(f".{function_name}")[0]
        self.module = importlib.import_module(module_name)
        function = getattr(self.module, function_name)
        return function(*args, **kwargs)


def singleton(cls):
    """Not displayed in API Docs.

    Singleton decorator.
    """
    instances = {}

    def _singleton(*args, **kw):
        """Create a singleton object."""
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


@contextmanager
def time_limit(seconds):
    """Limit the time for context execution."""
    if seconds == 0:
        # seconds = threading.TIMEOUT_MAX
        # TODO WA for fixed the crash for py 3.11.3
        seconds = 3600 * 24 * 365
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def get_size(obj, seen=None):
    """Recursively finds size of objects."""
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
        _graph_node = _graph_def.node if isinstance(_graph_def, tf.compat.v1.GraphDef) else _graph_def.graph_def.node
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


def compute_sparsity(tensor):
    """Compute the sparsity.

    Args:
        tensor: Tensorflow or Pytorch tensor

    Return:
        (the original tensor size, number of zero elements, number of non-zero elements)
    """
    mask = np.ones_like(tensor)
    tensor_size = tensor.size
    dense_mask = tensor != 0
    dense_size = dense_mask.sum()
    return tensor_size, tensor_size - dense_size, dense_size


@contextmanager
def fault_tolerant_file(name):
    """Make another temporary copy of the file."""
    dirpath, filename = osp.split(name)
    with NamedTemporaryFile(dir=os.path.abspath(os.path.expanduser(dirpath)), delete=False, suffix=".tmp") as f:
        yield f
        f.flush()
        os.fsync(f)
        f.close()
        os.replace(f.name, name)


def equal_dicts(d1, d2, compare_keys=None, ignore_keys=None):
    """Check whether two dicts are same except for those ignored keys."""
    assert not (compare_keys and ignore_keys)
    if compare_keys is None and ignore_keys is None:
        return d1 == d2
    elif compare_keys is None and ignore_keys is not None:
        return {k: v for k, v in d1.items() if k not in ignore_keys} == {
            k: v for k, v in d2.items() if k not in ignore_keys
        }
    elif compare_keys is not None and ignore_keys is None:
        return {k: v for k, v in d1.items() if k in compare_keys} == {k: v for k, v in d2.items() if k in compare_keys}
    else:
        assert False


@singleton
class CpuInfo(object):
    """Get CPU Info."""

    def __init__(self):
        """Get whether the cpu numerical format is bf16, the number of sockets, cores and cores per socket."""
        self._bf16 = False
        self._vnni = False
        info = cpuinfo.get_cpu_info()
        if "arch" in info and "X86" in info["arch"]:
            cpuid = cpuinfo.CPUID()
            max_extension_support = cpuid.get_max_extension_support()
            if max_extension_support >= 7:
                ecx = cpuid._run_asm(
                    b"\x31\xC9",  # xor ecx, ecx
                    b"\xB8\x07\x00\x00\x00" b"\x0f\xa2" b"\x89\xC8" b"\xC3",  # mov eax, 7  # cpuid  # mov ax, cx  # ret
                )
                self._vnni = bool(ecx & (1 << 11))
                eax = cpuid._run_asm(
                    b"\xB9\x01\x00\x00\x00",  # mov ecx, 1
                    b"\xB8\x07\x00\x00\x00" b"\x0f\xa2" b"\xC3",  # mov eax, 7  # cpuid  # ret
                )
                self._bf16 = bool(eax & (1 << 5))
        self._info = info
        # detect the below info when needed
        self._cores = None
        self._sockets = None
        self._cores_per_socket = None

    @staticmethod
    def _detect_cores():
        physical_cores = psutil.cpu_count(logical=False)
        return physical_cores

    @property
    def cores(self):
        """Get the number of cores in platform."""
        if self._cores is None:
            self._cores = self._detect_cores()
        return self._cores

    @cores.setter
    def cores(self, num_of_cores):
        """Set the number of cores in platform."""
        self._cores = num_of_cores

    @property
    def bf16(self):
        """Get whether it is bf16."""
        return self._bf16

    @property
    def vnni(self):
        """Get whether it is vnni."""
        return self._vnni

    @property
    def cores_per_socket(self) -> int:
        """Get the cores per socket."""
        if self._cores_per_socket is None:
            self._cores_per_socket = self.cores // self.sockets
        return self._cores_per_socket

    @property
    def sockets(self):
        """Get the number of sockets in platform."""
        if self._sockets is None:
            self._sockets = self._get_number_of_sockets()
        return self._sockets

    @sockets.setter
    def sockets(self, num_of_sockets):
        """Set the number of sockets in platform."""
        self._sockets = num_of_sockets

    def _get_number_of_sockets(self) -> int:
        if "arch" in self._info and "ARM" in self._info["arch"]:  # pragma: no cover
            return 1

        num_sockets = None
        cmd = "cat /proc/cpuinfo | grep 'physical id' | sort -u | wc -l"
        if psutil.WINDOWS:
            cmd = r'wmic cpu get DeviceID | C:\Windows\System32\find.exe /C "CPU"'
        elif psutil.MACOS:  # pragma: no cover
            cmd = "sysctl -n machdep.cpu.core_count"

        num_sockets = None
        try:
            with subprocess.Popen(
                args=cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
            ) as proc:
                proc.wait()
                if proc.stdout:
                    for line in proc.stdout:
                        num_sockets = int(line.decode("utf-8", errors="ignore").strip())
        except Exception as e:
            logger.error("Failed to get number of sockets: %s" % e)
        if isinstance(num_sockets, int) and num_sockets >= 1:
            return num_sockets
        else:
            logger.warning("Failed to get number of sockets, return 1 as default.")
            return 1


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


def combine_histogram(old_hist, arr):
    """Collect layer histogram for arr and combine it with old histogram."""
    new_max = np.max(arr)
    new_min = np.min(arr)
    new_th = max(abs(new_min), abs(new_max))
    (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist
    if new_th <= old_th:
        hist, _ = np.histogram(arr, bins=len(old_hist), range=(-old_th, old_th))
        return (old_hist + hist, old_hist_edges, min(old_min, new_min), max(old_max, new_max), old_th)
    else:
        old_num_bins = len(old_hist)
        old_step = 2 * old_th / old_num_bins
        half_increased_bins = int((new_th - old_th) // old_step + 1)
        new_num_bins = half_increased_bins * 2 + old_num_bins
        new_th = half_increased_bins * old_step + old_th
        hist, hist_edges = np.histogram(arr, bins=new_num_bins, range=(-new_th, new_th))
        hist[half_increased_bins : new_num_bins - half_increased_bins] += old_hist
        return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_th)


def get_tensor_histogram(tensor_data, bins=2048):
    """Get the histogram of the tensor data."""
    max_val = np.max(tensor_data)
    min_val = np.min(tensor_data)
    th = max(abs(min_val), abs(max_val))
    hist, hist_edges = np.histogram(tensor_data, bins=2048, range=(-th, th))
    return (hist, hist_edges, min_val, max_val, th)


def get_all_fp32_data(data):
    """Get all the fp32 data."""
    return [float(i) for i in data.replace("[", " ").replace("]", " ").split(" ") if i.strip() and len(i) < 32]


def get_tuning_history(tuning_history_path):
    """Get tuning history.

    Args:
        tuning_history_path: The tuning history path, which need users to assign
    """
    with open(tuning_history_path, "rb") as f:
        strategy_object = pickle.load(f)
    tuning_history = strategy_object.tuning_history
    return tuning_history


def recover(fp32_model, tuning_history_path, num, **kwargs):
    """Get offline recover tuned model.

    Args:
        fp32_model: Input model path
        tuning_history_path: The tuning history path, which needs user to assign
        num: tune index
    """
    tuning_history = get_tuning_history(tuning_history_path)
    target_history = tuning_history[0]["history"]
    q_config = target_history[num]["q_config"]
    try:
        framework = tuning_history[0]["cfg"]["model"]["framework"]
    except Exception as e:
        framework = tuning_history[0]["cfg"].quantization.framework

    if "pytorch" in framework:
        from neural_compressor.utils.pytorch import load

        tune_index_qmodel = load(model=fp32_model, history_cfg=q_config, **kwargs)
        return tune_index_qmodel

    from neural_compressor.adaptor import FRAMEWORKS

    adaptor = FRAMEWORKS[framework](q_config["framework_specific_info"])
    if "onnx" in framework:
        from neural_compressor.model import Model

        ox_fp32_model = Model(fp32_model)
        tune_index_qmodel = adaptor.recover(ox_fp32_model, q_config)
        return tune_index_qmodel
    elif "tensorflow" in framework:
        from neural_compressor.model import Model

        tf_fp32_model = Model(fp32_model)
        tune_index_qmodel = adaptor.recover_tuned_model(tf_fp32_model, q_config)
        return tune_index_qmodel
    elif "mxnet" in framework:
        from neural_compressor.model import Model

        mx_fp32_model = Model(fp32_model)
        tune_index_qmodel = adaptor.recover_tuned_model(mx_fp32_model, q_config)
        return tune_index_qmodel


def str2array(s):
    """Get the array of the string."""
    s = re.sub(r"\[ +", "[", s.strip())
    s = re.sub(r"[,\s]+", ", ", s)
    s = re.sub(r"\]\[", "], [", s)

    return np.array(ast.literal_eval(s))


def dequantize_weight(weight_tensor, min_filter_tensor, max_filter_tensor):
    """Dequantize the weight with min-max filter tensors."""
    weight_channel = weight_tensor.shape[-1]
    if len(min_filter_tensor) == 1:
        weight_tensor = weight_tensor * ((max_filter_tensor[0] - min_filter_tensor[0]) / 127.0)
    else:
        # TODO to calculate the de-quantized result in a parallel way
        for i in range(weight_channel):
            weight_tensor[:, :, :, i] = weight_tensor[:, :, :, i] * (
                (max_filter_tensor[i] - min_filter_tensor[i]) / 127.0
            )
    return weight_tensor


def Dequantize(data, scale_info):
    """Dequantize the data with the scale_info."""
    import numpy as np

    original_shape = data.shape
    max_value = 255.0 if scale_info[0].find("Relu") != -1.0 else 127.0
    _scale = (np.array(scale_info[2]) - np.array(scale_info[1])) / max_value
    de_scale = np.ones(original_shape) * _scale
    de_data = np.multiply(data, de_scale).astype(np.float32)
    return de_data


class CaptureOutputToFile(object):
    """Not displayed in API Docs.

    Capture the output to file.
    """

    def __init__(self, tmp_file_path, stream=sys.stderr):
        """Open a temporary file."""
        self.orig_stream_fileno = stream.fileno()
        self.tmp_file = open(tmp_file_path, "w")

    def __enter__(self):
        """Duplicate the file descriptor to the stream."""
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        os.dup2(self.tmp_file.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        """Duplicate the stream descriptor to the file."""
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.tmp_file.close()


class Statistics:
    """The statistics printer."""

    def __init__(self, data, header, field_names, output_handle=logger.info):
        """Init a Statistics object.

        Args:
            data: The statistics data
            header: The table header
            field_names: The field names
            output_handle: The output logging method
        """
        self.field_names = field_names
        self.header = header
        self.data = data
        self.output_handle = output_handle
        self.tb = pt.PrettyTable(min_table_width=40)

    def print_stat(self):
        """Print the statistics."""
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
        lines = self.tb.get_string().split("\n")
        self.output_handle("|" + self.header.center(len(lines[0]) - 2, "*") + "|")
        for i in lines:
            self.output_handle(i)


class MODE(Enum):
    """Mode: Quantization, Benchmark or Pruning."""

    QUANTIZATION = 1
    BENCHMARK = 2
    PRUNING = 3


class GLOBAL_STATE:
    """Access the global model."""

    STATE = MODE.QUANTIZATION


def load_data_from_pkl(path, filename):
    """Load data from local pkl file.

    Args:
        path: The directory to load data
        filename: The filename to load
    """
    try:
        file_path = os.path.join(path, filename)
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
            return data
    except FileExistsError:
        logging.getLogger("neural_compressor").info("Can not open %s." % path)


def dump_data_to_local(data, path, filename):
    """Dump data to local as pkl file.

    Args:
        data: Data used to dump
        path: The directory to save data
        filename: The filename to dump

    Returns:
        loaded data
    """
    from pathlib import Path

    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(path, filename)
    with open(file_path, "wb") as fp:
        pickle.dump(data, fp)
        logging.getLogger("neural_compressor").info("Dumped data to %s" % file_path)


def set_random_seed(seed: int):
    """Set the random seed in config."""
    from neural_compressor.config import options

    options.random_seed = seed


def set_workspace(workspace: str):
    """Set the workspace in config."""
    from neural_compressor.config import options

    options.workspace = workspace


def set_resume_from(resume_from: str):
    """Set the resume_from in config."""
    from neural_compressor.config import options

    options.resume_from = resume_from


def set_tensorboard(tensorboard: bool):
    """Set the tensorboard in config."""
    from neural_compressor.config import options

    options.tensorboard = tensorboard


def show_memory_info(hint):
    """Show process full memory."""
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024.0 / 1024
    print("{} memory used: {} MB".format(hint, memory))


def dump_class_attrs(obj, result={}):
    """Dump the attributes and values of a config class.

    Args:
        obj: An instance of a config class.
        result: An dict for recording attributes and values.
    """
    obj_name = obj.__class__.__name__
    if obj_name not in result:
        result[obj_name] = {}
    for attr in dir(obj):
        if not attr.startswith("__"):
            value = getattr(obj, attr)
            value_class_name = value.__class__.__name__
            if "Config" in value_class_name or "Criterion" in value_class_name:
                dump_class_attrs(value, result=result[obj_name])
            else:
                attr = attr[1:] if attr.startswith("_") else attr
                result[obj_name][attr] = value


from functools import reduce


def deep_get(dictionary, keys, default=None):
    """Get the dot key's item in nested dict.

       For example, person = {'person':{'name':{'first':'John'}}}
       deep_get(person, "person.name.first") will output 'John'.

    Args:
        dictionary (dict): The dict object to get keys
        keys (dict): The deep keys
        default (object): The return item if key not exists
    Returns:
        item: the item of the deep dot keys
    """
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)


def deep_set(dictionary, keys, value):
    """Set the dot key's item in nested dict.

       For example, person = {'person':{'name':{'first':'John'}}}
       deep_set(person, "person.sex", 'male') will output
       {'person': {'name': {'first': 'John'}, 'sex': 'male'}}

    Args:
        dictionary (dict): The dict object to get keys
        keys (dict): The deep keys
        value (object): The value of the setting key
    """
    keys = keys.split(".")
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, DotDict())
    dictionary[keys[-1]] = value


class DotDict(dict):
    """Access yaml using attributes instead of using the dictionary notation.

    Args:
        value (dict): The dict object to access.
    """

    def __init__(self, value=None):
        """Init DotDict.

        Args:
            value: The value to be initialized. Defaults to None.
        """
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError("expected dict")

    def __getitem__(self, key):
        """Get value by key.

        Args:
            key: The query item.
        """
        value = self.get(key, None)
        return value

    def __setitem__(self, key, value):
        """Add new key and value pair.

        Args:
            key: something like key in dict.
            value: value assigned to key.
        """
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            value = DotDict(value[0])
        if isinstance(value, list) and len(value) > 1 and all(isinstance(v, dict) for v in value):
            value = DotDict({k: v for d in value for k, v in d.items()})
        super(DotDict, self).__setitem__(key, value)

    def __getstate__(self):
        """Return self dict."""
        return self.__dict__

    def __setstate__(self, d):
        """Update self dict."""
        self.__dict__.update(d)

    __setattr__, __getattr__ = __setitem__, __getitem__


def compare_objects(obj1, obj2, ignore_attrs):
    """Compare two objects and ignore the specified attributes.

    Args:
        obj1: The first object to compare.
        obj2: The second object to compare.
        ignore_attrs: A list of attribute names to ignore during the comparison.

    Returns:
        True if the objects are equal ignoring the specified attributes, False otherwise.
    """
    # Check if the objects are of the same type
    if type(obj1) != type(obj2):
        return False

    # Check if the objects have the same set of attributes
    attrs1 = set(obj1.__dict__.keys())
    attrs2 = set(obj2.__dict__.keys())
    if attrs1 != attrs2:
        return False
    # Compare the attributes, ignoring the specified ones
    for attr in attrs1 - set(ignore_attrs):
        if getattr(obj1, attr) != getattr(obj2, attr):
            return False
    return True


def alias_param(param_name: str, param_alias: str):
    """Decorator for aliasing a param in a function.

    Args:
        param_name: Name of param in function to alias.
        param_alias: Alias that can be used for this param.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            alias_param_value = kwargs.get(param_alias)
            if alias_param_value:  # pragma: no cover
                kwargs[param_name] = alias_param_value
                del kwargs[param_alias]
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def print_table(
    column_mapping: Dict[str, str],
    table_entries: List[Any],
    output_handler=logger.info,
    title: Optional[str] = None,
    insert_newlines=False,
    precision: Optional[int] = None,
) -> None:
    """Print table with prettytable.

    Args:
        :
        column_mapping (dict): Dictionary where key is a column header and value corresponds
             to object's attribute.
        table_entries (list): List of objects to be included in the table.
        output_handler (func, optional): Output handler function.
        title (str): Title for the table
        insert_newlines (bool): Add newlines to each row
        precision (int): Number of digits after the decimal point

    Returns:
        None
    """
    from functools import reduce

    import numpy as np

    table = pt.PrettyTable(min_table_width=40)
    if title is not None:
        table.title = title
    table.field_names = list(column_mapping.keys())
    for entry in table_entries:
        table_row = []
        for _, attribute in column_mapping.items():
            if isinstance(entry, dict):
                table_row.append(entry.get(attribute))
            else:
                value = reduce(getattr, [entry] + attribute.split("."))
                if (isinstance(value, np.floating) or isinstance(value, float)) and isinstance(precision, int):
                    if "e" in str(value):
                        value = f"{value:.{precision}e}"
                    else:
                        value = round(value, precision)
                table_row.append(value)
        table.add_row(table_row)
    lines = table.get_string().split("\n")
    for i in lines:
        if insert_newlines:
            i += "\n"
        output_handler(i)


def get_tensors_info(workload_location, model_type: str = "optimized") -> dict:
    """Get information about tensors.

    Args:
        workload_location: path to workload directory
        model_type: type of model. Supported model types: "input", "optimized"

    Returns:
        dictionary with tensors info
    """
    tensors_filenames = {
        "input": os.path.join("fp32", "inspect_result.pkl"),
        "optimized": os.path.join("quan", "inspect_result.pkl"),
    }

    tensors_filename = tensors_filenames.get(model_type, None)
    if tensors_filename is None:
        raise Exception(f"Could not find tensors data for {model_type} model.")
    tensors_path = os.path.join(
        workload_location,
        "inspect_saved",
        tensors_filename,
    )
    if not os.path.exists(tensors_path):
        raise Exception("Could not find tensor data for specified optimization.")
    with open(tensors_path, "rb") as tensors_pickle:
        dump_tensor_result = pickle.load(tensors_pickle)
    return dump_tensor_result


def get_weights_details(workload_location: str) -> list:
    """Get weights details for model.

    Args:
        workload_location: path to workload directory

    Returns:
        list of WeightDetails objects
    """
    from neural_compressor.utils.weights_details import WeightsDetails

    weights_details = []

    input_model_tensors: dict = get_tensors_info(workload_location, model_type="input")["weight"]
    optimized_model_tensors: dict = get_tensors_info(workload_location, model_type="optimized")["weight"]

    common_ops = list(set(input_model_tensors.keys()) & set(optimized_model_tensors.keys()))
    for op_name in common_ops:
        input_model_op_tensors = input_model_tensors[op_name]
        optimized_model_op_tensors = optimized_model_tensors[op_name]

        if isinstance(input_model_op_tensors, dict):
            tensors_data = zip(input_model_op_tensors.items(), optimized_model_op_tensors.items())
            for (_, input_op_tensor_values), (_, optimized_op_tensor_values) in tensors_data:
                if input_op_tensor_values.shape != optimized_op_tensor_values.shape:
                    continue

                weights_entry = WeightsDetails(
                    op_name,
                    input_op_tensor_values,
                    optimized_op_tensor_values,
                )
                weights_details.append(weights_entry)
    return weights_details


def dump_table(
    filepath: str,
    column_mapping: Dict[str, str],
    table_entries: List[Any],
    file_type: str = "csv",
) -> None:
    """Dump table data to file.

    Args:
        :
        filepath (str): The path of the file to which the data is to be dumped.
        column_mapping (dict): Dictionary where key is a column header and value corresponds
             to object's attribute.
        table_entries (list): List of objects to be included in the table.
        file_type (str): name of the file extension. Currently only csv is supported.

    Returns:
        None
    """
    supported_filetypes = {
        "csv": dump_table_to_csv,
    }

    dump_function = supported_filetypes.get(file_type, None)
    if dump_function is None:
        raise Exception(f"{file_type} is not supported")
    dump_function(
        filepath=filepath,
        column_mapping=column_mapping,
        table_entries=table_entries,
    )


def dump_table_to_csv(
    filepath: str,
    column_mapping: Dict[str, str],
    table_entries: List[Any],
):
    """Dump table data to csv file.

    Args:
        :
        filepath (str): The path of the file to which the data is to be dumped.
        column_mapping (dict): Dictionary where key is a column header and value corresponds
             to object's attribute.
        table_entries (list): List of objects to be included in the table.

    Returns:
        None
    """
    from csv import DictWriter
    from functools import reduce

    fieldnames = list(column_mapping.keys())

    parsed_entries = []

    for entry in table_entries:
        parsed_entry = {}
        for column_name, attribute in column_mapping.items():
            if isinstance(entry, dict):
                parsed_entry.update({column_name: entry.get(attribute)})
            else:
                value = reduce(getattr, [entry] + attribute.split("."))
                parsed_entry.update({column_name: value})
        parsed_entries.append(parsed_entry)

    with open(filepath, "w", newline="") as csv_file:
        writer = DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(parsed_entries)


def get_number_of_sockets() -> int:
    """Get number of sockets in platform."""
    cmd = "cat /proc/cpuinfo | grep 'physical id' | sort -u | wc -l"
    if sys.platform == "win32":
        cmd = 'wmic cpu get DeviceID | find /c "CPU"'
    elif sys.platform == "darwin":  # pragma: no cover
        cmd = "sysctl -n machdep.cpu.core_count"

    proc = subprocess.Popen(
        args=cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=False,
    )
    proc.wait()
    if proc.stdout:
        for line in proc.stdout:
            return int(line.decode("utf-8", errors="ignore").strip())
    return 0


class OpEntry:
    """OP entry class."""

    def __init__(self, op_name: str, mse: float, activation_min: float, activation_max: float):
        """Initialize OP entry.

        Args:
            op_name: name of the OP
            mse: MSE value for OP
            activation_min: minimum value of activation
            activation_max: maximum value of activation

        Returns:
            None
        """
        self.op_name: str = op_name
        self.mse: float = mse
        self.activation_min: float = activation_min
        self.activation_max: float = activation_max


def print_op_list(workload_location: str):
    """Print OP table and dump it to CSV.

    Args:
        workload_location: path to workload directory

    Returns:
        None
    """
    minmax_file_path = os.path.join(workload_location, "inspect_saved", "activation_min_max.pkl")
    if not os.path.exists(minmax_file_path):
        logging.getLogger("neural_compressor").warning("Could not find activation min max data.")
        return
    input_model_tensors = get_tensors_info(
        workload_location,
        model_type="input",
    )[
        "activation"
    ][0]
    optimized_model_tensors = get_tensors_info(
        workload_location,
        model_type="optimized",
    )[
        "activation"
    ][0]
    op_list = get_op_list(minmax_file_path, input_model_tensors, optimized_model_tensors)
    sorted_op_list = sorted(op_list, key=lambda x: x.mse, reverse=True)
    if len(op_list) <= 0:
        return
    print_table(
        title="Activations summary",
        column_mapping={
            "OP name": "op_name",
            "MSE": "mse",
            "Activation min": "activation_min",
            "Activation max": "activation_max",
        },
        table_entries=sorted_op_list,
        precision=5,
    )

    activations_table_file = os.path.join(
        workload_location,
        "activations_table.csv",
    )
    dump_table(
        filepath=activations_table_file,
        column_mapping={
            "OP name": "op_name",
            "MSE": "mse",
            "Activation min": "activation_min",
            "Activation max": "activation_max",
        },
        table_entries=sorted_op_list,
        file_type="csv",
    )


def get_op_list(minmax_file_path, input_model_tensors, optimized_model_tensors) -> List[OpEntry]:
    """Get OP list for model.

    Args:
        minmax_file_path: path to activation_min_max.pkl
        input_model_tensors: dict with input tensors details
        optimized_model_tensors: dict with optimized tensors details

    Returns:
        list of OpEntry elements
    """
    with open(minmax_file_path, "rb") as min_max_file:
        min_max_data: dict = pickle.load(min_max_file)

    op_list: List[OpEntry] = []

    for op_name, min_max in min_max_data.items():
        mse = calculate_mse(op_name, input_model_tensors, optimized_model_tensors)
        if mse is None:
            continue
        min = float(min_max.get("min", None))
        max = float(min_max.get("max", None))
        op_entry = OpEntry(op_name, mse, min, max)
        op_list.append(op_entry)
    return op_list


def calculate_mse(
    op_name: str,
    input_model_tensors: dict,
    optimized_model_tensors: dict,
) -> Optional[float]:
    """Calculate MSE for specified OP.

    Args:
        op_name: name of the OP
        input_model_tensors: dict with input tensors details
        optimized_model_tensors: dict with optimized tensors details

    Returns:
        MSE value for specified op_name
    """
    input_model_op_data = input_model_tensors.get(op_name, None)
    optimized_model_op_data = optimized_model_tensors.get(op_name, None)

    if input_model_op_data is None or optimized_model_op_data is None:
        return None

    mse: float = mse_metric_gap(
        next(iter(input_model_op_data.values()))[0],
        next(iter(optimized_model_op_data.values()))[0],
    )

    return mse


def mse_metric_gap(fp32_tensor: Any, dequantize_tensor: Any) -> float:
    """Calculate the euclidean distance between fp32 tensor and int8 dequantize tensor.

    Args:
        fp32_tensor (tensor): The FP32 tensor.
        dequantize_tensor (tensor): The INT8 dequantize tensor.

    Returns:
        MSE value
    """
    import numpy as np

    fp32_max = np.max(fp32_tensor)  # type: ignore
    fp32_min = np.min(fp32_tensor)  # type: ignore
    dequantize_max = np.max(dequantize_tensor)  # type: ignore
    dequantize_min = np.min(dequantize_tensor)  # type: ignore
    fp32_tensor_norm = fp32_tensor
    dequantize_tensor_norm = dequantize_tensor
    if (fp32_max - fp32_min) != 0:
        fp32_tensor_norm = (fp32_tensor - fp32_min) / (fp32_max - fp32_min)

    if (dequantize_max - dequantize_min) != 0:
        dequantize_tensor_norm = (dequantize_tensor - dequantize_min) / (dequantize_max - dequantize_min)

    diff_tensor = fp32_tensor_norm - dequantize_tensor_norm
    euclidean_dist = np.sum(diff_tensor**2)  # type: ignore
    return euclidean_dist / fp32_tensor.size


def check_key_exist(data, key):
    """Recursively checks if a key exists in a dictionary or list.

    Args:
        data (dict or list): The dictionary or list to search.
        key (any): The key to search for.

    Returns:
        bool: True if the key exists in the data structure, False otherwise.

    Examples:
        >>> check_key_exist({'a': 1, 'b': {'c': 2}}, 'c')
        True
        >>> check_key_exist([{'a': 1}, {'b': 2}], 'b')
        True
        >>> check_key_exist({'a': 1, 'b': [1, 2, 3]}, 'c')
        False
    """
    if isinstance(data, dict):
        if key in data:
            return True
        for value in data.values():
            if check_key_exist(value, key):
                return True
    elif isinstance(data, list):
        for item in data:
            if check_key_exist(item, key):
                return True
    return False
