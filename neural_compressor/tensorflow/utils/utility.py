# Copyright (c) 2024 Intel Corporation
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

import importlib
import logging
import os
import pickle
import subprocess
import sys
import time
from functools import reduce
from typing import Callable, Dict

import cpuinfo
import numpy as np
import prettytable as pt
import psutil
from pkg_resources import parse_version

from neural_compressor.common import logger

# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}


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


def register_algo(name):
    """Decorator function to register algorithms in the algos_mapping dictionary.

    Usage example:
        @register_algo(name=example_algo)
        def example_algo(model: tf.keras.Model, quant_config: StaticQuantConfig) -> tf.keras.Model:
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


def itex_installed():
    """Check if the Intel® Extension for TensorFlow has been installed."""
    try:
        import intel_extension_for_tensorflow

        return True
    except:
        return False


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


def get_all_fp32_data(data):
    """Get all the fp32 data."""
    return [float(i) for i in data.replace("[", " ").replace("]", " ").split(" ") if i.strip() and len(i) < 32]


def get_tensor_histogram(tensor_data, bins=2048):
    """Get the histogram of the tensor data."""
    max_val = np.max(tensor_data)
    min_val = np.min(tensor_data)
    th = max(abs(min_val), abs(max_val))
    hist, hist_edges = np.histogram(tensor_data, bins=2048, range=(-th, th))
    return (hist, hist_edges, min_val, max_val, th)


def Dequantize(data, scale_info):
    """Dequantize the data with the scale_info."""
    original_shape = data.shape
    max_value = 255.0 if scale_info[0].find("Relu") != -1.0 else 127.0
    _scale = (np.array(scale_info[2]) - np.array(scale_info[1])) / max_value
    de_scale = np.ones(original_shape) * _scale
    de_data = np.multiply(data, de_scale).astype(np.float32)
    return de_data


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


def disable_random(seed=1):
    """A Decorator to disable tf random seed."""
    import tensorflow as tf

    def decorator(func):
        def wrapper(*args, **kw):
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.reset_default_graph()
            tf.compat.v1.set_random_seed(seed)
            return func(*args, **kw)

        return wrapper

    return decorator


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
        if "arch" in info and "ARM" in info["arch"]:  # pragma: no cover
            self._sockets = 1
        else:
            self._sockets = self.get_number_of_sockets()
        self._cores = psutil.cpu_count(logical=False)
        self._cores_per_socket = int(self._cores / self._sockets)

    @property
    def bf16(self):
        """Get whether it is bf16."""
        return self._bf16

    @property
    def vnni(self):
        """Get whether it is vnni."""
        return self._vnni

    @property
    def cores_per_socket(self):
        """Get the cores per socket."""
        return self._cores_per_socket

    def get_number_of_sockets(self) -> int:
        """Get number of sockets in platform."""
        cmd = "cat /proc/cpuinfo | grep 'physical id' | sort -u | wc -l"
        if psutil.WINDOWS:
            cmd = r'wmic cpu get DeviceID | C:\Windows\System32\find.exe /C "CPU"'

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
                    return int(line.decode("utf-8", errors="ignore").strip())
        return 0


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
