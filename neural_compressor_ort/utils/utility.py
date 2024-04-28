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

import importlib
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import cpuinfo
import numpy as np
import onnx
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
import psutil
from packaging.version import Version

from neural_compressor_ort.utils import logger

__all__ = [
    "set_workspace",
    "set_random_seed",
    "set_resume_from",
    "dump_elapsed_time",
    "log_quant_execution",
    "singleton",
    "LazyImport",
    "CpuInfo",
    "algos_mapping",
    "dtype_mapping",
    "find_by_name",
    "simple_progress_bar",
    "register_algo",
    "get_model_info",
    "is_B_transposed",
    "get_qrange_for_qType",
    "quantize_data",
    "check_model_with_infer_shapes",
]


# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}


def singleton(cls):
    """Singleton decorator."""

    instances = {}

    def _singleton(*args, **kw):
        """Create a singleton object."""
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


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


@singleton
class CpuInfo(object):
    """CPU info collection."""

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
        # TODO: The implementation will be refined in the future.
        # https://github.com/intel/neural-compressor/tree/detect_sockets
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
        elif psutil.MACOS:  # pragma: no cover
            cmd = "sysctl -n machdep.cpu.core_count"

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
            logger.info(
                "%s elapsed time: %s ms"
                % (customized_msg if customized_msg else func.__qualname__, round((end - start) * 1000, 2))
            )
            return res

        return fi

    return f


def set_random_seed(seed: int):
    """Set the random seed in config."""
    from neural_compressor_ort.utils.base_config import options

    options.random_seed = seed


def set_workspace(workspace: str):
    """Set the workspace in config."""
    from neural_compressor_ort.utils.base_config import options

    options.workspace = workspace


def set_resume_from(resume_from: str):
    """Set the resume_from in config."""
    from neural_compressor_ort.utils.base_config import options

    options.resume_from = resume_from


def log_quant_execution(func):
    from neural_compressor_ort.utils import TuningLogger

    default_tuning_logger = TuningLogger()

    def wrapper(*args, **kwargs):
        default_tuning_logger.quantization_start(stacklevel=4)

        # Call the original function
        result = func(*args, **kwargs)

        default_tuning_logger.quantization_end(stacklevel=4)
        return result

    return wrapper


dtype_mapping = {
    "fp32": 1,
    "float32": 1,
    "uint8": 2,
    "int8": 3,
    "uint16": 4,
    "int16": 5,
    "int32": 6,
    "int64": 7,
    "string": 8,
    "bool": 9,
    "fp16": 10,
    "float16": 10,
    "double": 11,
    "uint32": 12,
    "uint64": 13,
    "complex64": 14,
    "complex128": 15,
    "bf16": 16,
    "bfloat16": 16,
}


def find_by_name(name, item_list):
    """Helper function to find item by name in a list."""
    items = []
    for item in item_list:
        assert hasattr(item, "name"), "{} should have a 'name' attribute defined".format(item)  # pragma: no cover
        if item.name == name:
            items.append(item)
    if len(items) > 0:
        return items[0]
    else:
        return None


def simple_progress_bar(total, i):
    """Progress bar for cases where tqdm can't be used."""
    progress = i / total
    bar_length = 20
    bar = "#" * int(bar_length * progress)
    spaces = " " * (bar_length - len(bar))
    percentage = progress * 100
    print(f"\rProgress: [{bar}{spaces}] {percentage:.2f}%", end="")


def register_algo(name):
    """Decorator function to register algorithms in the algos_mapping dictionary.

    Usage example:
        @register_algo(name=example_algo)
        def example_algo(model: Union[onnx.ModelProto, Path, str],
                         quant_config: RTNConfig) -> onnx.ModelProto:
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


def get_model_info(
    model: Union[onnx.ModelProto, Path, str], white_op_type_list: List[Callable]
) -> List[Tuple[str, Callable]]:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)
    filter_result = []
    filter_result_set = set()
    for node in model.graph.node:
        if node.op_type in white_op_type_list:
            pair = (node.name, node.op_type)
            if pair not in filter_result_set:
                filter_result_set.add(pair)
                filter_result.append(pair)
    logger.debug(f"Get model info: {filter_result}")
    return filter_result


def is_B_transposed(node):
    """Whether inuput B is transposed."""
    transB = [attr for attr in node.attribute if attr.name == "transB"]
    if len(transB):
        return 0 < onnx.helper.get_attribute_value(transB[0])
    return False


def get_qrange_for_qType(qType, reduce_range=False):
    """Helper function to get the quantization range for a type.

    Args:
        qType (int): data type
        reduce_range (bool, optional): use 7 bit or not. Defaults to False.
    """
    if qType == onnx.onnx_pb.TensorProto.UINT8:
        return 127 if reduce_range else 255
    elif qType == onnx.onnx_pb.TensorProto.INT8:
        # [-64, 64] for reduce_range, and [-127, 127] full_range.
        return 128 if reduce_range else 254
    else:
        raise ValueError("unsupported quantization data type")


def _quantize_data_with_scale_zero(data, qType, scheme, scale, zero_point):
    """Quantize data with scale and zero point.

    To pack weights, we compute a linear transformation
        - when data type == uint8 mode, from [rmin, rmax] -> [0, 2^{b-1}] and
        - when data type == int8, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
            m = max(abs(rmin), abs(rmax))

    Args:
        data (np.array): data to quantize
        qType (int): data type to quantize to. Supported types UINT8 and INT8
        scheme (string): sym or asym quantization.
        scale (float): computed scale of quantized data
        zero_point (uint8 or int8): computed zero point of quantized data
    """
    data = np.asarray(data)
    if qType == onnx.onnx_pb.TensorProto.INT8 and scheme == "sym":
        # signed byte type
        quantized_data = (data.astype(np.float32) / scale).round().astype("b")
    elif qType == onnx.onnx_pb.TensorProto.UINT8 and scheme == "asym":
        quantized_data = ((data.astype(np.float32) / scale).round() + zero_point).astype("B")
    else:
        raise ValueError("Unexpected combination of data type {} and scheme {}.".format(qType, scheme))
    return quantized_data


def _calculate_scale_zp(rmin, rmax, quantize_range, qType, scheme):
    """Calculate scale and zero point."""
    if isinstance(rmax, np.ndarray):
        if scheme == "sym":
            max_range = np.maximum(abs(rmin), abs(rmax))
            scale = np.ones(rmax.shape, dtype="float32")
            scale[max_range > 0] = np.array(
                [float(i) / quantize_range for i in (max_range[max_range > 0] * 2.0).flatten().tolist()],
                dtype="float32",
            )
        else:
            scale = np.ones(rmax.shape, dtype="float32")
            scale[rmin != rmax] = np.array(
                [float(i) / quantize_range for i in (rmax - rmin)[rmin != rmax].flatten().tolist()], dtype="float32"
            )

        if scheme == "sym" and qType == onnx.onnx_pb.TensorProto.INT8:
            zero_point = np.zeros(scale.shape, dtype="int8") if isinstance(scale, np.ndarray) else 0
        elif isinstance(scale, np.ndarray) and (scale == 1).all():
            zero_point = (
                np.zeros(scale.shape, dtype="int8")
                if qType == onnx.onnx_pb.TensorProto.INT8
                else np.zeros(scale.shape, dtype="uint8")
            )
        elif qType == onnx.onnx_pb.TensorProto.UINT8:
            zero_point = np.maximum(0, np.minimum(255, ((0 - float(rmin)) / scale).round()).round()).astype("uint8")
        else:
            zero_point = (
                (-64 - rmin) / float(scale) if quantize_range == 128 else (-127 - rmin) / float(scale)
            ).round()

    else:
        if scheme == "sym":
            max_range = max(abs(rmin), abs(rmax))
            scale = (float(max_range) * 2) / quantize_range if max_range > 0 else 1
        else:
            scale = (float(rmax) - float(rmin)) / quantize_range if rmin != rmax else 1

        if scale == 1 or (scheme == "sym" and qType == onnx.onnx_pb.TensorProto.INT8):
            zero_point = 0
        elif qType == onnx.onnx_pb.TensorProto.UINT8:
            zero_point = round((0 - float(rmin)) / scale)
            zero_point = np.uint8(round(max(0, min(255, zero_point))))
        else:
            zero_point = (
                round((-64 - float(rmin)) / scale) if quantize_range == 128 else round((-127 - float(rmin)) / scale)
            )
    return scale, zero_point


def quantize_data(data, quantize_range, qType, scheme):
    """Quantize data.

    To pack weights, we compute a linear transformation
        - when data type == uint8 mode, from [rmin, rmax] -> [0, 2^{b-1}] and
        - when data type == int8, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
            m = max(abs(rmin), abs(rmax))
    and add necessary intermediate nodes to transform quantized weight to full weight
    using the equation r = S(q-z), where
        r: real original value
        q: quantized value
        S: scale
        z: zero point

    Args:
        data (array): data to quantize
        quantize_range (list): list of data to weight pack.
        qType (int): data type to quantize to. Supported types UINT8 and INT8
        scheme (string): sym or asym quantization.
    """
    rmin = min(min(data), 0)
    rmax = max(max(data), 0)

    scale, zero_point = _calculate_scale_zp(rmin, rmax, quantize_range, qType, scheme)
    quantized_data = _quantize_data_with_scale_zero(data, qType, scheme, scale, zero_point)
    return rmin, rmax, zero_point, scale, quantized_data


def check_model_with_infer_shapes(model):
    """Check if the model has been shape inferred."""
    from neural_compressor_ort.utils.onnx_model import ONNXModel

    if isinstance(model, (Path, str)):
        model = onnx.load(model, load_external_data=False)
    elif isinstance(model, ONNXModel):
        model = model.model
    if len(model.graph.value_info) > 0:
        return True
    return False
