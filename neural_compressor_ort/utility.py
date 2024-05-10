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
import pathlib
from typing import Callable, Dict, List, Tuple, Union
import logging
import cpuinfo
import numpy as np
import onnx
import psutil
from neural_compressor_ort import constants

# Dictionary to store a mapping between algorithm names and corresponding algo implementation(function)
algos_mapping: Dict[str, Callable] = {}


#######################################################
####   Options
#######################################################


def _check_value(name, src, supported_type, supported_value=[]):
    """Check if the given object is the given supported type and in the given supported value.

    Example::

        from neural_compressor_ort.base_config import _check_value

        def datatype(self, datatype):
            if _check_value("datatype", datatype, list, ["fp32", "bf16", "uint8", "int8"]):
                self._datatype = datatype
    """
    if isinstance(src, list) and any(
        [not isinstance(i, supported_type) for i in src]):
        assert False, "Type of {} items should be {} but not {}".format(
            name, str(supported_type), [type(i) for i in src])
    elif not isinstance(src, list) and not isinstance(src, supported_type):
        assert False, "Type of {} should be {} but not {}".format(
            name, str(supported_type), type(src))

    if len(supported_value) > 0:
        if isinstance(src, str) and src not in supported_value:
            assert False, "{} is not in supported {}: {}. Skip setting it.".format(
                src, name, str(supported_value))
        elif (isinstance(src, list) and
              all([isinstance(i, str) for i in src]) and
              any([i not in supported_value for i in src])):
            assert False, "{} is not in supported {}: {}. Skip setting it.".format(
                src, name, str(supported_value))

    return True


class Options:
    """Option Class for configs.

    This class is used for configuring global variables. The global variable options is created with this class.
    If you want to change global variables, you should use functions from neural_compressor_ort.utility.py:
        set_random_seed(seed: int)
        set_workspace(workspace: str)
        set_resume_from(resume_from: str)

    Args:
        random_seed(int): Random seed used in neural compressor.
                          Default value is 1978.
        workspace(str): The directory where intermediate files and tuning history file are stored.
                        Default value is:
                            "./nc_workspace/{}/".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")).
        resume_from(str): The directory you want to resume tuning history file from.
                          The tuning history was automatically saved in the workspace directory
                               during the last tune process.
                          Default value is None.

    Example::

        from neural_compressor_ort import set_random_seed, set_workspace, set_resume_from
        set_random_seed(2022)
        set_workspace("workspace_path")
        set_resume_from("workspace_path")
    """

    def __init__(self,
                 random_seed=1978,
                 workspace=constants.DEFAULT_WORKSPACE,
                 resume_from=None):
        """Init an Option object."""
        self.random_seed = random_seed
        self.workspace = workspace
        self.resume_from = resume_from

    @property
    def random_seed(self):
        """Get random seed."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, random_seed):
        """Set random seed."""
        if _check_value("random_seed", random_seed, int):
            self._random_seed = random_seed

    @property
    def workspace(self):
        """Get workspace."""
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        """Set workspace."""
        if _check_value("workspace", workspace, str):
            self._workspace = workspace

    @property
    def resume_from(self):
        """Get resume_from."""
        return self._resume_from

    @resume_from.setter
    def resume_from(self, resume_from):
        """Set resume_from."""
        if resume_from is None or _check_value("resume_from", resume_from, str):
            self._resume_from = resume_from


options = Options()

def _pretty_dict(value, indent=0):
    """Make the logger dict pretty."""
    prefix = "\n" + " " * (indent + 4)
    if isinstance(value, dict):
        items = [
            prefix + repr(key) + ": " + _pretty_dict(value[key], indent + 4)
            for key in value
        ]
        return "{%s}" % (",".join(items) + "\n" + " " * indent)
    elif isinstance(value, list):
        items = [prefix + _pretty_dict(item, indent + 4) for item in value]
        return "[%s]" % (",".join(items) + "\n" + " " * indent)
    elif isinstance(value, tuple):
        items = [prefix + _pretty_dict(item, indent + 4) for item in value]
        return "(%s)" % (",".join(items) + "\n" + " " * indent)
    else:
        return repr(value)


class Logger(object):
    """Logger class."""

    __instance = None

    def __new__(cls):
        """Create a singleton Logger instance."""
        if Logger.__instance is None:
            Logger.__instance = object.__new__(cls)
            Logger.__instance._log()
        return Logger.__instance

    def _log(self):
        """Setup the logger format and handler."""
        LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
        self._logger = logging.getLogger("neural_compressor_ort")
        self._logger.handlers.clear()
        self._logger.setLevel(LOGLEVEL)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d] %(message)s",
            "%Y-%m-%d %H:%M:%S")
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self._logger.addHandler(streamHandler)
        self._logger.propagate = False

    def get_logger(self):
        """Get the logger."""
        return self._logger

    @staticmethod
    def log(level, msg, *args, **kwargs):
        """Output log with the level as a parameter."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().log(level, line, *args, **kwargs)
        else:
            Logger().get_logger().log(level, msg, *args, **kwargs)

    @staticmethod
    def debug(msg, *args, **kwargs):
        """Output log with the debug level."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().debug(line, *args, **kwargs)
        else:
            Logger().get_logger().debug(msg, *args, **kwargs)

    @staticmethod
    def error(msg, *args, **kwargs):
        """Output log with the error level."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().error(line, *args, **kwargs)
        else:
            Logger().get_logger().error(msg, *args, **kwargs)

    @staticmethod
    def fatal(msg, *args, **kwargs):
        """Output log with the fatal level."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().fatal(line, *args, **kwargs)
        else:
            Logger().get_logger().fatal(msg, *args, **kwargs)

    @staticmethod
    def info(msg, *args, **kwargs):
        """Output log with the info level."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().info(line, *args, **kwargs)
        else:
            Logger().get_logger().info(msg, *args, **kwargs)

    @staticmethod
    def warning(msg, *args, **kwargs):
        """Output log with the warning level (Alias of the method warn)."""
        kwargs.setdefault("stacklevel", 2)
        if isinstance(msg, dict):
            for _, line in enumerate(_pretty_dict(msg).split("\n")):
                Logger().get_logger().warning(line, *args, **kwargs)
        else:
            Logger().get_logger().warning(msg, *args, **kwargs)


level = Logger().get_logger().level

logger = Logger


class TuningLogger:
    """A unified logger for the tuning/quantization process.

    It assists validation teams in retrieving logs.
    """

    @classmethod
    def tuning_start(cls) -> None:
        logger.info("Tuning started.")

    @classmethod
    def trial_start(cls, trial_index: int = None) -> None:
        logger.info("%d-trail started.", trial_index)

    @classmethod
    def quantization_start(cls, stacklevel=2) -> None:
        logger.info("Quantization started.", stacklevel=stacklevel)

    @classmethod
    def quantization_end(cls, stacklevel=2) -> None:
        logger.info("Quantization end.", stacklevel=stacklevel)

    @classmethod
    def evaluation_start(cls) -> None:
        logger.info("Evaluation started.")

    @classmethod
    def evaluation_end(cls) -> None:
        logger.info("Evaluation end.")

    @classmethod
    def trial_end(cls, trial_index: int = None) -> None:
        logger.info("%d-trail end.", trial_index)

    @classmethod
    def tuning_end(cls) -> None:
        logger.info("Tuning completed.")

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
                    b"\xB8\x07\x00\x00\x00"
                    b"\x0f\xa2"
                    b"\x89\xC8"
                    b"\xC3",  # mov eax, 7  # cpuid  # mov ax, cx  # ret
                )
                self._vnni = bool(ecx & (1 << 11))
                eax = cpuid._run_asm(
                    b"\xB9\x01\x00\x00\x00",  # mov ecx, 1
                    b"\xB8\x07\x00\x00\x00"
                    b"\x0f\xa2"
                    b"\xC3",  # mov eax, 7  # cpuid  # ret
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
                "%s elapsed time: %s ms" %
                (customized_msg if customized_msg else func.__qualname__,
                 round((end - start) * 1000, 2)))
            return res

        return fi

    return f


def set_random_seed(seed: int):
    """Set the random seed in config."""
    options.random_seed = seed


def set_workspace(workspace: str):
    """Set the workspace in config."""
    options.workspace = workspace


def set_resume_from(resume_from: str):
    """Set the resume_from in config."""
    options.resume_from = resume_from


def log_quant_execution(func):
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
        assert hasattr(
            item, "name"), "{} should have a 'name' attribute defined".format(
                item)  # pragma: no cover
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
        def example_algo(model: Union[onnx.ModelProto, pathlib.Path, str],
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
        model: Union[onnx.ModelProto, pathlib.Path, str],
        white_op_type_list: List[Callable]) -> List[Tuple[str, Callable]]:
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
    utility.logger.debug(f"Get model info: {filter_result}")
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
        quantized_data = ((data.astype(np.float32) / scale).round() +
                          zero_point).astype("B")
    else:
        raise ValueError(
            "Unexpected combination of data type {} and scheme {}.".format(
                qType, scheme))
    return quantized_data


def _calculate_scale_zp(rmin, rmax, quantize_range, qType, scheme):
    """Calculate scale and zero point."""
    if isinstance(rmax, np.ndarray):
        if scheme == "sym":
            max_range = np.maximum(abs(rmin), abs(rmax))
            scale = np.ones(rmax.shape, dtype="float32")
            scale[max_range > 0] = np.array(
                [
                    float(i) / quantize_range
                    for i in (max_range[max_range > 0] *
                              2.0).flatten().tolist()
                ],
                dtype="float32",
            )
        else:
            scale = np.ones(rmax.shape, dtype="float32")
            scale[rmin != rmax] = np.array([
                float(i) / quantize_range
                for i in (rmax - rmin)[rmin != rmax].flatten().tolist()
            ],
                                           dtype="float32")

        if scheme == "sym" and qType == onnx.onnx_pb.TensorProto.INT8:
            zero_point = np.zeros(scale.shape, dtype="int8") if isinstance(
                scale, np.ndarray) else 0
        elif isinstance(scale, np.ndarray) and (scale == 1).all():
            zero_point = (np.zeros(scale.shape, dtype="int8")
                          if qType == onnx.onnx_pb.TensorProto.INT8 else
                          np.zeros(scale.shape, dtype="uint8"))
        elif qType == onnx.onnx_pb.TensorProto.UINT8:
            zero_point = np.maximum(
                0,
                np.minimum(255, ((0 - float(rmin)) /
                                 scale).round()).round()).astype("uint8")
        else:
            zero_point = ((-64 - rmin) /
                          float(scale) if quantize_range == 128 else
                          (-127 - rmin) / float(scale)).round()

    else:
        if scheme == "sym":
            max_range = max(abs(rmin), abs(rmax))
            scale = (float(max_range) *
                     2) / quantize_range if max_range > 0 else 1
        else:
            scale = (float(rmax) -
                     float(rmin)) / quantize_range if rmin != rmax else 1

        if scale == 1 or (scheme == "sym" and
                          qType == onnx.onnx_pb.TensorProto.INT8):
            zero_point = 0
        elif qType == onnx.onnx_pb.TensorProto.UINT8:
            zero_point = round((0 - float(rmin)) / scale)
            zero_point = np.uint8(round(max(0, min(255, zero_point))))
        else:
            zero_point = (round((-64 - float(rmin)) /
                                scale) if quantize_range == 128 else round(
                                    (-127 - float(rmin)) / scale))
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

    scale, zero_point = _calculate_scale_zp(rmin, rmax, quantize_range, qType,
                                            scheme)
    quantized_data = _quantize_data_with_scale_zero(data, qType, scheme, scale,
                                                    zero_point)
    return rmin, rmax, zero_point, scale, quantized_data


def check_model_with_infer_shapes(model):
    """Check if the model has been shape inferred."""
    

    if isinstance(model, (pathlib.Path, str)):
        model = onnx.load(model, load_external_data=False)
    elif isinstance(model, onnx_model.ONNXModel):
        model = model.model
    if len(model.graph.value_info) > 0:
        return True
    return False
