#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
"""The utility of common module."""

import collections
import enum
import importlib
import subprocess
import time
from typing import Dict

import cpuinfo
import psutil
from prettytable import PrettyTable

from neural_compressor.common.utils import Mode, TuningLogger, constants, logger

__all__ = [
    "set_workspace",
    "get_workspace",
    "set_random_seed",
    "set_resume_from",
    "set_tensorboard",
    "dump_elapsed_time",
    "log_process",
    "singleton",
    "LazyImport",
    "CpuInfo",
    "default_tuning_logger",
    "call_counter",
    "cpu_info",
    "ProcessorType",
    "detect_processor_type_based_on_hw",
    "Statistics",
]


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
        self._brand_raw = info.get("brand_raw", "")
        # detect the below info when needed
        self._cores = None
        self._sockets = None
        self._cores_per_socket = None

    @property
    def brand_raw(self):
        """Get the brand name of the CPU."""
        return self._brand_raw

    @brand_raw.setter
    def brand_raw(self, brand_name):
        """Set the brand name of the CPU."""
        self._brand_raw = brand_name

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


cpu_info = CpuInfo()


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
    from neural_compressor.common import options

    options.random_seed = seed


def set_workspace(workspace: str):
    """Set the workspace in config."""
    from neural_compressor.common import options

    options.workspace = workspace


def get_workspace():
    """Get the workspace in config."""
    from neural_compressor.common import options

    return options.workspace


def set_resume_from(resume_from: str):
    """Set the resume_from in config."""
    from neural_compressor.common import options

    options.resume_from = resume_from


def set_tensorboard(tensorboard: bool):
    """Set the tensorboard in config."""
    from neural_compressor.common import options

    options.tensorboard = tensorboard


default_tuning_logger = TuningLogger()


def log_process(mode=Mode.QUANTIZE):
    """Decorator function that logs the stage of process.

    Args:
        mode (Mode): The mode of the process.

    Returns:
        The decorated function.
    """

    def log_process_wrapper(func):
        def inner_wrapper(*args, **kwargs):
            start_log = default_tuning_logger.execution_start
            end_log = default_tuning_logger.execution_end

            start_log(mode=mode, stacklevel=4)

            # Call the original function
            result = func(*args, **kwargs)

            end_log(mode=mode, stacklevel=4)

            return result

        return inner_wrapper

    return log_process_wrapper


# decorator for recording number of times a function is called
FUNC_CALL_COUNTS: Dict[str, int] = collections.defaultdict(int)


def call_counter(func):
    """A decorator that keeps track of the number of times a function is called.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function.
    """

    def wrapper(*args, **kwargs):
        FUNC_CALL_COUNTS[func.__name__] += 1
        return func(*args, **kwargs)

    return wrapper


class ProcessorType(enum.Enum):
    """The processor type."""

    Client = "Client"
    Server = "Server"


def detect_processor_type_based_on_hw():
    """Detects the processor type based on the hardware configuration.

    Returns:
        ProcessorType: The detected processor type (Server or Client).
    """
    # Detect the processor type based on below conditions:
    #   If there are more than one sockets, it is a server.
    #   If the brand name includes key word in `SERVER_PROCESSOR_BRAND_KEY_WORLD_LST`, it is a server.
    #   If the memory size is greater than 32GB, it is a server.
    log_mgs = "Processor type detected as {processor_type} due to {reason}."
    if cpu_info.sockets > 1:
        logger.info(log_mgs.format(processor_type=ProcessorType.Server.value, reason="there are more than one sockets"))
        return ProcessorType.Server
    elif any(brand in cpu_info.brand_raw for brand in constants.SERVER_PROCESSOR_BRAND_KEY_WORLD_LST):
        logger.info(
            log_mgs.format(processor_type=ProcessorType.Server.value, reason=f"the brand name is {cpu_info.brand_raw}.")
        )
        return ProcessorType.Server
    elif psutil.virtual_memory().total / (1024**3) > 32:
        logger.info(
            log_mgs.format(processor_type=ProcessorType.Server.value, reason="the memory size is greater than 32GB")
        )
        return ProcessorType.Server
    else:
        logger.info(
            "Processor type detected as %s, pass `processor_type='server'` to override it if needed.",
            ProcessorType.Client.value,
        )
        return ProcessorType.Client


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
        self.tb = PrettyTable(min_table_width=40)

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
