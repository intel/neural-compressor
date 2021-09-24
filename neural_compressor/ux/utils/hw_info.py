# -*- coding: utf-8 -*-
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
"""UX server HW info module."""

import platform
import re
import subprocess
from typing import Any, Dict, Union

import cpuinfo
import psutil

from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import determine_ip


class HWInfo(JsonSerializer):
    """Class responsible for gathering information about platform hardware."""

    initialized: bool = False

    sockets: int = 0
    cores: int = 0
    cores_per_socket: int = 0
    threads_per_socket: int = 0
    total_memory: str = ""
    system: str = ""
    ip: str = ""
    platform: str = ""
    hyperthreading_enabled: bool = False
    turboboost_enabled: Union[str, bool] = False
    bios_version: str = ""
    kernel: str = ""
    min_cpu_freq: str = ""
    max_cpu_freq: str = ""

    def __init__(self) -> None:
        """Initialize HW Info class and gather information platform hardware."""
        super().__init__()
        if not HWInfo.initialized:
            self.initialize()

    def serialize(
        self,
        serialization_type: str = "default",
    ) -> Dict[str, Any]:
        """Serialize class to dict."""
        return {
            "sockets": self.sockets,
            "cores": self.cores,
            "cores_per_socket": self.cores_per_socket,
            "threads_per_socket": self.threads_per_socket,
            "total_memory": self.total_memory,
            "system": self.system,
            "ip": self.ip,
            "platform": self.platform,
            "hyperthreading_enabled": self.hyperthreading_enabled,
            "turboboost_enabled": self.turboboost_enabled,
            "bios_version": self.bios_version,
            "kernel": self.kernel,
            "min_cpu_freq": self.min_cpu_freq,
            "max_cpu_freq": self.max_cpu_freq,
        }

    def initialize(self) -> None:
        """Set class variables."""
        cpu_info = cpuinfo.get_cpu_info()
        HWInfo.sockets = get_number_of_sockets()
        HWInfo.cores = psutil.cpu_count(logical=False)
        HWInfo.cores_per_socket = int(psutil.cpu_count(logical=False) / self.sockets)
        HWInfo.threads_per_socket = int(psutil.cpu_count(logical=True) / self.sockets)
        HWInfo.total_memory = f"{psutil.virtual_memory().total / (1024 ** 3):.3f}GB"
        HWInfo.system = get_distribution()
        HWInfo.ip = determine_ip()
        if cpu_info.get("brand"):
            HWInfo.platform = cpu_info.get("brand")
        elif cpu_info.get("brand_raw"):
            HWInfo.platform = cpu_info.get("brand_raw")
        else:
            HWInfo.platform = ""
        HWInfo.hyperthreading_enabled = psutil.cpu_count(logical=False) != psutil.cpu_count(
            logical=True,
        )
        HWInfo.turboboost_enabled = is_turbo_boost_enabled()
        HWInfo.bios_version = get_bios_version()
        HWInfo.kernel = get_kernel_version()
        try:
            min_cpu_freq = int(psutil.cpu_freq(percpu=False).min)
            max_cpu_freq = int(psutil.cpu_freq(percpu=False).max)
        except Exception:
            log.warning("Cannot collect cpu frequency information.")
            min_cpu_freq = 0
            max_cpu_freq = 0
        finally:
            if min_cpu_freq == 0:
                HWInfo.min_cpu_freq = "n/a"
            else:
                HWInfo.min_cpu_freq = f"{min_cpu_freq}Hz"
            if max_cpu_freq == 0:
                HWInfo.max_cpu_freq = "n/a"
            else:
                HWInfo.max_cpu_freq = f"{max_cpu_freq}Hz"
        HWInfo.initialized = True


def get_number_of_sockets() -> int:
    """Get number of sockets in platform."""
    cmd = "lscpu | grep 'Socket(s)' | cut -d ':' -f 2"
    if psutil.WINDOWS:
        cmd = 'wmic cpu get DeviceID | find /c "CPU"'

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


def get_distribution() -> str:
    """
    Return system distibution.

    :return: distribution name
    :rtype: str
    """
    return f"{platform.system()} {platform.release()}"


def get_bios_version() -> str:
    """Return bios version."""
    if psutil.LINUX:
        try:
            cmd = "cat /sys/class/dmi/id/bios_version"

            proc = subprocess.Popen(
                args=cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
            )
            proc.wait()
            bios_version = "n/a"
            if proc.stdout and proc.returncode == 0:
                for line in proc.stdout:
                    bios_version = line.decode("utf-8", errors="ignore").strip()

            cmd = "grep microcode  /proc/cpuinfo -m1"
            proc = subprocess.Popen(
                args=cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
            )
            proc.wait()
            microcode = "n/a"
            if proc.stdout and proc.returncode == 0:
                for line in proc.stdout:
                    microcode_raw = line.decode("utf-8", errors="ignore").strip()
                    result = re.search(r"microcode.*: (.*)", microcode_raw)
                    if result:
                        microcode = result.group(1)

            return f"BIOS {bios_version} Microcode {microcode}"
        except Exception:
            return "n/a"
    return "n/a"


def is_turbo_boost_enabled() -> Union[str, bool]:
    """
    Check if turbo boost is enabled.

    Return True if enabled, False if disabled, None if cannot collect info.
    """
    if psutil.LINUX:
        try:
            cmd = "cat /sys/devices/system/cpu/intel_pstate/no_turbo"
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
                    raw_status = int(line.decode("utf-8", errors="ignore").strip())
                    if raw_status == 0:
                        return True
                    elif raw_status == 1:
                        return False
                    else:
                        return "n/a"
        except Exception:
            return "n/a"
    return "n/a"


def get_kernel_version() -> str:
    """Return kernel version."""
    if psutil.LINUX:
        return platform.release()
    else:
        return platform.platform()
