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
from typing import Union

import cpuinfo
import psutil

from lpot.ux.utils.logger import log
from lpot.ux.utils.utils import determine_ip


class HWInfo:
    """Class responsible for gathering information about platform hardware."""

    def __init__(self) -> None:
        """Initialize HW Info class and gather information platform hardware."""
        cpu_info = cpuinfo.get_cpu_info()
        self.sockets: int = get_number_of_sockets()
        self.cores: int = psutil.cpu_count(logical=False)
        self.cores_per_socket: int = int(psutil.cpu_count(logical=False) / self.sockets)
        self.threads_per_socket: int = int(psutil.cpu_count(logical=True) / self.sockets)
        self.total_memory: str = f"{psutil.virtual_memory().total / (1024 ** 3):.3f}GB"
        self.system: str = get_distribution()
        self.ip = determine_ip()
        if cpu_info.get("brand"):
            self.platform = cpu_info.get("brand")
        elif cpu_info.get("brand_raw"):
            self.platform = cpu_info.get("brand_raw")
        else:
            self.platform = ""
        self.hyperthreading_enabled: bool = psutil.cpu_count(logical=False) != psutil.cpu_count(
            logical=True,
        )
        self.turboboost_enabled = is_turbo_boost_enabled()
        self.bios_version = get_bios_version()
        self.kernel: str = get_kernel_version()
        try:
            min_cpu_freq = int(psutil.cpu_freq(percpu=False).min)
            max_cpu_freq = int(psutil.cpu_freq(percpu=False).max)
        except Exception:
            log.warning("Cannot collect cpu frequency information.")
            min_cpu_freq = 0
            max_cpu_freq = 0
        finally:
            if min_cpu_freq == 0:
                self.min_cpu_freq = "n/a"
            else:
                self.min_cpu_freq = f"{min_cpu_freq}Hz"
            if max_cpu_freq == 0:
                self.max_cpu_freq = "n/a"
            else:
                self.max_cpu_freq = f"{max_cpu_freq}Hz"


def get_number_of_sockets() -> int:
    """Get number of sockets in platform."""
    cmd = "lscpu | grep 'Socket(s)' | cut -d ':' -f 2"

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
    if psutil.WINDOWS:
        return f"{platform.system()} {platform.release()}"
    elif psutil.LINUX:
        try:
            return " ".join(platform.dist())
        except AttributeError:
            return f"{platform.system()} {platform.release()}"
    else:
        return platform.platform()


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
            if proc.stdout and proc.returncode == 0:
                for line in proc.stdout:
                    bios_version = line.decode("utf-8", errors="ignore").strip()
            else:
                bios_version = "n/a"

            cmd = "grep microcode  /proc/cpuinfo -m1"
            proc = subprocess.Popen(
                args=cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
            )
            proc.wait()
            if proc.stdout and proc.returncode == 0:
                for line in proc.stdout:
                    microcode_raw = line.decode("utf-8", errors="ignore").strip()
                    result = re.search(r"microcode.*: (.*)", microcode_raw)
                    if result:
                        microcode = result.group(1)
            else:
                microcode = "n/a"
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
