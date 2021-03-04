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
import subprocess

import psutil


class HWInfo:
    """Class responsible for gathering information about platform hardware."""

    def __init__(self) -> None:
        """Initialize HW Info class and gather information platform hardware."""
        self.sockets: int = get_number_of_sockets()
        self.cores: int = psutil.cpu_count(logical=False)
        self.total_memory: float = psutil.virtual_memory().total / (1024 ** 3)
        self.system: str = get_distribution()


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
