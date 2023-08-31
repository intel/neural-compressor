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
"""Benchmarking: measure the model performance with the objective settings."""

import argparse
import subprocess

import numpy as np

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--cores_per_instance", type=int, required=True)
parser.add_argument("--num_of_instance", type=int, required=True)
args = parser.parse_args()


def get_architecture():
    """Get the architecture name of the system."""
    p1 = subprocess.Popen("lscpu", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2 = subprocess.Popen(["grep", "Architecture"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["cut", "-d", ":", "-f2"], stdin=p2.stdout, stdout=subprocess.PIPE)
    res = None
    for line in iter(p3.stdout.readline, b""):
        res = line.decode("utf-8").strip()
    return res


def get_threads_per_core():
    """Get the threads per core."""
    p1 = subprocess.Popen("lscpu", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2 = subprocess.Popen(["grep", "Thread(s) per core"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["cut", "-d", ":", "-f2"], stdin=p2.stdout, stdout=subprocess.PIPE)
    res = None
    for line in iter(p3.stdout.readline, b""):
        res = line.decode("utf-8").strip()
    return res


def get_threads():
    """Get the list of threads."""
    p1 = subprocess.Popen(["cat", "/proc/cpuinfo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2 = subprocess.Popen(["grep", "processor"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["cut", "-d", ":", "-f2"], stdin=p2.stdout, stdout=subprocess.PIPE)
    res = []
    for line in iter(p3.stdout.readline, b""):
        res.append(line.decode("utf-8").strip())
    return res


def get_physical_ids():
    """Get the list of sockets."""
    p1 = subprocess.Popen(["cat", "/proc/cpuinfo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2 = subprocess.Popen(["grep", "physical id"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["cut", "-d", ":", "-f2"], stdin=p2.stdout, stdout=subprocess.PIPE)
    res = []
    for line in iter(p3.stdout.readline, b""):
        res.append(line.decode("utf-8").strip())
    return res


def get_core_ids():
    """Get the ids list of the cores."""
    p1 = subprocess.Popen(["cat", "/proc/cpuinfo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2 = subprocess.Popen(["grep", "core id"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["cut", "-d", ":", "-f2"], stdin=p2.stdout, stdout=subprocess.PIPE)
    res = []
    for line in iter(p3.stdout.readline, b""):
        res.append(line.decode("utf-8").strip())
    return res


def get_bounded_threads(core_ids, threads, sockets):
    """Return the threads id list that we will bind instances to."""
    res = []
    existing_socket_core_list = []
    for idx, x in enumerate(core_ids):
        socket_core = sockets[idx] + ":" + x
        if socket_core not in existing_socket_core_list:
            res.append(int(threads[idx]))
            existing_socket_core_list.append(socket_core)
    return res


def config_instance(cores_per_instance, num_of_instance):
    """Configure the multi-instance commands and trigger benchmark with sub process."""
    core = []

    if get_architecture() == "aarch64" and int(get_threads_per_core()) > 1:
        raise OSError("Currently no support on AMD with hyperthreads")
    else:
        bounded_threads = get_bounded_threads(get_core_ids(), get_threads(), get_physical_ids())

    for i in range(0, num_of_instance):
        if get_architecture() == "x86_64":
            core_list_idx = np.arange(0, cores_per_instance) + i * cores_per_instance
            core_list = np.array(bounded_threads)[core_list_idx]
        else:
            core_list = np.arange(0, cores_per_instance) + i * cores_per_instance
        core.append(core_list.tolist())

    for i in range(len(core)):
        core[i] = [str(j) for j in core[i]]
        core[i] = ",".join(core[i])

    core = ";".join(core)
    return core


if __name__ == "__main__":
    print(config_instance(args.cores_per_instance, args.num_of_instance))
