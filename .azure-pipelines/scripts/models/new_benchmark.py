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

import subprocess


def get_architecture():
    """Get the architecture name of the system."""
    p1 = subprocess.Popen("lscpu", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2 = subprocess.Popen(["grep", "Architecture"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["cut", "-d", ":", "-f2"], stdin=p2.stdout, stdout=subprocess.PIPE)
    res = None
    for line in iter(p3.stdout.readline, b''):
        res = line.decode("utf-8").strip()
    return res


def get_threads_per_core():
    """Get the threads per core."""
    p1 = subprocess.Popen("lscpu", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2 = subprocess.Popen(["grep", "Thread(s) per core"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["cut", "-d", ":", "-f2"], stdin=p2.stdout, stdout=subprocess.PIPE)
    res = None
    for line in iter(p3.stdout.readline, b''):
        res = line.decode("utf-8").strip()
    return res


def get_threads():
    """Get the list of threads."""
    p1 = subprocess.Popen(["cat", "/proc/cpuinfo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2 = subprocess.Popen(["grep", "processor"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["cut", "-d", ":", "-f2"], stdin=p2.stdout, stdout=subprocess.PIPE)
    res = []
    for line in iter(p3.stdout.readline, b''):
        res.append(line.decode("utf-8").strip())
    return res


def get_physical_ids():
    """Get the list of sockets."""
    p1 = subprocess.Popen(["cat", "/proc/cpuinfo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2 = subprocess.Popen(["grep", "physical id"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["cut", "-d", ":", "-f2"], stdin=p2.stdout, stdout=subprocess.PIPE)
    res = []
    for line in iter(p3.stdout.readline, b''):
        res.append(line.decode("utf-8").strip())
    return res


def get_core_ids():
    """Get the ids list of the cores."""
    p1 = subprocess.Popen(["cat", "/proc/cpuinfo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p2 = subprocess.Popen(["grep", "core id"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["cut", "-d", ":", "-f2"], stdin=p2.stdout, stdout=subprocess.PIPE)
    res = []
    for line in iter(p3.stdout.readline, b''):
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


def config_instance():
    if (get_architecture() == 'aarch64' and int(get_threads_per_core()) > 1):
        raise OSError('Currently no support on AMD with hyperthreads')
    else:
        bounded_threads = get_bounded_threads(get_core_ids(), get_threads(), get_physical_ids())
    return ",".join([str(i) for i in bounded_threads])

if __name__ == "__main__":
    print(config_instance())
    # print(','.join([1,2,3]))
