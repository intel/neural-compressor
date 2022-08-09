# Copyright (c) 2022 Intel Corporation
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


import subprocess
import os


def get_num_cpu_cores() -> int:

    cmd_cpu_info = ''
    cmd_cpu_info += "sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')"
    cmd_cpu_info += ' && '
    cmd_cpu_info += "cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')"
    cmd_cpu_info += ' && '
    cmd_cpu_info += 'phsical_cores_num=$( echo "${sockets_num} * ${cores_per_socket}" |bc )'
    cmd_cpu_info += ' && '
    cmd_cpu_info += "numa_nodes_num=$(lscpu |grep 'NUMA node(s):' |sed 's/[^0-9]//g')"
    cmd_cpu_info += ' && '
    cmd_cpu_info += 'cores_per_node=$( echo "${phsical_cores_num} / ${numa_nodes_num}" |bc )'
    cmd_cpu_info += ' && '
    cmd_cpu_info += "echo ${cores_per_node}"
    sp_grep_cpu_info = subprocess.Popen(
        cmd_cpu_info, env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
    sp_grep_cpu_info.wait()

    log_cpu_info, _ = sp_grep_cpu_info.communicate()
    ncores = int(str(log_cpu_info)[2:-3])

    return ncores
