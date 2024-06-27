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

import argparse
import os
import re
import subprocess
import sys

import psutil

from neural_compressor.common.utils import Statistics, logger

description = """
##################################################################################################################
This is the command used to launch the Intel CPU performance benchmark, supports both Linux and Windows platform.
To get the peak performance on Intel Xeon CPU, we should avoid crossing NUMA node in one instance.
By default, `incbench` will trigger 1 instance on the first NUMA node.

Params in `incbench`:
 - num_instances            Default to 1.
 - num_cores_per_instance   Default to None.
 - C, cores                 Default to 0-${num_cores_on_NUMA-1}, decides the visible core range.
 - cross_memory             Default to False, decides whether to allocate memory cross NUMA.
                                Note: Use it only when memory for instance is not enough.

# General use cases:
1. `incbench main.py`: run 1 instance on NUMA:0.
2. `incbench --num_i 2 main.py`: run 2 instances on NUMA:0.
3. `incbench --num_c 2 main.py`: run multi-instances with 2 cores per instance on NUMA:0.
4. `incbench -C 24-47 main.py`: run 1 instance on COREs:24-47.
5. `incbench -C 24-47 --num_c 4 main.py`: run multi-instances with 4 COREs per instance on COREs:24-47.

Note:
    - `num_i` works the same as `num_instances`
    - `num_c` works the same as `num_cores_per_instance`
##################################################################################################################
"""


def get_linux_numa_info():
    """Collect numa/socket information on linux system.

    Returns:
        numa_info (dict):   demo: {numa_index: {"physical_cpus": "xxx"; "logical_cpus": "xxx"}}
                            E.g.    numa_info = {
                                        0: {"physical_cpus": "0-23", "logical_cpus": "0-23,48-71"},
                                        1: {"physical_cpus": "24-47", "logical_cpus": "24-47,72-95"}
                                    }
    """
    result = subprocess.run(["lscpu"], capture_output=True, text=True)
    output = result.stdout

    numa_info = {}
    for line in output.splitlines():
        # demo: "NUMA node0 CPU(s): 0-3"
        node_match = re.match(r"^NUMA node(\d+) CPU\(s\):\s+(.*)$", line)
        if node_match:
            node_id = int(node_match.group(1))
            cpus = node_match.group(2).strip()
            numa_info[node_id] = {
                "physical_cpus": cpus.split(",")[0],
                "logical_cpus": ",".join(cpus.split(",")),
            }

    # if numa_info is not collected, we go back to socket_info
    if not numa_info:
        for line in output.splitlines():
            # demo: "Socket(s):             2"
            socket_match = re.match(r"^Socket\(s\):\s+(.*)$", line)
            if socket_match:
                num_socket = int(socket_match.group(1))
        # process big cores (w/ physical cores) and small cores (w/o physical cores)
        physical_cpus = psutil.cpu_count(logical=False)
        logical_cpus = psutil.cpu_count(logical=True)
        physical_cpus_per_socket = physical_cpus // num_socket
        logical_cpus_per_socket = logical_cpus // num_socket
        for i in range(num_socket):
            physical_cpus_str = str(i * physical_cpus_per_socket) + "-" + str((i + 1) * physical_cpus_per_socket - 1)
            if num_socket == 1:
                logical_cpus_str = str(i * logical_cpus_per_socket) + "-" + str((i + 1) * logical_cpus_per_socket - 1)
            else:
                remain_cpus = logical_cpus_per_socket - physical_cpus_per_socket
                logical_cpus_str = (
                    physical_cpus_str
                    + ","
                    + str(i * (remain_cpus) + physical_cpus)
                    + "-"
                    + str((i + 1) * remain_cpus + physical_cpus - 1)
                )
            numa_info[i] = {
                "physical_cpus": physical_cpus_str,
                "logical_cpus": logical_cpus_str,
            }
    return numa_info


def get_windows_numa_info():
    """Collect socket information on Windows system due to no available numa info.

    Returns:
        numa_info (dict):   demo: {numa_index: {"physical_cpus": "xxx"; "logical_cpus": "xxx"}}
                            E.g.    numa_info = {
                                        0: {"physical_cpus": "0-23", "logical_cpus": "0-23,48-71"},
                                        1: {"physical_cpus": "24-47", "logical_cpus": "24-47,72-95"}
                                    }
    """
    import wmi

    c = wmi.WMI()
    processors = c.Win32_Processor()
    socket_designations = set()
    for processor in processors:
        socket_designations.add(processor.SocketDesignation)
    num_socket = len(socket_designations)
    physical_cpus = sum(processor.NumberOfCores for processor in processors)
    logical_cpus = sum(processor.NumberOfLogicalProcessors for processor in processors)
    physical_cpus_per_socket = physical_cpus // num_socket
    logical_cpus_per_socket = logical_cpus // num_socket

    numa_info = {}
    for i in range(num_socket):
        physical_cpus_str = str(i * physical_cpus_per_socket) + "-" + str((i + 1) * physical_cpus_per_socket - 1)
        if num_socket == 1:
            logical_cpus_str = str(i * logical_cpus_per_socket) + "-" + str((i + 1) * logical_cpus_per_socket - 1)
        else:
            remain_cpus = logical_cpus_per_socket - physical_cpus_per_socket
            logical_cpus_str = (
                physical_cpus_str
                + ","
                + str(i * (remain_cpus) + physical_cpus)
                + "-"
                + str((i + 1) * remain_cpus + physical_cpus - 1)
            )
        numa_info[i] = {
            "physical_cpus": physical_cpus_str,
            "logical_cpus": logical_cpus_str,
        }
    return numa_info


def dump_numa_info():
    """Fetch NUMA info and dump stats in shell, return numa_info.

    Returns:
        numa_info (dict): {numa_node_index: list of Physical CPUs in this numa node, ...}
    """
    if psutil.WINDOWS:
        numa_info = get_windows_numa_info()
    elif psutil.LINUX:
        numa_info = get_linux_numa_info()
    else:
        logger.error(f"Unsupported platform detected: {sys.platform}, only supported on Linux and Windows")

    # dump stats to shell
    field_names = ["NUMA node", "physical CPUs", "logical CPUs"]
    output_data = []
    for op_type in numa_info.keys():
        field_results = [op_type, numa_info[op_type]["physical_cpus"], numa_info[op_type]["logical_cpus"]]
        output_data.append(field_results)
    Statistics(output_data, header="CPU Information", field_names=field_names).print_stat()

    # parse numa_info for ease-of-use
    for n in numa_info:
        numa_info[n] = parse_str2list(numa_info[n]["physical_cpus"])
    return numa_info


def parse_str2list(cpu_ranges):
    """Parse '0-4,7,8' into [0,1,2,3,4,7,8] for machine readable."""
    cpus = []
    ranges = cpu_ranges.split(",")
    for r in ranges:
        if "-" in r:
            try:
                start, end = r.split("-")
                cpus.extend(range(int(start), int(end) + 1))
            except ValueError:
                raise ValueError(f"Invalid range: {r}")
        else:
            try:
                cpus.append(int(r))
            except ValueError:
                raise ValueError(f"Invalid number: {r}")
    return cpus


def format_list2str(cpus):
    """Format [0,1,2,3,4,7,8] back to '0-4,7,8' for human readable."""
    if not cpus:
        return ""
    cpus = sorted(set(cpus))
    ranges = []
    start = cpus[0]
    end = start
    for i in range(1, len(cpus)):
        if cpus[i] == end + 1:
            end = cpus[i]
        else:
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")
            start = cpus[i]
            end = start
    if start == end:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{end}")
    return ",".join(ranges)


def get_reversed_numa_info(numa_info):
    """Reverse numa_info."""
    reversed_numa_info = {}
    for n, cpu_info in numa_info.items():
        for i in cpu_info:
            reversed_numa_info[i] = n
    return reversed_numa_info


def get_numa_node(core_list, reversed_numa_info):
    """Return numa node used in current core_list."""
    numa_set = set()
    for c in core_list:
        assert c in reversed_numa_info, "Cores should be in physical CPUs"
        numa_set.add(reversed_numa_info[c])
    return numa_set


def set_cores_for_instance(args, numa_info):
    """All use cases are listed below:
        Params: a=num_instance; b=num_cores_per_instance; c=cores;
            - no a, b, c: a=1, c=numa:0
            - no a, b: a=1, c=c
            - no a, c: a=numa:0/b, c=numa:0
            - no b, c: a=a, c=numa:0
            - no a: a=numa:0/b, c=c
            - no b: a=a, c=c
            - no c: a=a, c=a*b
            - a, b, c: a=a, c=a*b

    Args:
        args (argparse): arguments for setting different configurations
        numa_info (dict): {numa_node_index: list of Physical CPUs in this numa node, ...}

    Returns:
        core_list_per_instance (dict): {"instance_index": ["node_index", "cpu_index", num_cpu]}
    """
    available_cores_list = []
    for n in numa_info:
        available_cores_list.extend(numa_info[n])
    # preprocess args.cores to set default values
    if args.cores is None:
        if args.num_cores_per_instance and args.num_instances:
            target_cores = args.num_instances * args.num_cores_per_instance
            assert target_cores <= len(
                available_cores_list
            ), f"num_instances * num_cores_per_instance = {target_cores} exceeds the range of physical CPUs:{len(available_cores_list)}"
            cores_list = list(range(target_cores))
        else:
            # default behavior, only use numa:0
            cores_list = numa_info[0]
    else:
        cores_list = parse_str2list(args.cores)
        if args.num_cores_per_instance and args.num_instances:
            target_cores = args.num_instances * args.num_cores_per_instance
            assert target_cores <= len(
                cores_list
            ), f"num_instances * num_cores_per_instance = {target_cores} exceeds the range of available CPUs:{len(cores_list)}"
            cores_list = cores_list[:target_cores]
    # preprocess args.num_instances to set default values
    if args.num_instances is None:
        if args.num_cores_per_instance:
            args.num_instances = len(cores_list) // args.num_cores_per_instance
        else:
            args.num_instances = 1

    # only need to process num_cores_per_instance now
    core_list_per_instance = {}
    # num_cores_per_instance = all_cores / num_instances
    num_cores_per_instance = len(cores_list) // args.num_instances
    for i in range(args.num_instances):
        core_list_per_instance[i] = cores_list[i * num_cores_per_instance : (i + 1) * num_cores_per_instance]
    if len(cores_list) % args.num_instances != 0:  # pragma: no cover
        last_index = args.num_instances - 1
        core_list_per_instance[last_index] = cores_list[last_index * num_cores_per_instance :]

    # convert core_list_per_instance = {"instance_index": cpu_index_list} -> {"instance_index": ["node_index", "cpu_index", num_cpu]}
    reversed_numa_info = get_reversed_numa_info(numa_info)
    for i, core_list in core_list_per_instance.items():
        core_list_per_instance[i] = [
            format_list2str(get_numa_node(core_list, reversed_numa_info)),
            format_list2str(core_list),
            len(core_list),
        ]

    # dump stats to shell
    field_names = ["Instance", "NUMA node", "physical CPUs"]
    output_data = []
    for i, core_list in core_list_per_instance.items():
        field_results = [i + 1, core_list[0], core_list[1]]
        output_data.append(field_results)
    Statistics(output_data, header="Instance Binding Information", field_names=field_names).print_stat()
    return core_list_per_instance


def generate_prefix(args, core_list):
    """Generate the command prefix with `numactl` (Linux) or `start` (Windows) command.

    Args:
        args (argparse): arguments for setting different configurations
        core_list: ["node_index", "cpu_index", num_cpu]
    Returns:
        command_prefix (str): command_prefix with specific core list for Linux or Windows.
    """
    if sys.platform in ["linux"] and os.system("numactl --show >/dev/null 2>&1") == 0:
        if args.cross_memory:
            return "OMP_NUM_THREADS={} numactl -l -C {}".format(core_list[2], core_list[1])
        else:
            return "OMP_NUM_THREADS={} numactl -m {} -C {}".format(core_list[2], core_list[0], core_list[1])
    elif sys.platform in ["win32"]:  # pragma: no cover
        socket_id = core_list[0]
        from functools import reduce

        hex_core = hex(reduce(lambda x, y: x | y, [1 << p for p in parse_str2list(core_list[1])]))
        return "start /B /WAIT /node {} /affinity {}".format(socket_id, hex_core)
    else:
        return ""


def run_multi_instance_command(args, core_list_per_instance, raw_cmd):
    """Build and trigger commands for multi-instances with subprocess.

    Args:
        args (argparse): arguments for setting different configurations
        core_list_per_instance (dict): {"instance_index": ["node_index", "cpu_index", num_cpu]}
        raw_cmd (str): script.py and parameters for this script
    """
    instance_cmd = ""
    if not os.getenv("PYTHON_PATH"):  # pragma: no cover
        logger.info("The interpreter path is not set, using `python` command.")
    interpreter = os.getenv("PYTHON_PATH", "python")
    current_work_dir = os.getcwd()
    logfile_process_map = {}
    for i, core_list in core_list_per_instance.items():
        # build cmd and log file path
        prefix = generate_prefix(args, core_list)
        instance_cmd = "{} {} {}".format(prefix, interpreter, raw_cmd)
        logger.info(f"Instance {i+1}: {instance_cmd}")
        instance_log_file = "{}_{}_{}C.log".format(i + 1, len(core_list_per_instance), core_list[2])
        instance_log_file = os.path.join(current_work_dir, instance_log_file)
        # trigger subprocess
        p = subprocess.Popen(
            instance_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
        )  # nosec
        # log_file_path: [process_object, instance_command, instance_index]
        logfile_process_map[instance_log_file] = [p, instance_cmd, i + 1]

    # Dump each instance's standard output to the corresponding log file
    for instance_log_file, p_cmd_i in logfile_process_map.items():
        with open(instance_log_file, "w", 1, encoding="utf-8") as log_file:
            log_file.write(f"[COMMAND]: {p_cmd_i[1]}\n")
            for line in p_cmd_i[0].stdout:
                decoded_line = line.decode("utf-8", errors="ignore").strip()
                log_file.write(decoded_line + "\n")
        logger.info(f"The log of instance {p_cmd_i[2]} is saved to {instance_log_file}")

    p.communicate()


def benchmark():
    """Benchmark API interface."""
    logger.info("Start benchmark with Intel Neural Compressor.")
    logger.info("By default, Intel Neural Compressor triggers only one instance on numa:0.")

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--num_instances", type=int, default=None, help="Determine the number of instances.")
    parser.add_argument(
        "--num_cores_per_instance", type=int, default=None, help="Determine the number of cores in 1 instance."
    )
    parser.add_argument("-C", "--cores", type=str, default=None, help="Determine the visible core range.")
    parser.add_argument("--cross_memory", action="store_true", help="Determine the visible core range.")
    parser.add_argument("script", type=str, help="The path to the script to launch.")
    parser.add_argument("parameters", nargs=argparse.REMAINDER, help="arguments to the script.")

    args = parser.parse_args()

    assert sys.platform in ["linux", "win32"], "only support platform windows and linux..."

    numa_info = dump_numa_info()  # show numa info and current usage of cores
    logger.info("Intel Neural Compressor only uses physical CPUs for the best performance.")
    core_list_per_instance = set_cores_for_instance(args, numa_info=numa_info)
    script_and_parameters = args.script + " " + " ".join(args.parameters)
    run_multi_instance_command(args, core_list_per_instance, raw_cmd=script_and_parameters)
