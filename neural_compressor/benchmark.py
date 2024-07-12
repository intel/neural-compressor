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
"""Benchmark is used for evaluating the model performance."""
import json
import os
import re
import signal
import subprocess
import sys
from threading import Thread

import numpy as np
import psutil

from neural_compressor.profiling.parser.factory import ParserFactory
from neural_compressor.profiling.profiler.factory import ProfilerFactory

from .adaptor import FRAMEWORKS
from .config import BenchmarkConfig, options
from .data import check_dataloader
from .model import BaseModel, Model
from .objective import MultiObjective
from .profiling.parser.parser import ProfilingParser
from .profiling.profiler.profiler import Profiler
from .utils import OPTIONS, alias_param, logger
from .utils.utility import GLOBAL_STATE, MODE, Statistics, dump_table, print_table


def set_env_var(env_var, value, overwrite_existing=False):
    """Set the specified environment variable.

    Only set new env in two cases:
    1. env not exists
    2. env already exists but overwrite_existing params set True
    """
    if overwrite_existing or not os.environ.get(env_var):
        os.environ[env_var] = str(value)


def set_all_env_var(conf, overwrite_existing=False):
    """Set all the environment variables with the configuration dict.

    Neural Compressor only uses physical cores
    """
    cpu_counts = psutil.cpu_count(logical=False)
    assert isinstance(conf, BenchmarkConfig), "input has to be a Config object"

    if conf.cores_per_instance is not None:
        assert (
            conf.cores_per_instance * conf.num_of_instance <= cpu_counts
        ), "num_of_instance * cores_per_instance should <= cpu physical cores"
    else:
        assert conf.num_of_instance <= cpu_counts, "num_of_instance should <= cpu counts"
        conf.cores_per_instance = int(cpu_counts / conf.num_of_instance)
    for var, value in dict(conf).items():
        set_env_var(var.upper(), value, overwrite_existing)


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


def run_instance(model, conf, b_dataloader=None, b_func=None):
    """Run the instance with the configuration.

    Args:
        model (object):           The model to be benchmarked.
        conf (BenchmarkConfig): The configuration for benchmark containing accuracy goal,
                                  tuning objective and preferred calibration & quantization
                                  tuning space etc.
        b_dataloader:             The dataloader for frameworks.
        b_func:                   Customized benchmark function. If user passes the dataloader,
                                  then b_func is not needed.
    """
    results = {}
    if b_func is None:
        GLOBAL_STATE.STATE = MODE.BENCHMARK
        framework_specific_info = {
            "device": conf.device,
            "approach": None,
            "random_seed": options.random_seed,
            "backend": conf.backend if conf.backend is not None else "default",
            "format": "default",
        }
        framework = conf.framework.lower()
        if "tensorflow" in framework:
            framework_specific_info.update(
                {"inputs": conf.inputs, "outputs": conf.outputs, "recipes": {}, "workspace_path": options.workspace}
            )
        if framework == "keras":
            framework_specific_info.update({"workspace_path": options.workspace})
        if framework == "mxnet":
            framework_specific_info.update({"b_dataloader": b_dataloader})
        if "onnx" in framework:
            framework_specific_info.update(
                {"workspace_path": options.workspace, "graph_optimization": OPTIONS[framework].graph_optimization}
            )
        if framework == "pytorch_ipex" or framework == "pytorch" or framework == "pytorch_fx":
            framework_specific_info.update({"workspace_path": options.workspace, "q_dataloader": None})

        assert isinstance(model, BaseModel), "need set neural_compressor Model for quantization...."

        adaptor = FRAMEWORKS[framework](framework_specific_info)

        assert b_dataloader is not None, "dataloader should not be None"

        from neural_compressor.utils.create_obj_from_config import create_eval_func

        b_func = create_eval_func(conf.framework, b_dataloader, adaptor, None, iteration=conf.iteration)

        objectives = MultiObjective(["performance"], {"relative": 0.1}, is_measure=True)

        val = objectives.evaluate(b_func, model)
        # measurer contain info not only performance(eg, memory, model_size)
        # also measurer have result list among steps
        acc, _ = val
        batch_size = b_dataloader.batch_size
        warmup = conf.warmup
        if len(objectives.objectives[0].result_list()) < warmup:
            if len(objectives.objectives[0].result_list()) > 1 and warmup != 0:
                warmup = 1
            else:
                warmup = 0

        result_list = objectives.objectives[0].result_list()[warmup:]
        latency = np.array(result_list).mean() / batch_size
        results["performance"] = acc, batch_size, result_list

        logger.info("\nbenchmark result:")
        for i, res in enumerate(result_list):
            logger.debug("Iteration {} result {}:".format(i, res))
        logger.info("Batch size = {}".format(batch_size))
        logger.info("Latency: {:.3f} ms".format(latency * 1000))
        logger.info("Throughput: {:.3f} images/sec".format(1.0 / latency))
        return results
    else:
        b_func(model.model)


def generate_prefix(core_list):
    """Generate the command prefix with numactl.

    Args:
        core_list: a list of core indexes bound with specific instances
    """
    if sys.platform in ["linux"] and os.system("numactl --show >/dev/null 2>&1") == 0:
        return "OMP_NUM_THREADS={} numactl --localalloc --physcpubind={}".format(
            len(core_list), ",".join(core_list.astype(str))
        )
    elif sys.platform in ["win32"]:  # pragma: no cover
        # (TODO) should we move the hw_info from ux?
        from neural_compressor.utils.utility import get_number_of_sockets

        num_of_socket = int(get_number_of_sockets())
        cores_per_instance = int(os.environ.get("CORES_PER_INSTANCE"))
        cores_per_socket = int(psutil.cpu_count(logical=False)) / num_of_socket
        socket_id = int(core_list[0] // cores_per_socket)
        # cores per socket should integral multiple of cores per instance, else not bind core
        if cores_per_socket % cores_per_instance == 0:
            from functools import reduce

            hex_core = hex(reduce(lambda x, y: x | y, [1 << p for p in core_list]))
            return "start /b /WAIT /node {} /affinity {} CMD /c".format(socket_id, hex_core)
    else:
        return ""


def call_one(cmd, log_file):
    """Execute one command for one instance in one thread and dump the log (for Windows)."""
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )  # nosec
    with open(log_file, "w", 1, encoding="utf-8") as log_file:
        log_file.write(f"[ COMMAND ] {cmd} \n")
        for line in proc.stdout:
            decoded_line = line.decode("utf-8", errors="ignore").strip()
            logger.info(decoded_line)  # redirect to terminal
            log_file.write(decoded_line + "\n")


def config_instance(raw_cmd):
    """Configure the multi-instance commands and trigger benchmark with sub process.

    Args:
        raw_cmd: raw command used for benchmark
    """
    multi_instance_cmd = ""
    num_of_instance = int(os.environ.get("NUM_OF_INSTANCE"))
    cores_per_instance = int(os.environ.get("CORES_PER_INSTANCE"))

    logger.info("num of instance: {}".format(num_of_instance))
    logger.info("cores per instance: {}".format(cores_per_instance))

    if sys.platform in ["linux"] and get_architecture() == "aarch64" and int(get_threads_per_core()) > 1:
        raise OSError("Currently no support on ARM with hyperthreads")
    elif sys.platform in ["linux"]:
        bounded_threads = get_bounded_threads(get_core_ids(), get_threads(), get_physical_ids())

    for i in range(0, num_of_instance):
        if sys.platform in ["linux"] and get_architecture() == "x86_64":
            core_list_idx = np.arange(0, cores_per_instance) + i * cores_per_instance
            core_list = np.array(bounded_threads)[core_list_idx]
        else:
            core_list = np.arange(0, cores_per_instance) + i * cores_per_instance
        # bind cores only allowed in linux/mac os with numactl enabled
        prefix = generate_prefix(core_list)
        instance_cmd = "{} {}".format(prefix, raw_cmd)
        if sys.platform in ["linux"]:
            instance_log = "{}_{}_{}.log".format(num_of_instance, cores_per_instance, i)
            multi_instance_cmd += "{} 2>&1|tee {} & \\\n".format(instance_cmd, instance_log)
        else:  # pragma: no cover
            multi_instance_cmd += "{} \n".format(instance_cmd)

    multi_instance_cmd += "wait" if sys.platform in ["linux"] else ""
    logger.info("Running command is\n{}".format(multi_instance_cmd))
    # each instance will execute single instance
    set_env_var("NC_ENV_CONF", True, overwrite_existing=True)
    if sys.platform in ["linux"]:
        p = subprocess.Popen(multi_instance_cmd, preexec_fn=os.setsid, shell=True)  # nosec
    elif sys.platform in ["win32"]:  # pragma: no cover
        cmd_list = multi_instance_cmd.split("\n")[:-1]
        threads = []
        for idx, cmd in enumerate(cmd_list):
            # wrap each execution of windows bat file in one thread
            # write the log to the log file of the corresponding instance
            logger.info("Will dump to {}_{}_{}.log".format(num_of_instance, cores_per_instance, idx))
            threads.append(
                Thread(target=call_one, args=(cmd, "{}_{}_{}.log".format(num_of_instance, cores_per_instance, idx)))
            )
        for command_thread in threads:
            command_thread.start()
            logger.info("Worker threads start")
        # Wait for all of them to finish
        for command_thread in threads:
            command_thread.join()
            logger.info("Worker threads join")
        return
    try:
        p.communicate()
    except KeyboardInterrupt:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)


def summary_benchmark():
    """Get the summary of the benchmark."""
    if sys.platform in ["linux"]:
        num_of_instance = int(os.environ.get("NUM_OF_INSTANCE"))
        cores_per_instance = int(os.environ.get("CORES_PER_INSTANCE"))
        latency_l = []
        throughput_l = []
        for i in range(0, num_of_instance):
            log = "{}_{}_{}.log".format(num_of_instance, cores_per_instance, i)
            with open(log, "r") as f:
                for line in f:
                    latency = re.search(r"[L,l]atency:\s+(\d+(\.\d+)?)", line)
                    latency_l.append(float(latency.group(1))) if latency and latency.group(1) else None
                    throughput = re.search(r"[T,t]hroughput:\s+(\d+(\.\d+)?)", line)
                    throughput_l.append(float(throughput.group(1))) if throughput and throughput.group(1) else None
        if throughput_l and latency_l:
            assert (
                len(latency_l) == len(throughput_l) == num_of_instance
            ), "Multiple instance benchmark failed with some instance!"

            output_data = [
                ["Latency average [second/sample]", "{:.6f}".format((sum(latency_l) / len(latency_l)) / 1000)],
                ["Throughput sum [samples/second]", "{:.3f}".format(sum(throughput_l))],
            ]
            logger.info("********************************************")
            Statistics(
                output_data, header="Multiple Instance Benchmark Summary", field_names=["Items", "Result"]
            ).print_stat()
    else:
        # (TODO) should add summary after win32 benchmark has log
        pass


def profile(model, conf, b_dataloader) -> None:
    """Execute profiling for benchmark configuration.

    Args:
        model: The model to be profiled.
        conf: The configuration for benchmark containing accuracy goal,
              tuning objective and preferred calibration & quantization
              tuning space etc.
        b_dataloader: The dataloader for frameworks.

    Returns:
        None
    """
    intra_num_of_threads = 1
    inter_num_of_threads = 1
    num_warmup = 10

    intra_num_of_threads_conf = conf.intra_num_of_threads
    if intra_num_of_threads_conf is not None:
        intra_num_of_threads = intra_num_of_threads_conf
    else:
        logger.warning(
            f"Could not find intra_num_of_threads value in config. Using: {intra_num_of_threads}",
        )

    inter_num_of_threads_conf = conf.inter_num_of_threads
    if inter_num_of_threads_conf is not None:
        inter_num_of_threads = inter_num_of_threads_conf
    else:
        logger.warning(
            f"Could not find inter_num_of_threads value in config. Using: {inter_num_of_threads}",
        )

    num_warmup_conf = conf.warmup
    if num_warmup_conf is not None:
        num_warmup = num_warmup_conf
    else:
        logger.warning(
            f"Could not get find num_warmup value in config. Using: {num_warmup}",
        )

    profiling_log = os.path.abspath(
        os.path.join(
            options.workspace,
        ),
    )
    profiler: Profiler = ProfilerFactory.get_profiler(
        model=model,
        dataloader=b_dataloader,
        log_file=profiling_log,
    )
    profiler.profile_model(
        intra_num_of_threads=intra_num_of_threads,
        inter_num_of_threads=inter_num_of_threads,
        num_warmup=num_warmup,
    )
    parser: ProfilingParser = ParserFactory.get_parser(
        model=model,
        logs=[profiling_log],
    )
    parsed_results = parser.process()
    print_table(
        title="Profiling",
        column_mapping={
            "Node name": "node_name",
            "Total execution time [us]": "total_execution_time",
            "Accelerator execution time [us]": "accelerator_execution_time",
            "CPU execution time [us]": "cpu_execution_time",
            "OP run": "op_run",
            "OP defined": "op_defined",
        },
        table_entries=parsed_results,
    )

    profiling_table_file = os.path.join(
        options.workspace,
        "profiling_table.csv",
    )

    dump_table(
        filepath=profiling_table_file,
        column_mapping={
            "Node name": "node_name",
            "Total execution time [us]": "total_execution_time",
            "Accelerator execution time [us]": "accelerator_execution_time",
            "CPU execution time [us]": "cpu_execution_time",
            "OP run": "op_run",
            "OP defined": "op_defined",
        },
        table_entries=parsed_results,
        file_type="csv",
    )

    profiling_data_file = os.path.join(
        options.workspace,
        "profiling_data.json",
    )
    with open(profiling_data_file, "w") as profiling_json:
        json.dump(parsed_results, profiling_json, indent=4)


def benchmark_with_raw_cmd(raw_cmd, conf=None):
    """Benchmark the model performance with the raw command.

    Args:
        raw_cmd (string):           The command to be benchmarked.
        conf (BenchmarkConfig): The configuration for benchmark containing accuracy goal,
                                  tuning objective and preferred calibration & quantization
                                  tuning space etc.

    Example::

        # Run benchmark according to config
        from neural_compressor.benchmark import fit_with_raw_cmd

        conf = BenchmarkConfig(iteration=100, cores_per_instance=4, num_of_instance=7)
        fit_with_raw_cmd("test.py", conf)
    """
    if conf is not None:
        if conf.backend == "ipex":
            import intel_extension_for_pytorch
        assert sys.platform in ["linux", "win32"], "only support platform windows and linux..."
        # disable multi-instance for running benchmark on GPU device
        set_all_env_var(conf)

    config_instance(raw_cmd)
    summary_benchmark()


@alias_param("conf", param_alias="config")
def fit(model, conf, b_dataloader=None, b_func=None):
    """Benchmark the model performance with the configure.

    Args:
        model (object):           The model to be benchmarked.
        conf (BenchmarkConfig): The configuration for benchmark containing accuracy goal,
                                  tuning objective and preferred calibration & quantization
                                  tuning space etc.
        b_dataloader:             The dataloader for frameworks.
        b_func:                   Customized benchmark function. If user passes the dataloader,
                                  then b_func is not needed.

    Example::

        # Run benchmark according to config
        from neural_compressor.benchmark import fit

        conf = BenchmarkConfig(iteration=100, cores_per_instance=4, num_of_instance=7)
        fit(model='./int8.pb', conf=conf, b_dataloader=eval_dataloader)
    """
    if conf.backend == "ipex":
        import intel_extension_for_pytorch

    wrapped_model = Model(model, conf=conf)

    if b_dataloader is not None:
        check_dataloader(b_dataloader)
    assert sys.platform in ["linux", "win32", "darwin"], "platform not supported..."
    # disable multi-instance for running benchmark on GPU device
    set_all_env_var(conf)
    if conf.device == "gpu" or conf.device == "npu" or sys.platform == "darwin":
        set_env_var("NC_ENV_CONF", True, overwrite_existing=True)

    logger.info("Start to run Benchmark.")
    if os.environ.get("NC_ENV_CONF") == "True":
        return run_instance(model=wrapped_model, conf=conf, b_dataloader=b_dataloader, b_func=b_func)
    raw_cmd = sys.executable + " " + " ".join(sys.argv)
    benchmark_with_raw_cmd(raw_cmd)
