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


import time
import threading
import subprocess
import os
import socket
import re
import json
import glob
import shutil
from .task import Task
from .cluster import Cluster
from .task_db import TaskDB
from neural_solution.config import NUM_THREADS_PER_PROCESS, INC_ENV_PATH_TEMP
from .utils.utility import (
    serialize,
    dump_elapsed_time,
    get_task_log_path,
    get_current_time,
    build_workspace,
    is_remote_url
)

from neural_solution.utility import get_task_log_workspace, get_task_workspace
from .utils import logger

cmd="echo $(conda info --base)/etc/profile.d/conda.sh"
CONDA_SOURCE_PATH = subprocess.getoutput(cmd)

class Scheduler:
    def __init__(self, cluster: Cluster, task_db: TaskDB, result_monitor_port, \
        conda_env_name=None, upload_path="./examples", config=None):
        """Scheduler dispatches the task with the available resources, calls the mpi command and report results.

        Attributes:
            cluster: the Cluster object that manages the server resources
            task_db: the TaskDb object that manages the tasks
            result_monitor_port: The result monitor port to report the accuracy and performance result
            conda_env_name: The basic environment for task execution
            upload_path： Custom example path.
        """
        self.cluster = cluster
        self.task_db = task_db
        self.result_monitor_port = result_monitor_port
        self.conda_env_name = conda_env_name
        self.upload_path = upload_path
        self.config = config

    def prepare_env(self, task:Task):
        """Check and create a conda environment.
        If the required packages are not installed in the conda environment,
        create a new conda environment and install the required packages.

        Args:
            task (Task): _description_
        """
        # Define the prefix of the conda environment name
        env_prefix = self.conda_env_name
        requirement = task.requirement.split(" ")
        # Skip check when requirement is empty.
        if requirement == ['']:
            return env_prefix
        # Construct the command to list all the conda environments
        cmd = f"conda env list"
        output = subprocess.getoutput(cmd)
        # Parse the output to get a list of conda environment names
        env_list = [line.strip().split()[0] for line in output.splitlines()[2:]]
        conda_env = None
        for env_name in env_list:
            # Only check the conda environments that start with the specified prefix
            if env_name.startswith(env_prefix):
                conda_bash_cmd = f"source {CONDA_SOURCE_PATH}"
                cmd = f"{conda_bash_cmd} && conda activate {env_name} && conda list"
                output = subprocess.getoutput(cmd)
                # Parse the output to get a list of installed package names
                installed_packages = [line.split()[0] for line in output.splitlines()[2:]]
                installed_packages_version = [line.split()[0] + "=" + line.split()[1] for line in output.splitlines()[2:]]
                missing_packages = set(requirement) - set(installed_packages) - set(installed_packages_version)
                if not missing_packages:
                    conda_env = env_name
                    break
        if conda_env is None:
            # Construct the command to create a new conda environment and install the required packages
            from datetime import datetime
            now = datetime.now()
            suffix = now.strftime("%Y%m%d-%H%M%S")
            conda_env = f"{env_prefix}_{suffix}"
            # Construct the name of the new conda environment
            cmd = (f"source {CONDA_SOURCE_PATH} && conda create -n {conda_env} --clone {env_prefix}"
                f" && conda activate {conda_env} && pip install {task.requirement.replace('=','==')}")
            p = subprocess.Popen(cmd, shell=True)
            logger.info(f"[Scheduler] Creating new environment {conda_env} start.")
            p.wait()
            logger.info(f"[Scheduler] Creating new environment {conda_env} end.")
        return conda_env

    def prepare_task(self, task:Task):
        """prepare workspace and download run_task.py for task

        Args:
            task (Task): _description_
        """
        self.task_path = build_workspace(path=get_task_workspace(self.config.workspace), task_id=task.task_id)
        logger.info(f"****TASK PATH: {self.task_path}")
        if is_remote_url(task.script_url):
            task_url = task.script_url.replace('github.com', 'raw.githubusercontent.com').replace('blob','')
            try:
                subprocess.check_call(['wget', '-P', self.task_path, task_url])
            except subprocess.CalledProcessError as e:
                logger.info("Failed: {}".format(e.cmd))
        else:
            # Assuming the file is uploaded in directory examples
            example_path = os.path.abspath(os.path.join(self.upload_path, task.script_url))
            # only one python file
            script_path = glob.glob(os.path.join(example_path, '*.py'))[0]
            # script_path = glob.glob(os.path.join(example_path, f'*{extension}'))[0]
            self.script_name = script_path.split("/")[-1]
            shutil.copy(script_path, os.path.abspath(self.task_path))
            task.arguments = task.arguments.replace("=dataset", "=" + os.path.join(example_path,"dataset")).\
                replace("=model", "=" + os.path.join(example_path,"model"))
        if not task.optimized:
            # Generate quantization code with Neural Coder API
            neural_coder_cmd = ['python -m neural_coder --enable --approach']
            # for users to define approach: "static, ""static_ipex", "dynamic", "auto"
            approach = task.approach
            neural_coder_cmd.append(approach)
            if is_remote_url(task.script_url):
                self.script_name = task.script_url.split("/")[-1]
            neural_coder_cmd.append(self.script_name)
            neural_coder_cmd = " ".join(neural_coder_cmd)
            full_cmd = """cd {}\n{}""".format(self.task_path, neural_coder_cmd)
            p = subprocess.Popen(full_cmd, shell=True)
            logger.info("[Neural Coder] Generating optimized code start.")
            p.wait()
            logger.info("[Neural Coder] Generating optimized code end.")

    def check_task_status(self, log_path):
        for line in reversed(open(log_path).readlines()):
            res_pattern = r'[INFO] Save deploy yaml to'
            # res_matches = re.findall(res_pattern, line)
            if res_pattern in line:
                return "done"
        return "failed"

    def _parse_cmd(self, task: Task, resource):
        # TODO study mpi bind on socket
        # mpirun -np 3 -mca btl_tcp_if_include 192.168.20.0/24 -x OMP_NUM_THREADS=80
        # --host mlt-skx091,mlt-skx050,mlt-skx053 bash run_distributed_tuning.sh
        self.prepare_task(task)
        conda_env = self.prepare_env(task)
        host_str = ",".join([item.split(" ")[1] for item in resource])
        logger.info(f"[TaskScheduler] host resource: {host_str}")

        # Activate environment
        # TODO enhance conda bash cmd
        conda_bash_cmd = f"source {CONDA_SOURCE_PATH}"
        conda_env_cmd = f"conda activate {conda_env}"
        if INC_ENV_PATH_TEMP:
            conda_env_cmd = f"{conda_env_cmd}\nexport PYTHONPATH=$PYTHONPATH:{INC_ENV_PATH_TEMP}"
        mpi_cmd = ["mpirun",
                "-np",
                "{}".format(task.workers),
                "-host",
                "{}".format(host_str),
                "-map-by",
                "socket:pe={}".format(NUM_THREADS_PER_PROCESS),
                "-mca",
                "btl_tcp_if_include",
                "192.168.20.0/24", # TODO replace it according to the node
                "-x",
                "OMP_NUM_THREADS={}".format(NUM_THREADS_PER_PROCESS),
                '--report-bindings'
                ]
        mpi_cmd = " ".join(mpi_cmd)

        # Initial Task command
        task_cmd = ['python']
        task_cmd.append(self.script_name)
        task_cmd.append(self.sanitize_arguments(task.arguments))
        # TODO add q_model_path into arguments
        self.q_model_path = self.task_path + "/" + "q_model_path"
        if not os.path.exists(self.q_model_path):
            os.makedirs(self.q_model_path)
        # task_cmd.append("--output_path {}".format(self.q_model_path)) # TODO save q_model
        self.q_model_path = os.path.abspath(self.q_model_path)
        task_cmd = " ".join(task_cmd)

        # use optimized code by Neural Coder
        if not task.optimized:
            task_cmd = task_cmd.replace(".py", "_optimized.py")

        # build a bash script to run task.
        bash_script_name = "distributed_run.sh" if task.workers > 1 else "run.sh"
        bash_script = """{}\n{}\ncd {}\n{}""".format(conda_bash_cmd, conda_env_cmd, \
            self.task_path, task_cmd)
        bash_script_path = os.path.join(self.task_path, bash_script_name)
        with open(bash_script_path, 'w', encoding="utf-8") as f:
            f.write(bash_script)
        full_cmd = """cd {}\n{} bash {}""".format(self.task_path, mpi_cmd, bash_script_name)

        return full_cmd

    def report_result(self, task_id, log_path, task_runtime):
        """Report the result to the result monitor."""
        s = socket.socket()
        s.connect(("localhost", self.result_monitor_port))
        # TODO parse the log file, need append more pattern
        results = {"optimization time (seconds)" : "{:.2f}".format(task_runtime)}
        for line in reversed(open(log_path).readlines()):
            res_pattern = r'Tune (\d+) result is:\s.*?\(int8\|fp32\):\s+(\d+\.\d+).*?\(int8\|fp32\):\s+(\d+\.\d+).*?'
            res_matches = re.findall(res_pattern, line)
            if res_matches:
                # results["Tuning count"] = res_matches[0][0]
                results["Accuracy"] = res_matches[0][1]
                results["Duration (seconds)"] = res_matches[0][2]
                # break when the last result is matched
                break

        results = json.dumps(results)

        s.send(serialize({"task_id": task_id, "result": results, "q_model_path": self.q_model_path}))
        s.close()


    @dump_elapsed_time("Task execution")
    def launch_task(self, task: Task, resource):
        """Generate the mpi command and execute the task.

        Redirect the log to ./TASK_LOG_PATH/task_<id>/txt
        """
        full_cmd = self._parse_cmd(task, resource)
        logger.info(f"[TaskScheduler] Parsed the command from task: {full_cmd}")
        log_path = get_task_log_path(log_path=get_task_log_workspace(self.config.workspace), task_id=task.task_id)
        p = subprocess.Popen(full_cmd, stdout=open(log_path, 'w+'), stderr=subprocess.STDOUT, shell=True)
        logger.info(f"[TaskScheduler] Start run task {task.task_id}, dump log into {log_path}")
        start_time = time.time()
        p.wait()
        self.cluster.free_resource(resource)
        task_runtime = time.time() - start_time
        logger.info(f"[TaskScheduler] Finished task {task.task_id}, and free resource {resource}, dump log into {log_path}")
        task_status = self.check_task_status(log_path)
        self.task_db.update_task_status(task.task_id, task_status)
        self.report_result(task.task_id, log_path, task_runtime)

    def dispatch_task(self, task, resource):
        """Dispatch the task in a thread."""
        t = threading.Thread(target=self.launch_task, args=(task, resource))
        t.start()

    def schedule_tasks(self):
        """After each 5 seconds, check the task queue and try to schedule a task."""
        while True:
            time.sleep(5)
            logger.info(f"[TaskScheduler {get_current_time()}] try to dispatch a task...")
            if self.task_db.get_pending_task_num() > 0:
                logger.info(f"[TaskScheduler {get_current_time()}], there are {self.task_db.get_pending_task_num()} task pending.")
                task_id = self.task_db.task_queue[0]
                task = self.task_db.get_task_by_id(task_id)
                resource = self.cluster.reserve_resource(task)
                if resource:
                    self.task_db.task_queue.popleft()
                    self.task_db.update_task_status(task.task_id, "running")
                    self.dispatch_task(task, resource)
                else:
                    logger.info("[TaskScheduler] no enough node resources!")
            else:
                logger.info("[TaskScheduler] no requests in the deque!")

    def sanitize_arguments(self, arguments: str):
        return arguments.replace(u'\xa0', u' ')