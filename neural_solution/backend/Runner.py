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


from backend import Cluster, TaskDB, Scheduler, TaskMonitor, ResultMonitor
import threading
import socket
import time

from backend.constant import TASK_MONITOR_PORT, RESULT_MONITOR_PORT
from backend.cluster import Node, Cluster
from backend.utils.utility import build_cluster
from backend.utils import logger

import threading
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Incserve runner automatically schedules multiple inc tasks and\
        executes multi-node distributed tuning.")

    parser.add_argument("-H", "--hostfile", type=str, default=None, \
        help="Path to the host file which contains all available nodes.")
    # ...
    return parser.parse_args(args=args)

def main(args=None):
    args = parse_args(args)

    # Initialize cluster from the host file. If there is no host file, build one local cluster.
    cluster = build_cluster(args.hostfile)

    # initialize the task db
    task_db = TaskDB()

    # start three threads
    rm = ResultMonitor(RESULT_MONITOR_PORT, task_db)
    t_rm = threading.Thread(target=rm.wait_result)

    ts = Scheduler(cluster, task_db, RESULT_MONITOR_PORT)
    t_ts = threading.Thread(target=ts.schedule_tasks)

    tm = TaskMonitor(TASK_MONITOR_PORT, task_db)
    t_tm = threading.Thread(target=tm.wait_new_task)

    t_rm.start()
    t_ts.start()
    t_tm.start()
    logger.info("task monitor port {} and result monitor port {}".format(TASK_MONITOR_PORT, RESULT_MONITOR_PORT))
    logger.info("server start...")

    t_rm.join()
    t_ts.join()
    t_tm.join()


if __name__ == '__main__':
    main()
