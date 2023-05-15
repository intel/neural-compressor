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

#Constant for server
TASK_MONITOR_PORT = 2222
RESULT_MONITOR_PORT = 3333
SERVE_PORT = 8000


NEURAL_SOLUTION_WORKSPACE = "./ns_workspace"
DB_PATH = NEURAL_SOLUTION_WORKSPACE + "/db"
TASK_WORKSPACE =  NEURAL_SOLUTION_WORKSPACE + "/task_workspace"
TASK_LOG_path = NEURAL_SOLUTION_WORKSPACE + "/task_log"
SERVE_LOG_PATH = NEURAL_SOLUTION_WORKSPACE + "/serve_log"

#Constant for execute MPI task
NUM_THREADS_PER_PROCESS = 5
NUM_CORES_PER_SOCKET = 5 # TODO replace it according the node
NUM_SOCKETS = 2

INTERVAL_TIME_BETWEEN_DISPATCH_TASK = 3

#Constant for conda # TODO remove it
CONDA_ENV_NAME = "inc"
INC_ENV_PATH_TEMP = "YOURPATH/neural-compressor" # TODO detect it automatically