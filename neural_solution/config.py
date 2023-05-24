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

"""Config for both frontend and backend."""

#Constant for execute MPI task
NUM_THREADS_PER_PROCESS = 5
NUM_CORES_PER_SOCKET = 5 # TODO replace it according the node
NUM_SOCKETS = 2

INTERVAL_TIME_BETWEEN_DISPATCH_TASK = 3

#Constant for conda # TODO remove it
INC_ENV_PATH_TEMP = "YOURPATH/neural-compressor" # TODO detect it automatically

class Config:
    workspace: str = "./ns_workspace"
    task_monitor_port: int = 2222
    result_monitor_port: int = 3333
    service_address: str = "localhost"
    grpc_api_port: int = 4444
    #TODO add set and get methods for each attribute

config = Config()