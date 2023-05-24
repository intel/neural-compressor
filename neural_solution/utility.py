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

import os
import json

def get_db_path(workspace="./"):
    db_path = os.path.join(workspace, "db", "task.db")
    return os.path.abspath(db_path)

def get_task_workspace(workspace="./"):
    return os.path.join(workspace, "task_workspace")

def get_task_log_workspace(workspace="./"):
    return os.path.join(workspace, "task_log")

def get_serve_log_workspace(workspace="./"):
    return os.path.join(workspace, "serve_log")

def dict_to_str(d):
    result = json.dumps(d)
    return result