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
"""Neural Solution utility."""

import json
import os


def get_db_path(workspace="./"):
    """Get the database path.

    Args:
        workspace (str, optional): . Defaults to "./".

    Returns:
        str: the path of database
    """
    db_path = os.path.join(workspace, "db", "task.db")
    return os.path.abspath(db_path)


def get_task_workspace(workspace="./"):
    """Get the workspace of task.

    Args:
        workspace (str, optional): the workspace for Neural Solution. Defaults to "./".

    Returns:
        str: the workspace of task
    """
    return os.path.join(workspace, "task_workspace")


def get_task_log_workspace(workspace="./"):
    """Get the log workspace for task.

    Args:
        workspace (str, optional): the workspace for Neural Solution. Defaults to "./".

    Returns:
        str: the workspace of task.
    """
    return os.path.join(workspace, "task_log")


def get_serve_log_workspace(workspace="./"):
    """Get log workspace for service.

    Args:
        workspace (str, optional): the workspace for Neural Solution. Defaults to "./".

    Returns:
        str: the log workspace for service
    """
    return os.path.join(workspace, "serve_log")


def dict_to_str(d):
    """Convert a dict object to a string object.

    Args:
        d (dict): a dict object

    Returns:
        str: string
    """
    result = json.dumps(d)
    return result
