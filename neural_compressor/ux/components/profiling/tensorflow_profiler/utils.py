# -*- coding: utf-8 -*-
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
"""Tensorflow profiler utils."""

from typing import Any


def delete_assign(graph_def: Any) -> Any:
    """Modify graph nodes."""
    for node in graph_def.node:
        if node.op == "RefSwitch":
            node.op = "Switch"
            for index in range(len(node.input)):
                if "moving_" in node.input[index]:
                    node.input[index] = node.input[index] + "/read"
        elif node.op == "AssignAdd":
            node.op = "Add"
            if "use_locking" in node.attr:
                del node.attr["use_locking"]

        elif node.op == "AssignSub":
            node.op = "Sub"
            if "use_locking" in node.attr:
                del node.attr["use_locking"]
    return graph_def
