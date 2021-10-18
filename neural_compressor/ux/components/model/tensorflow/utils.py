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
"""TensorFlow model utils."""

try:
    import tensorflow.compat.v1 as tf_v1
except ImportError:
    import tensorflow as tf_v1

from typing import Any, List


def get_input_shape(graph_def: Any, fix_dynamic_shape: int) -> List[int]:
    """Get input shape of passed graph."""
    graph = tf_v1.Graph()
    with graph.as_default():  # pylint: disable=not-context-manager
        tf_v1.import_graph_def(graph_def, name="")
    for node in graph.as_graph_def().node:  # pylint: disable=no-member
        if node.op == "Placeholder":
            node_dict = {}
            node_dict["type"] = tf_v1.DType(node.attr["dtype"].type).name

            if node_dict["type"] != "bool":
                # convert shape to list
                try:
                    _shape = list(tf_v1.TensorShape(node.attr["shape"].shape))
                    if tf_v1.__version__ >= "2.0.0":
                        node_dict["shape"] = [
                            item if item is not None else fix_dynamic_shape for item in _shape
                        ]
                    else:
                        node_dict["shape"] = [
                            item.value if item.value is not None else fix_dynamic_shape
                            for item in _shape
                        ]
                    # if shape dimension > 1, suppose first dimension is batch-size
                    if len(node_dict["shape"]) > 1:
                        node_dict["shape"] = node_dict["shape"][1:]
                except ValueError:
                    _shape = [fix_dynamic_shape, fix_dynamic_shape, 3]
                    node_dict["shape"] = _shape

            else:  # deal with bool dtype inputs, now assign bool dtype input False value
                node_dict["shape"] = None
                node_dict["value"] = False

    return node_dict["shape"]
