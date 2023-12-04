# -*- coding: utf-8 -*-
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
"""Tensorflow profiler utils."""

from typing import Any


def delete_assign(graph_def: Any) -> Any:
    """Modify graph nodes.

    Args:
        graph_def: TensorFlow GraphDef

    Returns:
    """
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


def create_tf_config(tf_module: Any, intra_num_of_threads: int, inter_num_of_threads: int) -> Any:
    """Create tensorflow config.

    Args:
        tf_module: tensorflow module
        intra_num_of_threads: number of threads used within an individual op for parallelism
        inter_num_of_threads: number of threads used for parallelism between independent operations

    Returns:
        TensorFlow ConfigProto object
    """
    config = tf_module.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = intra_num_of_threads
    config.inter_op_parallelism_threads = inter_num_of_threads
    return config


def set_eager_execution(input_graph: str) -> None:
    """Set eager execution as required by model.

    Args:
        input_graph: path to tensorflow model

    Returns:
        None
    """
    from neural_compressor.model.model import get_model_type

    model_type = get_model_type(input_graph)

    try:
        import tensorflow as tf

        if "keras" == model_type:
            tf.compat.v1.enable_eager_execution()
        else:
            tf.compat.v1.disable_eager_execution()
    except Exception as err:
        print(err)
