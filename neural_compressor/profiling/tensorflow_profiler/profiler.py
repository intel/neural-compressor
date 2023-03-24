# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022 Intel Corporation
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
"""Tensorflow profiler."""

from pathlib import Path
from typing import Any, List, Optional

from neural_compressor.experimental.data.dataloaders.tensorflow_dataloader import (
    TensorflowDataLoader,
)
from neural_compressor.model.tensorflow_model import TensorflowBaseModel
from neural_compressor.profiling.profiler import Profiler as Parent
from neural_compressor.profiling.tensorflow_profiler import utils
from neural_compressor.profiling.tensorflow_profiler.utils import create_tf_config, \
    set_eager_execution
from neural_compressor.utils.create_obj_from_config import create_dataloader


class Profiler(Parent):
    """Tensorflow profiler class."""

    def __init__(
            self,
            model_path: str,
            model: TensorflowBaseModel,
            dataloader: TensorflowDataLoader,
            log_file: Optional[str] = None,
    ) -> None:
        """Initialize profiler for specified model."""
        import tensorflow.compat.v1 as tf_v1

        self.model_path: str = model_path
        self.input_nodes: List[str] = model.input_node_names
        self.output_nodes: List[str] = model.output_node_names

        self.original_dataloader = dataloader
        self.dataloader = self.build_dataloader()
        self.input_datatype = tf_v1.dtypes.float32.as_datatype_enum
        self.log_file = log_file

        if log_file is not None:
            profiling_log_file = Path(self.log_file)
            profiling_log_file.parent.mkdir(parents=True, exist_ok=True)

    def profile_model(
            self,
            intra_num_of_threads: int = 1,
            inter_num_of_threads: int = 1,
            num_warmup: int = 10,
    ) -> None:
        """Execute model profiling."""
        import tensorflow.compat.v1 as tf_module
        from tensorflow.python.profiler import model_analyzer, option_builder
        set_eager_execution(self.model_path)

        tf_config = create_tf_config(tf_module, intra_num_of_threads, inter_num_of_threads)
        graph = self.initialize_graph(tf_module)
        run_options = tf_module.RunOptions(trace_level=tf_module.RunOptions.FULL_TRACE)
        run_metadata = tf_module.RunMetadata()

        with tf_module.Session(config=tf_config, graph=graph) as sess:
            output_dict = {
                out_name: graph.get_tensor_by_name(out_name + ":0")
                for out_name in self.output_nodes
            }

            input_tensor = [
                graph.get_tensor_by_name(in_name + ":0") for in_name in self.input_nodes
            ]
            profiler = model_analyzer.Profiler(graph=graph)
            for idx, (inputs, labels) in enumerate(self.dataloader):
                # dataloader should keep the order and len of inputs same with input_tensor
                if len(input_tensor) == 1:
                    input_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
                else:
                    assert len(input_tensor) == len(
                        inputs,
                    ), "inputs len must equal with input_tensor"
                    input_dict = dict(zip(input_tensor, inputs))

                if idx < num_warmup:
                    sess.run(output_dict, feed_dict=input_dict)
                    continue

                profile_step = idx - num_warmup
                sess.run(
                    output_dict,
                    feed_dict=input_dict,
                    options=run_options,
                    run_metadata=run_metadata,
                )

                profiler.add_step(step=profile_step, run_meta=run_metadata)
                if profile_step > 10:
                    break

            profile_op_opt_builder = option_builder.ProfileOptionBuilder()
            profile_op_opt_builder.select(["micros", "occurrence"])
            profile_op_opt_builder.order_by("micros")
            profile_op_opt_builder.with_max_depth(50)
            if self.log_file is not None:
                profile_op_opt_builder.with_file_output(self.log_file)
            profiler.profile_operations(profile_op_opt_builder.build())

    def initialize_graph(self, tf_module: Any) -> Any:
        """Initialize tensorflow model graph."""
        from tensorflow.python.tools import optimize_for_inference_lib

        graph = tf_module.Graph()
        with graph.as_default():
            od_graph_def = tf_module.GraphDef()
            with tf_module.gfile.GFile(self.model_path, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                od_graph_def = utils.delete_assign(od_graph_def)

            input_node_def = self.get_node_by_name(od_graph_def, self.input_nodes[0])

            if "dtype" in input_node_def.attr:
                self.input_datatype = input_node_def.attr["dtype"].type

            od_graph_def = optimize_for_inference_lib.optimize_for_inference(
                od_graph_def,  # inputGraph,
                self.input_nodes,  # an array of the input nodes
                self.output_nodes,  # an array of output nodes
                self.input_datatype,
            )

            tf_module.import_graph_def(od_graph_def, name="")

        return graph


    @staticmethod
    def get_node_by_name(graph_def: Any, node_name: str) -> Any:
        """Get NodeDef from GraphDef by name."""
        for node in graph_def.node:
            if node.name == node_name:
                return node
        raise Exception(f"Node '{node_name}' not found in graph.")

    def build_dataloader(self) -> TensorflowDataLoader:
        """Build dataloader based on config."""
        dataloader_cfg = {
            "batch_size": self.original_dataloader.batch_size,
            "dataset": {
                "dummy_v2": {
                    "input_shape": [list(self.original_dataloader.dataloader.dataset.element_spec[0].shape)],
                    "label_shape": list(self.original_dataloader.dataloader.dataset.element_spec[1].shape)
                }
            },
            "transform": None,
            "filter": None
        }
        dataloader = create_dataloader("tensorflow", dataloader_cfg)
        return dataloader

