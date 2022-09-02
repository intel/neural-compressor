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

from typing import Any, List

from neural_compressor.conf.dotdict import DotDict
from neural_compressor.experimental.data.dataloaders.tensorflow_dataloader import (
    TensorflowDataLoader,
)
from neural_compressor.utils.create_obj_from_config import create_dataloader
from neural_compressor.ux.components.config_generator.profiling_config_generator import (
    ProfilingConfigGenerator,
)
from neural_compressor.ux.components.profiling.profiler import Profiler as Parent
from neural_compressor.ux.components.profiling.tensorflow_profiler import utils
from neural_compressor.ux.utils.exceptions import NotFoundException
from neural_compressor.ux.utils.logger import log


class Profiler(Parent):
    """Tensorflow profiler class."""

    def __init__(self, profiling_data: dict) -> None:
        """Initialize profiler for specified model."""
        import tensorflow.compat.v1 as tf_v1

        self.model_path: str = profiling_data["model_path"]
        self.batch_size: int = profiling_data["batch_size"]
        self.dataloader_data: dict = {
            "dataset": profiling_data["dataset"],
            "transforms": profiling_data.get("transforms", None),
            "filter": profiling_data.get("filter", None),
            "metric": profiling_data.get("metric", None),
        }
        self.model_domain: str = profiling_data["model_domain"]
        self.model_domain_flavour: str = profiling_data["model_domain_flavour"]
        self.input_nodes: List[str] = profiling_data["model_inputs"]
        self.output_nodes: List[str] = profiling_data["model_outputs"]
        self.num_threads: int = profiling_data["num_threads"]
        self.dataloader = self.build_dataloader()
        self.input_datatype = tf_v1.dtypes.float32.as_datatype_enum

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

    def create_tf_config(self, tf_module: Any) -> Any:
        """Create tensorflow config."""
        config = tf_module.ConfigProto()
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = self.num_threads
        config.inter_op_parallelism_threads = 1
        return config

    def profile_model(self, num_warmup: int = 10, batch_size: int = 1) -> None:
        """Execute model profiling."""
        import tensorflow.compat.v1 as tf_module
        from tensorflow.python.profiler import model_analyzer, option_builder

        tf_config = self.create_tf_config(tf_module)
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
            profiler.profile_operations(profile_op_opt_builder.build())

    @staticmethod
    def get_node_by_name(graph_def: Any, node_name: str) -> Any:
        """Get NodeDef from GraphDef by name."""
        for node in graph_def.node:
            if node.name == node_name:
                return node
        raise NotFoundException()

    @staticmethod
    def convert_nodes_to_list(nodes: str) -> List[str]:
        """Convert string node into list of nodes."""
        return [node.strip() for node in nodes.split(",")]

    def build_dataloader(self) -> TensorflowDataLoader:
        """Build dataloader based on config."""
        dataset_type = ""
        try:
            dataset_type = list(self.dataloader_data.get("dataset", {}).keys())[0]
        except IndexError:
            log.debug("Could not get dataset type.")
        config_generator = ProfilingConfigGenerator(
            workload_directory="",
            configuration_path="",
            data={
                "framework": "tensorflow",
                "model": {
                    "name": "",
                    "input_graph": self.model_path,
                    "domain": self.model_domain,
                    "domain_flavour": self.model_domain_flavour,
                    "input_nodes": self.input_nodes,
                    "output_nodes": self.output_nodes,
                },
                "batch_size": self.batch_size,
                "dataloader": self.dataloader_data,
                "num_threads": self.num_threads,
                "dataset_type": dataset_type,
            },
        )
        dataloader_config = config_generator.generate_dataloader_config()
        dataloader_cfg = DotDict(
            dataloader_config.serialize(),
        )
        dataloader = create_dataloader("tensorflow", dataloader_cfg)
        return dataloader
