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
"""Tensorflow profiler."""

from typing import Any, List

import tensorflow.compat.v1 as tf_v1
from tensorflow.python.profiler import model_analyzer, option_builder
from tensorflow.python.tools import optimize_for_inference_lib

import neural_compressor.ux.components.profiling.tensorflow_profiler.utils as utils
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.experimental.data.dataloaders.tensorflow_dataloader import (
    TensorflowDataLoader,
)
from neural_compressor.utils.create_obj_from_config import create_dataloader
from neural_compressor.ux.components.profiling.profiler import Profiler as Parent
from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.utils.workload.workload import Workload


class Profiler(Parent):
    """Tensorflow profiler class."""

    def __init__(self, workload_id: str, model_path: str) -> None:
        """Initialize profiler for specified model."""
        workdir = Workdir(request_id=workload_id, overwrite=False)
        self.workload: Workload = workdir.get_workload_object()
        self.model_name: str = self.workload.model_name
        self.model_path: str = model_path
        self.input_nodes: List[str] = self.workload.input_nodes  # type: ignore
        self.output_nodes: List[str] = self.workload.output_nodes  # type: ignore
        self.dataloader = self.build_dataloader()

    @property
    def num_threads(self) -> int:
        """Get number of threads for profiling."""
        if (
            self.workload.config.evaluation
            and self.workload.config.evaluation.performance
            and self.workload.config.evaluation.performance.configs
            and self.workload.config.evaluation.performance.configs.intra_num_of_threads
        ):
            return self.workload.config.evaluation.performance.configs.intra_num_of_threads
        return 1

    def initialize_graph(self) -> Any:
        """Initialize tensorflow model graph."""
        graph = tf_v1.Graph()
        with graph.as_default():
            od_graph_def = tf_v1.GraphDef()
            with tf_v1.gfile.GFile(self.model_path, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                od_graph_def = utils.delete_assign(od_graph_def)

            od_graph_def = optimize_for_inference_lib.optimize_for_inference(
                od_graph_def,  # inputGraph,
                self.input_nodes,  # an array of the input nodes
                self.output_nodes,  # an array of output nodes
                tf_v1.dtypes.float32.as_datatype_enum,
            )

            tf_v1.import_graph_def(od_graph_def, name="")

        return graph

    def create_tf_config(self) -> Any:
        """Create tensorflow config."""
        config = tf_v1.ConfigProto()
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = self.num_threads
        config.inter_op_parallelism_threads = 1
        return config

    def profile_model(self, num_warmup: int = 10, batch_size: int = 1) -> None:
        """Execute model profiling."""
        tf_config = self.create_tf_config()
        graph = self.initialize_graph()
        run_options = tf_v1.RunOptions(trace_level=tf_v1.RunOptions.FULL_TRACE)
        run_metadata = tf_v1.RunMetadata()

        with tf_v1.Session(config=tf_config, graph=graph) as sess:
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

    def build_dataloader(self) -> TensorflowDataLoader:
        """Build dataloader based on config."""
        if not (
            self.workload.config.evaluation
            and self.workload.config.evaluation.performance
            and self.workload.config.evaluation.performance.dataloader
        ):
            raise Exception("Could not find performance dataloader.")
        dataloader_cfg = DotDict(
            self.workload.config.evaluation.performance.dataloader.serialize(),
        )
        dataloader = create_dataloader("tensorflow", dataloader_cfg)
        return dataloader
