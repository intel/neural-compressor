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
from collections import OrderedDict, UserDict
from pathlib import Path
from typing import Optional

from neural_compressor.data.dataloaders.tensorflow_dataloader import TensorflowDataLoader
from neural_compressor.model.tensorflow_model import TensorflowBaseModel
from neural_compressor.profiling.profiler.profiler import Profiler as Parent


class Profiler(Parent):
    """Tensorflow profiler class."""

    def __init__(
        self,
        model: TensorflowBaseModel,
        dataloader: TensorflowDataLoader,
        log_file: Optional[str] = None,
    ) -> None:
        """Initialize profiler for specified model.

        Args:
            model: model to be profiled
            dataloader: DataLoader object
            log_file: optional path to log file
        """
        import tensorflow.compat.v1 as tf_v1

        self.model = model

        self.dataloader = dataloader
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
        """ "Execute model profiling.

        Args:
            intra_num_of_threads: number of threads used within an individual op for parallelism
            inter_num_of_threads: number of threads used for parallelism
                                  between independent operations
            num_warmup: number of warmup iterations

        Returns:
            None
        """
        import tensorflow.compat.v1 as tf_v1
        from tensorflow.python.profiler import model_analyzer, option_builder

        tf_v1.enable_eager_execution()

        run_options = tf_v1.RunOptions(trace_level=tf_v1.RunOptions.FULL_TRACE)
        run_metadata = tf_v1.RunMetadata()

        profiler = model_analyzer.Profiler()

        input_tensor = self.model.input_tensor
        output_tensor = self.model.output_tensor if len(self.model.output_tensor) > 1 else self.model.output_tensor[0]
        for idx, (inputs, labels) in enumerate(self.dataloader):
            # dataloader should keep the order and len of inputs same with input_tensor
            if len(input_tensor) == 1:
                feed_dict = {}
                if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) or isinstance(inputs, UserDict):
                    for name in inputs:
                        for tensor in input_tensor:
                            pos = tensor.name.rfind(":")
                            t_name = tensor.name if pos < 0 else tensor.name[:pos]
                            if name == t_name:
                                feed_dict[tensor] = inputs[name]
                                break
                else:
                    feed_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
            else:
                assert len(input_tensor) == len(inputs), "inputs len must equal with input_tensor"
                feed_dict = {}
                if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) or isinstance(inputs, UserDict):
                    for name in inputs:
                        for tensor in input_tensor:
                            pos = tensor.name.rfind(":")
                            t_name = tensor.name if pos < 0 else tensor.name[:pos]
                            if name == t_name:
                                feed_dict[tensor] = inputs[name]
                                break
                else:
                    feed_dict = dict(zip(input_tensor, inputs))

            if idx < num_warmup:
                self.model.sess.run(output_tensor, feed_dict)
                continue

            profile_step = idx - num_warmup
            self.model.sess.run(
                output_tensor,
                feed_dict,
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
