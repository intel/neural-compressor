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
"""Onnxrt profiler."""

import os
from pathlib import Path
from typing import Optional

from neural_compressor.data.dataloaders.onnxrt_dataloader import ONNXRTDataLoader
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.profiling.profiler.onnxrt_profiler.utils import create_onnx_config
from neural_compressor.profiling.profiler.profiler import Profiler as Parent


class Profiler(Parent):
    """Tensorflow profiler class."""

    def __init__(
        self,
        model: ONNXModel,
        dataloader: ONNXRTDataLoader,
        log_file: Optional[str] = None,
    ) -> None:
        """Initialize profiler for specified model.

        Args:
            model: model to be profiled
            dataloader: DataLoader object
            log_file: optional path to log file
        """

        self.model = model.model
        self.dataloader = dataloader
        self.log_file = log_file

        if log_file is not None:
            profiling_log_file = Path(self.log_file)
            profiling_log_file.parent.mkdir(parents=True, exist_ok=True)

    def profile_model(
        self,
        intra_num_of_threads: int = 1,
        inter_num_of_threads: int = 1,
        num_warmup: int = 1,
    ) -> None:
        """Execute model profiling.

        Args:
            intra_num_of_threads: number of threads used within an individual op for parallelism
            inter_num_of_threads: number of threads used for parallelism
                                  between independent operations
            num_warmup: number of warmup iterations

        Returns:
            None
        """
        import numpy as np
        import onnxruntime as ort

        graph = self.model
        onnx_options = create_onnx_config(ort, intra_num_of_threads, inter_num_of_threads)
        # Create a profile session
        sess_profile = ort.InferenceSession(graph.SerializePartialToString(), onnx_options)
        input_tensors = sess_profile.get_inputs()

        for _, (inputs, _) in enumerate(self.dataloader):
            if not isinstance(inputs, np.ndarray) and len(input_tensors) != len(inputs):
                raise Exception("Input data number mismatch.")
            if len(input_tensors) == 1:
                input_dict = {input_tensors[0].name: inputs}
            else:
                input_dict = {input_tensor.name: input_data for input_tensor, input_data in zip(input_tensors, inputs)}
            sess_profile.run(None, input_dict)
            break

        profiling_data_path = sess_profile.end_profiling()
        os.replace(profiling_data_path, self.log_file)
