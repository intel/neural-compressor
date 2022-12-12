#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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

"""Utility functions to export model from PyTorch/TensorFlow to ONNX."""

import numpy as np
from neural_compressor.utils.utility import LazyImport

ort = LazyImport('onnxruntime')
ortq = LazyImport('onnxruntime.quantization')


def ONNX2Numpy_dtype(onnx_node_type):
    """Get Numpy data type from onnx data type.

    Args:
        onnx_node_type (str): data type description.

    Returns:
        dtype: numpy data type
    """
    # Only record sepcial data type
    ONNX2Numpy_dtype_mapping = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
    }
    if onnx_node_type in ONNX2Numpy_dtype_mapping:
        dtype = ONNX2Numpy_dtype_mapping[onnx_node_type]
        return dtype
    else:
        tmp = onnx_node_type.lstrip('tensor(').rstrip(')')
        dtype = eval(f'np.{tmp}')
        return dtype


class DummyDataReader(ortq.CalibrationDataReader):
    """Build dummy datareader for onnx static quantization."""

    def __init__(self, fp32_onnx_path):
        """Initialize data reader.

        Args:
            fp32_onnx_path (str): path to onnx file
        """
        session = ort.InferenceSession(fp32_onnx_path, None)
        input_tensors = session.get_inputs()
        input = {}
        for node in input_tensors:
            shape = []
            for dim in node.shape:
                shape.append(dim if isinstance(dim, int) else 1)
            dtype = ONNX2Numpy_dtype(node.type)
            input[node.name] = np.ones(shape).astype(dtype)
        self.data = [input]
        self.data = iter(self.data)

    def get_next(self):
        """Generate next data."""
        return next(self.data, None)
